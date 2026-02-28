"""
BrainEngine — the 72b "System 2" background thinker.

Runs a periodic loop that:
  1. Observes the camera (VLM scene description)
  2. Reviews recent conversation
  3. Retrieves relevant memories
  4. Produces a ContextBrief for the 7b conversation model
  5. Decides whether to initiate autonomous speech
  6. Triggers memory extraction at conversation boundaries

Uses Ollama native API (/api/chat) instead of OpenAI-compatible API
to ensure num_ctx is always passed — prevents 128k default VRAM explosion.

Logs every think cycle to logs/brain/ for debugging.
"""

import asyncio
import base64
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx

import config
from memory import MemoryManager
from vision import CameraCapture

# ANSI
_C = "\033[35m"   # magenta for brain logs
_R = "\033[0m"

# Ollama native API base (strip /v1 from OpenAI-compat URL)
_OLLAMA_BASE = config.LLM_BASE_URL.removesuffix("/v1").removesuffix("/")

# Log directory
_LOG_DIR = Path("logs/brain")


@dataclass
class ContextBrief:
    """Digested context that the 7b model reads before replying."""
    scene: str = ""                # what the camera sees right now
    memories: str = ""             # relevant memories, pre-formatted
    mood_hint: str = ""            # assessment of the user's current state
    speak_directive: str = "LISTEN"  # LISTEN / RESPOND / INITIATE:intent
    suggested_topics: str = ""     # what to talk about
    memory_note: str = ""          # anything worth remembering from recent chat
    updated_at: float = 0.0


class BrainEngine:
    def __init__(
        self,
        model: str,
        camera: CameraCapture | None,
        memory: MemoryManager | None,
        get_history: callable,
        get_bot_speaking: callable,
        on_autonomous_speech: callable,   # async callable(intent: str)
        interval: float = 20.0,
        autonomous_cooldown: float = 120.0,
        conversation_timeout: float = 30.0,
    ):
        self._model = model
        self._camera = camera
        self._memory = memory
        self._get_history = get_history
        self._get_bot_speaking = get_bot_speaking
        self._on_autonomous_speech = on_autonomous_speech

        self._interval = interval
        self._autonomous_cooldown = autonomous_cooldown
        self._conversation_timeout = conversation_timeout

        self._brief = ContextBrief()
        self._lock = asyncio.Lock()

        # Observation tracking
        self._recent_user_texts: list[tuple[float, str]] = []   # (mono_ts, text)
        self._recent_bot_texts: list[tuple[float, str]] = []
        self._last_user_speech_time: float = 0.0
        self._last_bot_speech_time: float = 0.0
        self._last_autonomous_time: float = 0.0
        self._prev_scene: str = ""

        # Conversation boundary tracking
        self._conversation_history_snapshot: list[dict] | None = None

        # Think cycle counter (for log filenames)
        self._think_count = 0

        # Ensure log directory exists
        _LOG_DIR.mkdir(parents=True, exist_ok=True)

    # ── public API (called from main.py) ──────────────────────────────────────

    def record_user_speech(self, text: str) -> None:
        now = time.monotonic()
        self._last_user_speech_time = now
        self._recent_user_texts.append((now, text))
        self._trim_recent()

    def record_bot_speech(self, text: str) -> None:
        now = time.monotonic()
        self._last_bot_speech_time = now
        self._recent_bot_texts.append((now, text))
        self._trim_recent()

    def get_context_brief(self) -> ContextBrief:
        """Read the latest brief (thread-safe via asyncio.Lock is same loop)."""
        return self._brief

    def should_respond(self, text: str) -> bool:
        """Quick heuristic: is this speech directed at 小悠?"""
        # Name mentioned → always respond (handle STT variants: 小悠/小優/小尤/xiǎo yōu)
        name_variants = ("小悠", "小優", "小尤", "小游", "小由", "小油")
        if any(n in text for n in name_variants):
            return True

        # Active conversation (recent exchange within timeout)
        now = time.monotonic()
        if now - self._last_bot_speech_time < self._conversation_timeout:
            return True

        # Brain hasn't had its first think yet — be generous, respond
        if self._brief.updated_at == 0.0:
            return True

        # Current directive from brain
        directive = self._brief.speak_directive
        if directive == "RESPOND":
            return True

        return False

    async def think_once(self) -> None:
        """Run a single think cycle (called during startup)."""
        try:
            await self._think()
        except Exception as e:
            print(f"[Brain] initial think error: {e}")

    # ── main loop ─────────────────────────────────────────────────────────────

    async def brain_loop(self) -> None:
        """Periodic thinking loop with dynamic interval.

        - Active conversation: back-to-back (continuous inference)
        - Idle < 2 min: normal interval (BRAIN_INTERVAL, default 20s)
        - Idle > 2 min: slow mode (60s)
        """
        print(f"[Brain] Started (model={self._model})")
        print(f"[Brain] Logs → {_LOG_DIR.resolve()}/")

        while True:
            # Dynamic interval based on conversation activity
            now = time.monotonic()
            last_speech = max(
                self._last_user_speech_time,
                self._last_bot_speech_time,
                0.1,
            )
            silence = now - last_speech

            if silence < self._conversation_timeout:
                # Active conversation — run continuously, minimal yield
                await asyncio.sleep(0.5)
            elif silence < 120:
                # Recent conversation ended — normal interval
                await asyncio.sleep(self._interval)
            else:
                # Long idle — slow down to conserve resources
                await asyncio.sleep(60.0)

            try:
                await self._think()
            except Exception as e:
                print(f"[Brain] error: {e}")

    # ── core thinking ─────────────────────────────────────────────────────────

    async def _think(self) -> None:
        t0 = time.perf_counter()
        self._think_count += 1
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_prefix = f"{ts}_{self._think_count:04d}"

        # Gather inputs
        frame_b64 = self._camera.get_latest_frame_b64() if self._camera else None
        recent_transcript = self._format_recent_transcript()
        memory_text = ""
        if self._memory:
            recent_topics = [t for _, t in self._recent_user_texts[-3:]]
            query = " ".join(recent_topics)
            memory_text = self._memory.format_for_prompt(query)

        silence_duration = time.monotonic() - max(
            self._last_user_speech_time, self._last_bot_speech_time, 1.0
        )
        autonomous_gap = time.monotonic() - self._last_autonomous_time

        # Build the brain prompt
        prompt = self._build_brain_prompt(
            recent_transcript, memory_text,
            silence_duration, autonomous_gap,
        )

        # Save input image to log
        if frame_b64:
            img_path = _LOG_DIR / f"{log_prefix}_input.jpg"
            img_path.write_bytes(base64.b64decode(frame_b64))

        # Call 72b via native Ollama API (ensures num_ctx is respected)
        raw = await self._call_ollama(prompt, frame_b64)
        if raw is None:
            return

        # Parse structured output
        brief = self._parse_brain_output(raw)
        elapsed = time.perf_counter() - t0

        # Console summary (one line)
        print(f"[Brain] #{self._think_count} {_C}{elapsed:.1f}s{_R} "
              f"| {brief.speak_directive} "
              f"| {brief.scene[:50]}")

        # Save full log to file
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "cycle": self._think_count,
            "elapsed_s": round(elapsed, 2),
            "input": {
                "prompt": prompt,
                "has_image": frame_b64 is not None,
                "image_file": f"{log_prefix}_input.jpg" if frame_b64 else None,
            },
            "output": {
                "raw": raw,
                "parsed": {
                    "scene": brief.scene,
                    "mood_hint": brief.mood_hint,
                    "speak_directive": brief.speak_directive,
                    "suggested_topics": brief.suggested_topics,
                    "memory_note": brief.memory_note,
                },
            },
        }
        log_path = _LOG_DIR / f"{log_prefix}_think.json"
        log_path.write_text(json.dumps(log_data, ensure_ascii=False, indent=2))

        # Handle memory notes
        if brief.memory_note and brief.memory_note != "无" and self._memory:
            await self._maybe_extract_memories()

        # Handle autonomous speech
        if brief.speak_directive.startswith("INITIATE:"):
            intent = brief.speak_directive[len("INITIATE:"):].strip()
            if intent and not self._get_bot_speaking():
                cooldown_ok = (
                    time.monotonic() - self._last_autonomous_time
                    > self._autonomous_cooldown
                )
                if cooldown_ok:
                    self._last_autonomous_time = time.monotonic()
                    asyncio.create_task(self._on_autonomous_speech(intent))

        async with self._lock:
            self._brief = brief

    # ── Ollama native API call ───────────────────────────────────────────────

    async def _call_ollama(
        self, prompt: str, image_b64: str | None
    ) -> str | None:
        """Call the brain model via Ollama native /api/chat with num_ctx.

        Using native API instead of OpenAI-compatible endpoint ensures
        num_ctx is always passed, preventing Ollama from reloading the
        model with default 128k context (which explodes VRAM).
        """
        messages = []

        # Build message with optional image
        if image_b64:
            messages.append({
                "role": "user",
                "content": prompt,
                "images": [image_b64],
            })
        else:
            messages.append({
                "role": "user",
                "content": prompt,
            })

        body = {
            "model": self._model,
            "messages": messages,
            "options": {"num_ctx": config.BRAIN_NUM_CTX},
            "stream": False,
        }

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{_OLLAMA_BASE}/api/chat",
                    json=body,
                    timeout=httpx.Timeout(connect=10, read=120, write=10, pool=10),
                )
                resp.raise_for_status()
                data = resp.json()
                return data.get("message", {}).get("content", "").strip()
        except Exception as e:
            print(f"[Brain] LLM error: {e}")
            return None

    # ── brain prompt ──────────────────────────────────────────────────────────

    def _build_brain_prompt(
        self,
        transcript: str,
        memories: str,
        silence: float,
        autonomous_gap: float,
    ) -> str:
        mem_section = memories if memories else "（暂无记忆）"
        transcript_section = transcript if transcript else "（最近没有对话）"

        return (
            "/no_think\n"
            "你是小悠的内心思维。你通过摄像头和对话记录感知世界，但你不直接说话。\n"
            "你的任务是观察、思考、记忆，然后给'说话的自己'提供指导。\n\n"

            f"最近的对话:\n{transcript_section}\n\n"
            f"你记得的事情:\n{mem_section}\n\n"
            f"距离上次有人说话: {silence:.0f}秒\n"
            f"距离你上次主动开口: {autonomous_gap:.0f}秒\n\n"

            "判断规则:\n"
            "- 如果有人明确在跟你说话，回复 RESPOND\n"
            "- 如果环境里的说话不是对你说的（背景对话、自言自语、跟别人说话），回复 LISTEN\n"
            "- 如果你看到明显变化（有人来了、在做新的事情），可以主动打招呼: INITIATE:意图\n"
            "- 如果很长时间没人说话（超过3分钟）且你有话想说，可以: INITIATE:意图\n"
            "- 大多数时候保持 LISTEN，不要话太多\n"
            "- 如果场景没什么变化，保持 LISTEN\n\n"

            "请严格按以下格式输出（每项一行）:\n"
            "[SCENE] 一句话描述你看到的\n"
            "[MOOD] 一句话判断对方当前状态\n"
            "[DIRECTIVE] LISTEN 或 RESPOND 或 INITIATE:意图\n"
            "[TOPICS] 如果要聊天可以聊什么\n"
            "[MEMORY_NOTE] 对话中值得记住的事（没有就写'无'）\n"
        )

    # ── parse brain output ────────────────────────────────────────────────────

    def _parse_brain_output(self, raw: str) -> ContextBrief:
        def extract(tag: str) -> str:
            pattern = rf"\[{tag}\]\s*(.*)"
            m = re.search(pattern, raw)
            return m.group(1).strip() if m else ""

        return ContextBrief(
            scene=extract("SCENE") or self._prev_scene,
            memories=self._brief.memories,  # keep existing until updated
            mood_hint=extract("MOOD"),
            speak_directive=extract("DIRECTIVE") or "LISTEN",
            suggested_topics=extract("TOPICS"),
            memory_note=extract("MEMORY_NOTE"),
            updated_at=time.monotonic(),
        )

    # ── memory extraction ─────────────────────────────────────────────────────

    async def _maybe_extract_memories(self) -> None:
        """Ask memory manager to extract from current conversation history."""
        if not self._memory:
            return
        history = self._get_history()
        if len([m for m in history if m["role"] == "user"]) < 2:
            return

        new = await self._memory.extract_memories(history)
        if new:
            print(f"[Brain] Extracted {len(new)} memories:")
            for m in new:
                print(f"  [{m.category}] {m.content}")

            # Update brief with fresh memories
            recent_topics = [t for _, t in self._recent_user_texts[-3:]]
            query = " ".join(recent_topics)
            async with self._lock:
                self._brief.memories = self._memory.format_for_prompt(query)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _format_recent_transcript(self, max_entries: int = 10) -> str:
        """Merge recent user and bot texts into a chronological transcript."""
        combined: list[tuple[float, str, str]] = []
        for ts, txt in self._recent_user_texts:
            combined.append((ts, "对方", txt))
        for ts, txt in self._recent_bot_texts:
            combined.append((ts, "小悠", txt))
        combined.sort(key=lambda x: x[0])
        combined = combined[-max_entries:]

        if not combined:
            return ""
        return "\n".join(f"{who}: {txt}" for _, who, txt in combined)

    def _trim_recent(self, max_age: float = 300.0, max_count: int = 20) -> None:
        cutoff = time.monotonic() - max_age
        self._recent_user_texts = [
            (t, s) for t, s in self._recent_user_texts if t > cutoff
        ][-max_count:]
        self._recent_bot_texts = [
            (t, s) for t, s in self._recent_bot_texts if t > cutoff
        ][-max_count:]
