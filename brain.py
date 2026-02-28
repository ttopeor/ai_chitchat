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
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx

import config
from memory import MemoryEntry, MemoryManager
from screen import ScreenCapture
from vision import CameraCapture

# ANSI
_C = "\033[35m"   # magenta for brain logs
_R = "\033[0m"

# Ollama native API base (strip /v1 from OpenAI-compat URL)
_OLLAMA_BASE = config.LLM_BASE_URL.removesuffix("/v1").removesuffix("/")

# Log directory
_LOG_DIR = Path("logs/brain")

# Token reserves (read from config)
_IMAGE_TOKEN_RESERVE = config.BRAIN_IMAGE_TOKEN_RESERVE
_OUTPUT_TOKEN_RESERVE = config.BRAIN_OUTPUT_TOKEN_RESERVE


def _estimate_tokens(text: str) -> int:
    """Token count estimate for Qwen2.5 models (~1.3x safety margin).

    Qwen2.5 tokenizer: CJK ≈ 0.73 tok/char, ASCII ≈ 0.25 tok/char.
    We use 1.0/0.3 for ~1.3x overestimate to avoid overflow.
    """
    if not text:
        return 0
    n_cjk = sum(1 for c in text if ord(c) > 0x2E7F)
    n_other = len(text) - n_cjk
    return int(n_cjk * 1.0 + n_other * 0.3) + 4


@dataclass
class ContextBrief:
    """Digested context that the 7b model reads before replying."""
    scene: str = ""                # what the camera sees right now
    memories: str = ""             # relevant memories, pre-formatted
    mood_hint: str = ""            # assessment of the user's current state
    speak_directive: str = "LISTEN"  # LISTEN / RESPOND / INITIATE:intent
    suggested_topics: str = ""     # what to talk about
    conversation_guide: str = ""   # actionable guidance for the 7b based on conversation context
    memory_note: str = ""          # anything worth remembering from recent chat
    media_playing: bool = False    # is media (video/music) playing on screen?
    updated_at: float = 0.0


class BrainEngine:
    def __init__(
        self,
        model: str,
        camera: CameraCapture | None,
        screen: ScreenCapture | None,
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
        self._screen = screen
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
        self._last_memory_extract_time: float = 0.0
        self._prev_scene: str = ""

        # Conversation boundary tracking
        self._conversation_history_snapshot: list[dict] | None = None

        # Dynamic background (synthesized from memory at startup, refreshed on new memories)
        self._dynamic_background: str = ""

        # Media audio buffer (STT transcriptions captured during media playback)
        self._media_audio_buffer: list[str] = []
        self._last_media_memory_time: float = 0.0

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
        # 1. Name mentioned → always respond
        name_variants = ("小悠", "小優", "小尤", "小游", "小由", "小油")
        if any(n in text for n in name_variants):
            return True

        # 2. Brain hasn't had its first think yet — be generous, respond
        if self._brief.updated_at == 0.0:
            return True

        # 3. Media playing → only respond if brain says RESPOND
        #    (skip conversation timeout to filter out media audio)
        if self._brief.media_playing:
            return self._brief.speak_directive == "RESPOND"

        # 4. Active conversation (recent exchange within timeout)
        now = time.monotonic()
        if now - self._last_bot_speech_time < self._conversation_timeout:
            return True

        # 5. Current directive from brain
        if self._brief.speak_directive == "RESPOND":
            return True

        return False

    def is_media_playing(self) -> bool:
        """Check if the brain detects media playing on screen."""
        return self._brief.media_playing

    def record_media_audio(self, text: str) -> None:
        """Buffer transcribed media audio for later memory extraction."""
        self._media_audio_buffer.append(text)
        print(f"  (media audio buffered, {len(self._media_audio_buffer)} segments)")

    async def think_once(self) -> None:
        """Run a single think cycle (called during startup)."""
        try:
            await self._think()
        except Exception as e:
            print(f"[Brain] initial think error: {e}")

    def get_dynamic_background(self) -> str:
        """Return the current dynamic background text for SYSTEM_PROMPT."""
        return self._dynamic_background

    async def synthesize_dynamic_background(self) -> None:
        """Synthesize a dynamic user background from high-importance memories.

        Called at startup and after new memories are extracted.
        Uses LLM to condense memory entries into a natural paragraph.
        Falls back to formatted memory list if LLM call fails.
        """
        if not self._memory:
            return

        profile_memories = self._memory.get_profile_memories()
        if not profile_memories:
            self._dynamic_background = ""
            return

        # Format memories for the synthesis prompt
        mem_lines = [
            f"- [{m.category}] {m.content}（{m.source}）"
            for m in profile_memories
        ]
        mem_text = "\n".join(mem_lines)

        prompt = config.PROFILE_SYNTHESIS_PROMPT.format(memories=mem_text)

        try:
            result, _ = await self._call_ollama(prompt)
            if result and len(result.strip()) > 5:
                self._dynamic_background = result.strip()
                print(f"[Brain] Dynamic background synthesized ({len(self._dynamic_background)} chars)")
            else:
                # Fallback: use raw memory list
                self._dynamic_background = mem_text
                print(f"[Brain] Dynamic background fallback (raw memories)")
        except Exception as e:
            print(f"[Brain] Dynamic background synthesis error: {e}")
            # Fallback: use raw memory list
            self._dynamic_background = mem_text

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
            # Skip thinking while bot is speaking — saves GPU cycles
            # and prevents stale RESPOND directives during playback
            if self._get_bot_speaking():
                await asyncio.sleep(1.0)
                continue

            # Dynamic interval based on conversation activity
            now = time.monotonic()
            last_speech = max(
                self._last_user_speech_time,
                self._last_bot_speech_time,
            )

            if last_speech == 0.0:
                # No speech yet — wait at normal interval
                await asyncio.sleep(self._interval)
            else:
                silence = now - last_speech
                if silence < self._conversation_timeout:
                    # Active conversation — run continuously
                    await asyncio.sleep(0.5)
                elif silence < 120:
                    # Recent conversation ended — normal interval
                    await asyncio.sleep(self._interval)
                else:
                    # Long idle — slow down
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
        screen_b64 = self._screen.get_latest_frame_b64() if self._screen else None
        images: list[str] = []
        if frame_b64:
            images.append(frame_b64)
        if screen_b64:
            images.append(screen_b64)
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

        # Build the brain prompt (image-aware token budgeting)
        prompt = self._build_brain_prompt(
            recent_transcript, memory_text,
            silence_duration, autonomous_gap,
            num_images=len(images),
        )

        # Save input images to log
        if frame_b64:
            img_path = _LOG_DIR / f"{log_prefix}_input.jpg"
            img_path.write_bytes(base64.b64decode(frame_b64))
        if screen_b64:
            img_path = _LOG_DIR / f"{log_prefix}_screen.jpg"
            img_path.write_bytes(base64.b64decode(screen_b64))

        # Call model via native Ollama API
        raw, thinking = await self._call_ollama(prompt, images or None)
        if raw is None:
            return

        # Parse structured output
        brief = self._parse_brain_output(raw)
        elapsed = time.perf_counter() - t0

        # Console summary
        image_tok = 0
        if frame_b64:
            image_tok += _IMAGE_TOKEN_RESERVE
        if screen_b64:
            image_tok += config.SCREEN_IMAGE_TOKEN_RESERVE
        prompt_tokens = _estimate_tokens(prompt) + image_tok
        guide_preview = brief.conversation_guide[:80] if brief.conversation_guide else ""
        print(f"[Brain] #{self._think_count} {_C}{elapsed:.1f}s{_R} "
              f"| ~{prompt_tokens}tok "
              f"| {brief.speak_directive} "
              f"| {brief.scene[:50]}")
        if guide_preview:
            print(f"  guide: {guide_preview}")
        if thinking:
            print(f"  think: {thinking[:100]}{'…' if len(thinking) > 100 else ''}")

        # Save full log to file
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "cycle": self._think_count,
            "elapsed_s": round(elapsed, 2),
            "input": {
                "model": self._model,
                "options": {"num_ctx": config.MODEL_NUM_CTX},
                "think": False,
                "prompt": prompt,
                "has_image": bool(frame_b64),
                "has_screen": bool(screen_b64),
                "image_file": f"{log_prefix}_input.jpg" if frame_b64 else None,
                "screen_file": f"{log_prefix}_screen.jpg" if screen_b64 else None,
                "token_estimates": {
                    "prompt": _estimate_tokens(prompt),
                    "image_reserve": image_tok,
                    "output_reserve": _OUTPUT_TOKEN_RESERVE,
                    "total_input": prompt_tokens,
                    "budget": config.MODEL_NUM_CTX,
                },
            },
            "output": {
                "raw": raw,
                "thinking": thinking,
                "tokens_est": _estimate_tokens(raw),
                "thinking_tokens_est": _estimate_tokens(thinking) if thinking else 0,
                "parsed": {
                    "scene": brief.scene,
                    "mood_hint": brief.mood_hint,
                    "speak_directive": brief.speak_directive,
                    "conversation_guide": brief.conversation_guide,
                    "memory_note": brief.memory_note,
                    "media_playing": brief.media_playing,
                },
            },
        }
        log_path = _LOG_DIR / f"{log_prefix}_think.json"
        log_path.write_text(json.dumps(log_data, ensure_ascii=False, indent=2))

        # Handle memory notes (cooldown: at most once per 60s)
        if brief.memory_note and brief.memory_note != "无" and self._memory:
            if time.monotonic() - self._last_memory_extract_time > 60:
                self._last_memory_extract_time = time.monotonic()
                await self._maybe_extract_memories()

        # Extract media memories when media stops or buffer is full
        if self._media_audio_buffer and self._memory:
            buffer_full = len(self._media_audio_buffer) >= 20
            media_stopped = not brief.media_playing
            cooldown_ok = time.monotonic() - self._last_media_memory_time > 60
            if (media_stopped or buffer_full) and cooldown_ok:
                self._last_media_memory_time = time.monotonic()
                await self._extract_media_memories()

        # Handle autonomous speech (INITIATE directive)
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

        self._prev_scene = brief.scene

        async with self._lock:
            self._brief = brief

    # ── Ollama native API call ───────────────────────────────────────────────

    async def _call_ollama(
        self, prompt: str, images: list[str] | None = None
    ) -> tuple[str | None, str]:
        """Call the brain model via Ollama native /api/chat.

        Returns (content, thinking) tuple. thinking contains the model's
        chain-of-thought reasoning (empty string if none).
        """
        msg: dict = {"role": "user", "content": prompt}
        if images:
            msg["images"] = images
        messages = [msg]

        body = {
            "model": self._model,
            "messages": messages,
            "options": {"num_ctx": config.MODEL_NUM_CTX},
            "stream": False,
            "think": False,
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
                msg = data.get("message", {})
                content = msg.get("content", "").strip()
                thinking = msg.get("thinking", "").strip()
                return content, thinking
        except Exception as e:
            print(f"[Brain] LLM error: {e}")
            return None, ""

    # ── brain prompt ──────────────────────────────────────────────────────────

    def _build_brain_prompt(
        self,
        transcript: str,
        memories: str,
        silence: float,
        autonomous_gap: float,
        num_images: int = 0,
    ) -> str:
        """Build brain prompt, dynamically trimming inputs to fit BRAIN_NUM_CTX.

        Token budget breakdown:
          BRAIN_NUM_CTX - image_reserve - output_reserve - template_fixed = variable_budget
          variable_budget is split: transcript 50%, memories 25%, prev_scene 15%, slack 10%
        """
        image_tokens = 0
        if num_images >= 1:
            image_tokens += _IMAGE_TOKEN_RESERVE                # camera
        if num_images >= 2:
            image_tokens += config.SCREEN_IMAGE_TOKEN_RESERVE   # screen
        text_budget = config.MODEL_NUM_CTX - image_tokens - _OUTPUT_TOKEN_RESERVE

        # Estimate template fixed text (placeholders → empty)
        template_fixed_tokens = _estimate_tokens(
            config.BRAIN_PROMPT_TEMPLATE.format(
                prev_scene="", transcript="", memories="",
                silence=0, autonomous_gap=0,
            )
        )
        variable_budget = max(text_budget - template_fixed_tokens, 200)

        # Prepare prev_scene (cap at 15% of variable budget)
        max_prev_tokens = int(variable_budget * 0.15)
        prev = self._prev_scene if self._prev_scene else "（第一次观察）"
        while _estimate_tokens(prev) > max_prev_tokens and len(prev) > 20:
            prev = prev[:int(len(prev) * 0.7)] + "..."

        # Prepare transcript (50% of variable budget, keep most recent)
        max_transcript_tokens = int(variable_budget * 0.5)
        if transcript:
            while _estimate_tokens(transcript) > max_transcript_tokens and len(transcript) > 50:
                lines = transcript.split("\n")
                if len(lines) > 2:
                    transcript = "...\n" + "\n".join(lines[2:])
                else:
                    transcript = "..." + transcript[-int(len(transcript) * 0.6):]

        # Prepare memories (25% of variable budget)
        max_memory_tokens = int(variable_budget * 0.25)
        if memories:
            while _estimate_tokens(memories) > max_memory_tokens and len(memories) > 30:
                lines = memories.split("\n")
                if len(lines) > 1:
                    memories = "\n".join(lines[:-1]) + "\n..."
                else:
                    memories = memories[:int(len(memories) * 0.6)] + "..."

        prompt = config.BRAIN_PROMPT_TEMPLATE.format(
            prev_scene=prev,
            transcript=transcript if transcript else "（最近没有对话）",
            memories=memories if memories else "（暂无记忆）",
            silence=silence,
            autonomous_gap=autonomous_gap,
        )

        # Final safety: hard-truncate if still over budget
        total = _estimate_tokens(prompt)
        if total > text_budget:
            # Keep template structure but truncate the assembled text
            target_chars = int(text_budget / 1.5)
            prompt = prompt[:target_chars] + "\n...(截断)"

        return prompt

    # ── parse brain output ────────────────────────────────────────────────────

    def _parse_brain_output(self, raw: str) -> ContextBrief:
        def extract(tag: str) -> str:
            pattern = rf"\[{tag}\]\s*(.*)"
            m = re.search(pattern, raw)
            return m.group(1).strip() if m else ""

        media_str = extract("MEDIA")
        media_playing = media_str.upper().startswith("Y") if media_str else False

        return ContextBrief(
            scene=extract("SCENE") or self._prev_scene,
            memories=self._brief.memories,  # keep existing until updated
            mood_hint=extract("MOOD"),
            speak_directive=extract("DIRECTIVE") or "LISTEN",
            suggested_topics=extract("TOPICS"),  # kept for backward compat
            conversation_guide=extract("GUIDE"),
            memory_note=extract("MEMORY_NOTE"),
            media_playing=media_playing,
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

            # Refresh dynamic background with new memories
            await self.synthesize_dynamic_background()

    # ── media memory extraction ────────────────────────────────────────────

    async def _extract_media_memories(self) -> None:
        """Extract memories from buffered media audio transcriptions."""
        if not self._media_audio_buffer or not self._memory:
            return

        audio_text = "\n".join(self._media_audio_buffer)
        self._media_audio_buffer.clear()

        # Truncate if too long (keep last ~4000 chars)
        if len(audio_text) > 4000:
            audio_text = "...\n" + audio_text[-4000:]

        prompt = (
            "/no_think\n"
            "以下是对方正在观看的视频/媒体的音频转录内容:\n\n"
            f"{audio_text}\n\n"
            "请总结这个视频/媒体的主题和关键内容，用一两句话概括他在看什么。\n"
            "用JSON数组格式回复，每条记忆包含:\n"
            '{"category": "media", "content": "他在看一个关于...的视频/节目", '
            '"keywords": ["关键词1", "关键词2"], "importance": 2}\n\n'
            "如果内容太碎片化无法总结，回复空数组 []\n"
            "只回复JSON，不要其他文字。"
        )

        try:
            result = await self._memory._call_ollama(prompt)
            if result is None:
                return

            if result.startswith("```"):
                result = result.split("\n", 1)[1].rsplit("```", 1)[0]

            entries_data = json.loads(result)
            if not isinstance(entries_data, list) or not entries_data:
                return

            now_str = datetime.now().isoformat()
            source = f"观看视频于{datetime.now().strftime('%m月%d日%H:%M')}"

            new_entries: list[MemoryEntry] = []
            for d in entries_data:
                if not isinstance(d, dict) or "content" not in d:
                    continue
                new_entries.append(MemoryEntry(
                    id=uuid.uuid4().hex[:8],
                    timestamp=now_str,
                    category="media",
                    content=d["content"],
                    keywords=d.get("keywords", []),
                    importance=min(3, max(1, int(d.get("importance", 2)))),
                    source=source,
                ))

            new_entries = self._memory._deduplicate(new_entries)
            if new_entries:
                self._memory._append(new_entries)
                self._memory._memories.extend(new_entries)
                print(f"[Brain] Extracted {len(new_entries)} media memories:")
                for m in new_entries:
                    print(f"  [media] {m.content}")

        except Exception as e:
            print(f"[Brain] media memory extraction error: {e}")

    # ── helpers ───────────────────────────────────────────────────────────────

    def _format_recent_transcript(self, max_entries: int = config.BRAIN_TRANSCRIPT_ENTRIES) -> str:
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

    def _trim_recent(self, max_age: float = config.BRAIN_RECENT_MAX_AGE, max_count: int = config.BRAIN_RECENT_MAX_COUNT) -> None:
        cutoff = time.monotonic() - max_age
        self._recent_user_texts = [
            (t, s) for t, s in self._recent_user_texts if t > cutoff
        ][-max_count:]
        self._recent_bot_texts = [
            (t, s) for t, s in self._recent_bot_texts if t > cutoff
        ][-max_count:]
