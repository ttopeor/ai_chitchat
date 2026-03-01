"""
VoiceBot — dual-system voice chat with vision, memory, and autonomous speech.

  Single model (qwen3.5:122b) with parallel slots:
    System 1 (conv): real-time conversation — mouth & reflexes
    System 2 (brain): background thinker — vision, memory, context generation

  STT : faster-whisper  (CUDA)
  TTS : ChatTTS          (CUDA)
  VAD : Silero VAD
  CAM : OpenCV           (USB webcam)
"""
import asyncio
import json
import logging
import os
import queue
import re
import signal
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

os.environ['TQDM_DISABLE'] = '1'

# Silence noisy third-party loggers (keep WARNING and above)
for _name in ("faster_whisper", "httpx", "ChatTTS", "onnxruntime"):
    logging.getLogger(_name).setLevel(logging.WARNING)

import ChatTTS
import numpy as np
import sounddevice as sd
import torch
from faster_whisper import WhisperModel
import httpx
import config


# ── Ollama model lifecycle ────────────────────────────────────────────────────

# Strip "/v1" to get native Ollama API base
_OLLAMA_BASE = config.LLM_BASE_URL.removesuffix("/v1").removesuffix("/")


async def _ollama_load(
    model: str,
    keep_alive: int | str,
    num_ctx: int | None = None,
) -> None:
    """Load/pin a model in Ollama VRAM with specific keep_alive and context size.

    Uses /api/chat (streaming) and reads the full response to ensure
    the model is actually loaded before returning.  Passing num_ctx here
    sets the KV-cache size for the loaded instance — critical for VRAM control.
    """
    body: dict = {
        "model": model,
        "keep_alive": keep_alive,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }
    if num_ctx is not None:
        body["options"] = {"num_ctx": num_ctx}
    try:
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{_OLLAMA_BASE}/api/chat",
                json=body,
                timeout=httpx.Timeout(connect=10, read=300, write=10, pool=10),
            ) as resp:
                resp.raise_for_status()
                async for _ in resp.aiter_lines():
                    pass  # drain stream — model loads as we read
    except Exception as e:
        print(f"  [Ollama] load({model}) failed: {e}")


async def _ollama_unload(model: str) -> None:
    """Unload a model from Ollama VRAM immediately.

    Uses /api/generate with keep_alive=0 and no prompt — the official way
    to evict a model without re-loading it with default context.
    """
    body = {"model": model, "keep_alive": 0}
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{_OLLAMA_BASE}/api/generate",
                json=body,
                timeout=httpx.Timeout(connect=10, read=30, write=10, pool=10),
            )
            resp.raise_for_status()
    except Exception as e:
        print(f"  [Ollama] unload({model}) failed: {e}")


from brain import BrainEngine
from memory import MemoryManager
from screen import ScreenCapture
from vision import CameraCapture

# ANSI color for timing output
_C = "\033[36m"   # cyan
_R = "\033[0m"    # reset


# ── conversation logger ─────────────────────────────────────────────────────

_CONV_DETAIL_DIR = Path("logs/conv")
_CONV_DETAIL_DIR.mkdir(parents=True, exist_ok=True)


# ── token estimation ─────────────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    """Token count estimate for Qwen2.5 models (~1.3x safety margin).

    Qwen2.5 tokenizer: CJK ≈ 0.73 tok/char, ASCII ≈ 0.25 tok/char.
    We use 1.0/0.3 for ~1.3x overestimate to avoid overflow.
    """
    if not text:
        return 0
    n_cjk = sum(1 for c in text if ord(c) > 0x2E7F)
    n_other = len(text) - n_cjk
    return int(n_cjk * 1.0 + n_other * 0.3) + 4   # +4 per-message overhead


# ── sentence splitter ─────────────────────────────────────────────────────────

_SENT_RE = re.compile(r'[。！？\n]|(?<=[.!?])[ \t]')
_MAX_BUF = 200   # force-flush if no punctuation found within this many chars


def pop_sentence(buf: str) -> tuple[str, str]:
    """Return (sentence, remainder). sentence=='' means keep accumulating."""
    m = _SENT_RE.search(buf)
    if m:
        return buf[: m.end()].strip(), buf[m.end():]
    if len(buf) > _MAX_BUF:
        for sep in ("，", ",", "、", ";", "；", " "):
            idx = buf.rfind(sep, 0, _MAX_BUF)
            if idx > 0:
                return buf[:idx].strip(), buf[idx + 1:]
        return buf[:_MAX_BUF].strip(), buf[_MAX_BUF:]
    return "", buf


# ── PipeWire AEC routing ──────────────────────────────────────────────────────

ENABLE_AEC       = False  # Set True to route audio through PipeWire Echo Cancellation
ENABLE_INTERRUPT = False  # Set True to allow user to interrupt bot mid-speech


def _setup_aec():
    """Set PipeWire Echo Cancellation nodes as default source/sink."""
    if not ENABLE_AEC:
        print("  AEC disabled (ENABLE_AEC = False)")
        return
    try:
        result = subprocess.run(
            ["wpctl", "status"], capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.splitlines():
            stripped = line.strip()
            if "Echo Cancellation Source" in stripped:
                node_id = stripped.split(".")[0]
                subprocess.run(["wpctl", "set-default", node_id], timeout=5)
                print(f"  AEC source → node {node_id}")
            elif "Echo Cancellation Sink" in stripped:
                node_id = stripped.split(".")[0]
                subprocess.run(["wpctl", "set-default", node_id], timeout=5)
                print(f"  AEC sink   → node {node_id}")
    except FileNotFoundError:
        print("  wpctl not found — PipeWire AEC not configured")
    except Exception as e:
        print(f"  AEC setup failed: {e}")


# ── VoiceBot ──────────────────────────────────────────────────────────────────

class VoiceBot:
    def __init__(self):
        print("Loading Whisper STT…")
        self.asr = WhisperModel(
            config.WHISPER_MODEL,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE,
        )

        print("Loading Silero VAD…")
        self._vad_model, _ = torch.hub.load(
            "snakers4/silero-vad", "silero_vad",
            force_reload=False, trust_repo=True,
        )
        self._vad_model.eval()

        print("Loading ChatTTS…")
        self.tts = ChatTTS.Chat()
        # ChatTTS uses torch.load without weights_only=False; patch for PyTorch 2.6+ compat
        _orig_load = torch.load
        torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "weights_only": False})
        try:
            self.tts.load_models(compile=False)
        finally:
            torch.load = _orig_load
        self.tts_sr = 24000
        self._tts_spk = self.tts.sample_random_speaker(seed=config.CHATTTS_SEED)

        self.history: list[dict] = [
            {"role": "system", "content": config.SYSTEM_PROMPT}
        ]

        # threading.Event so both asyncio and OutputStream callback can read/write
        self.bot_speaking = threading.Event()
        self.interrupt     = threading.Event()

        self._raw_q:   asyncio.Queue[np.ndarray] = asyncio.Queue()
        self.speech_q: asyncio.Queue[np.ndarray] = asyncio.Queue()

        # TTS pipeline queues: text → synthesis → audio → playback
        self._synth_q: queue.Queue[str | None]       = queue.Queue()
        self._audio_q: queue.Queue[np.ndarray | None] = queue.Queue()

        # Concurrency: only one speech generation at a time
        self._speech_lock = asyncio.Lock()

        # Conv API call counter for log filenames
        self._conv_call_count = 0

        # ── Vision ────────────────────────────────────────────────────────────
        self.camera: CameraCapture | None = None
        if config.CAMERA_ENABLED:
            self.camera = CameraCapture(
                device_index=config.CAMERA_DEVICE,
                width=config.CAMERA_WIDTH,
                height=config.CAMERA_HEIGHT,
                interval=config.CAMERA_INTERVAL,
            )

        # ── Screen Capture (remote Windows screenshot) ───────────────────────
        self.screen: ScreenCapture | None = None
        if config.SCREEN_ENABLED:
            self.screen = ScreenCapture(
                url=config.SCREEN_URL,
                interval=config.SCREEN_INTERVAL,
            )

        # ── Memory ────────────────────────────────────────────────────────────
        self.memory: MemoryManager | None = None
        if config.MEMORY_ENABLED:
            self.memory = MemoryManager(
                model=config.BRAIN_MODEL,
                storage_dir=config.MEMORY_DIR,
                storage_file=config.MEMORY_FILE,
                max_context=config.MEMORY_MAX_CONTEXT,
                min_turns=config.MEMORY_EXTRACT_MIN_TURNS,
            )

        # ── Brain (72b background thinker) ────────────────────────────────────
        self.brain: BrainEngine | None = None
        if config.BRAIN_ENABLED:
            self.brain = BrainEngine(
                model=config.BRAIN_MODEL,
                camera=self.camera,
                screen=self.screen,
                memory=self.memory,
                get_history=lambda: self.history,
                get_bot_speaking=lambda: self.bot_speaking.is_set(),
                on_autonomous_speech=self._handle_autonomous_speech,
                interval=config.BRAIN_INTERVAL,
                autonomous_cooldown=config.AUTONOMOUS_COOLDOWN,
                conversation_timeout=config.CONVERSATION_TIMEOUT,
            )

    # ── VAD ───────────────────────────────────────────────────────────────────

    def _vad_prob(self, chunk: np.ndarray) -> float:
        with torch.no_grad():
            return self._vad_model(
                torch.from_numpy(chunk), config.MIC_SAMPLE_RATE
            ).item()

    # ── mic / VAD loop ────────────────────────────────────────────────────────

    async def mic_loop(self) -> None:
        loop      = asyncio.get_running_loop()
        CHUNK     = 512   # Silero VAD requires exactly 512 samples @ 16 kHz
        SILENCE_N = int(config.SILENCE_S * config.MIC_SAMPLE_RATE / CHUNK)

        buf: list[np.ndarray] = []
        in_speech = False
        silence_n = 0
        interrupt_count = 0
        bot_was_speaking = False
        cooldown_n = 0
        COOLDOWN_FRAMES = int(0.5 * config.MIC_SAMPLE_RATE / CHUNK)

        def cb(indata, *_):
            loop.call_soon_threadsafe(
                self._raw_q.put_nowait,
                indata[:, 0].copy().astype(np.float32),
            )

        with sd.InputStream(
            samplerate=config.MIC_SAMPLE_RATE,
            channels=1,
            blocksize=CHUNK,
            dtype="float32",
            device=config.MIC_DEVICE,
            callback=cb,
        ):
            print("\nReady — speak now.\n")
            while True:
                chunk = await self._raw_q.get()

                # Cooldown after bot finishes speaking — discard residual echo
                bot_speaking_now = self.bot_speaking.is_set()
                if bot_was_speaking and not bot_speaking_now:
                    cooldown_n = COOLDOWN_FRAMES
                    buf.clear()
                    in_speech = False
                    silence_n = 0
                bot_was_speaking = bot_speaking_now

                if cooldown_n > 0:
                    cooldown_n -= 1
                    continue

                prob = self._vad_prob(chunk)

                # Adaptive threshold: higher when bot speaks to reject echo
                threshold = (
                    config.VAD_THRESHOLD_INTERRUPT
                    if bot_speaking_now
                    else config.VAD_THRESHOLD
                )
                is_speech = prob > threshold

                if is_speech:
                    if not in_speech:
                        if bot_speaking_now:
                            if not ENABLE_INTERRUPT:
                                continue
                            interrupt_count += 1
                            if interrupt_count >= config.INTERRUPT_MIN_FRAMES:
                                in_speech = True
                                self.interrupt.set()
                                interrupt_count = 0
                        else:
                            in_speech = True
                    buf.append(chunk)
                    silence_n = 0
                elif in_speech:
                    buf.append(chunk)
                    silence_n += 1
                    if silence_n >= SILENCE_N:
                        await self.speech_q.put(np.concatenate(buf))
                        buf.clear()
                        in_speech = False
                        silence_n = 0
                else:
                    interrupt_count = 0
                    buf.clear()

    # ── STT ───────────────────────────────────────────────────────────────────

    async def _transcribe(self, audio: np.ndarray) -> str:
        loop = asyncio.get_running_loop()

        def _run() -> str:
            segs, _ = self.asr.transcribe(audio, beam_size=5, vad_filter=True, language="zh")
            return "".join(s.text for s in segs).strip()

        return await loop.run_in_executor(None, _run)

    # ── TTS pipeline workers ─────────────────────────────────────────────────

    def _synthesize_worker(self) -> None:
        """Pull text from _synth_q, synthesize, push audio to _audio_q."""
        while True:
            text = self._synth_q.get()
            if text is None:
                self._audio_q.put(None)
                break
            if self.interrupt.is_set():
                self._drain_queue(self._synth_q)
                self._audio_q.put(None)
                break

            params = {
                "spk_emb": self._tts_spk,
                "temperature": config.CHATTTS_TEMPERATURE,
                "top_P": 0.7,
                "top_K": 20,
            }
            wavs = self.tts.infer([text], skip_refine_text=False, params_infer_code=params)

            if self.interrupt.is_set():
                self._drain_queue(self._synth_q)
                self._audio_q.put(None)
                break

            if wavs and wavs[0] is not None:
                audio = np.squeeze(wavs[0]).astype(np.float32)
                if len(audio) > 0:
                    pre  = np.zeros(int(0.05 * self.tts_sr), dtype=np.float32)
                    post = np.zeros(int(0.3  * self.tts_sr), dtype=np.float32)
                    audio = np.concatenate([pre, audio, post])
                    self._audio_q.put(audio)

    def _playback_worker(self) -> None:
        """Pull audio from _audio_q, play sequentially."""
        while True:
            audio = self._audio_q.get()
            if audio is None:
                break
            if self.interrupt.is_set():
                self._drain_queue(self._audio_q)
                break

            idx  = [0]
            done = threading.Event()

            def cb(outdata, frames, *_):
                rem = len(audio) - idx[0]
                if rem <= 0 or self.interrupt.is_set():
                    outdata[:] = 0
                    raise sd.CallbackStop()
                n               = min(frames, rem)
                outdata[:n, 0]  = audio[idx[0]: idx[0] + n]
                outdata[n:, 0]  = 0
                idx[0]         += n

            with sd.OutputStream(
                samplerate=self.tts_sr,
                channels=1,
                dtype="float32",
                device=config.SPEAKER_DEVICE,
                blocksize=1024,
                callback=cb,
                finished_callback=done.set,
            ):
                done.wait()

    @staticmethod
    def _drain_queue(q: queue.Queue) -> None:
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break

    # ── message building ─────────────────────────────────────────────────────

    def _build_messages(self, extra_user_msg: str | None = None) -> list[dict]:
        """Build message list for the conv model, fitting within CONV_NUM_CTX.

        Token budget: system_prompt + brain_context + history + output_reserve
        must fit within CONV_NUM_CTX.  History is trimmed from oldest first.

        Structure:
          [system] character prompt (static)
          [user/assistant] trimmed conversation history ...
          [system] brain context (scene + guide) ← injected LATE
          [user] latest message (from history or extra_user_msg)
        """
        budget = config.MODEL_NUM_CTX
        output_reserve = config.CONV_OUTPUT_RESERVE

        # System prompt with dynamic background injected
        dynamic_bg = self.brain.get_dynamic_background() if self.brain else ""
        system_prompt = config.SYSTEM_PROMPT.replace("{dynamic_background}", dynamic_bg)
        sys_tokens = _estimate_tokens(system_prompt)

        # Build brain context block (truncate each part to control size)
        brain_context = ""
        brain_tokens = 0
        if self.brain:
            brief = self.brain.get_context_brief()
            now = datetime.now()
            _weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
            time_str = f"{now.month}月{now.day}日 {_weekdays[now.weekday()]} {now.hour}:{now.minute:02d}"
            parts: list[str] = [f"现在是: {time_str}"]
            if brief.scene:
                parts.append(f"你看到: {brief.scene[:config.CONV_SCENE_MAX_CHARS]}")
            if brief.screen:
                parts.append(f"屏幕上: {brief.screen[:config.CONV_SCREEN_MAX_CHARS]}")
            if brief.conversation_guide:
                parts.append(f"你的想法: {brief.conversation_guide[:config.CONV_GUIDE_MAX_CHARS]}")
            brain_context = (
                "【感知】\n"
                + "\n".join(parts)
                + "\n自然回应，不要说'我看到'、'画面中'。"
            )
            brain_tokens = _estimate_tokens(brain_context)

        # Extra user message for autonomous speech
        extra_tokens = _estimate_tokens(extra_user_msg) if extra_user_msg else 0

        # Available budget for conversation history
        available = budget - sys_tokens - brain_tokens - extra_tokens - output_reserve
        available = max(available, 0)

        # Trim history to fit (keep most recent messages)
        history = self.history[1:]  # skip system prompt entry
        trimmed: list[dict] = []
        used = 0
        dropped = 0
        for msg in reversed(history):
            msg_tokens = _estimate_tokens(msg["content"])
            if used + msg_tokens > available:
                dropped += 1
                continue   # try older messages too — skip long ones
            trimmed.insert(0, msg)
            used += msg_tokens

        total_est = sys_tokens + brain_tokens + extra_tokens + used + output_reserve
        if dropped:
            print(f"  [ctx] ~{total_est}/{budget}tok (dropped {dropped} old msgs)")

        # Assemble final messages
        messages: list[dict] = [
            {"role": "system", "content": system_prompt}
        ]

        if brain_context and trimmed:
            # Insert brain context right before the last user/auto message
            messages.extend(trimmed[:-1])
            messages.append({"role": "system", "content": brain_context})
            messages.append(trimmed[-1])
        else:
            messages.extend(trimmed)

        # Autonomous speech: append the intent prompt as user message
        if extra_user_msg:
            messages.append({"role": "user", "content": extra_user_msg})

        return messages

    # ── stream LLM + TTS (shared by process_loop and autonomous speech) ──────

    async def _stream_and_speak(
        self, messages: list[dict], label: str = "bot"
    ) -> str:
        """Stream tokens from conv model via native Ollama API, split into sentences, pipe to TTS.
        Returns the full generated text.

        Uses /api/chat with num_ctx to prevent Ollama from reloading the model
        with default 128k context (which would evict both pinned models).
        """
        loop = asyncio.get_running_loop()
        full = ""
        buf  = ""
        t_llm = time.perf_counter()
        t_first_tok = None
        t_tts_start = None
        print(f"[{label}] ", end="", flush=True)

        self.bot_speaking.set()
        synth_future = loop.run_in_executor(None, self._synthesize_worker)
        play_future  = loop.run_in_executor(None, self._playback_worker)

        try:
            body = {
                "model": config.CONV_MODEL,
                "messages": messages,
                "options": {
                    "num_ctx": config.MODEL_NUM_CTX,
                    "num_predict": config.CONV_NUM_PREDICT,
                },
                "stream": True,
                "think": False,   # disable thinking at API level (prompt /no_think alone isn't enough)
            }
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST",
                    f"{_OLLAMA_BASE}/api/chat",
                    json=body,
                    timeout=httpx.Timeout(connect=10, read=120, write=10, pool=10),
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        tok = data.get("message", {}).get("content", "")
                        if tok and t_first_tok is None:
                            t_first_tok = time.perf_counter()
                        if tok:
                            print(tok, end="", flush=True)
                            full += tok
                            buf  += tok

                            sent, buf = pop_sentence(buf)
                            if sent:
                                if t_tts_start is None:
                                    t_tts_start = time.perf_counter()
                                self._synth_q.put(sent)

                        if self.interrupt.is_set():
                            break

                        if data.get("done", False):
                            break

            # Flush trailing text
            if buf.strip() and not self.interrupt.is_set():
                if t_tts_start is None:
                    t_tts_start = time.perf_counter()
                self._synth_q.put(buf.strip())

        except Exception as e:
            print(f"\n[LLM error] {e}")
        finally:
            self._synth_q.put(None)
            await synth_future
            await play_future
            self.bot_speaking.clear()

            self._drain_queue(self._synth_q)
            self._drain_queue(self._audio_q)

            ttft = (t_first_tok - t_llm) if t_first_tok else 0
            tts_elapsed = (time.perf_counter() - t_tts_start) if t_tts_start else 0
            print(f"  | TTFT {_C}{ttft:.2f}s{_R} | TTS+play {_C}{tts_elapsed:.2f}s{_R}")
            self.interrupt.clear()

        # Save detailed conv log (full input/output for token analysis)
        self._conv_call_count += 1
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "cycle": self._conv_call_count,
            "label": label,
            "timing": {
                "ttft_s": round(ttft, 2),
                "tts_play_s": round(tts_elapsed, 2),
            },
            "input": {
                "model": config.CONV_MODEL,
                "options": {
                    "num_ctx": config.MODEL_NUM_CTX,
                    "num_predict": config.CONV_NUM_PREDICT,
                },
                "think": False,
                "messages": messages,
                "token_estimates": {
                    "per_message": [
                        {"role": m["role"], "tokens": _estimate_tokens(m["content"]), "chars": len(m["content"])}
                        for m in messages
                    ],
                    "total_input": sum(_estimate_tokens(m["content"]) for m in messages),
                    "output_reserve": config.CONV_OUTPUT_RESERVE,
                    "budget": config.MODEL_NUM_CTX,
                },
            },
            "output": {
                "text": full,
                "tokens_est": _estimate_tokens(full),
            },
        }
        log_path = _CONV_DETAIL_DIR / f"{ts}_{self._conv_call_count:04d}_{label}.json"
        log_path.write_text(json.dumps(log_data, ensure_ascii=False, indent=2))

        return full

    # ── main pipeline ─────────────────────────────────────────────────────────

    async def process_loop(self) -> None:
        while True:
            audio = await self.speech_q.get()

            # Wait for any ongoing speech to finish
            while self.bot_speaking.is_set():
                await asyncio.sleep(0.02)
            self.interrupt.clear()

            # STT
            t_stt = time.perf_counter()
            text = await self._transcribe(audio)
            if not text:
                continue
            stt_elapsed = time.perf_counter() - t_stt
            print(f"[user] {text}  | STT {_C}{stt_elapsed:.2f}s{_R}")

            # Notify brain of user speech
            if self.brain:
                self.brain.record_user_speech(text)

            # Relevance check: is this speech directed at 小悠?
            if self.brain and not self.brain.should_respond(text):
                # Media playing → save transcription as media audio for later memory extraction
                if self.brain.is_media_playing():
                    self.brain.record_media_audio(text)
                print("  (not directed at bot, ignoring)")
                continue

            # Append to conversation history
            self.history.append({"role": "user", "content": text})

            # Build messages with brain context and generate response
            messages = self._build_messages()

            async with self._speech_lock:
                full = await self._stream_and_speak(messages, label="bot")

            if full:
                self.history.append({"role": "assistant", "content": full})
                if self.brain:
                    self.brain.record_bot_speech(full)

            # Trim history to prevent unbounded growth
            self._trim_history()

    # ── autonomous speech (triggered by brain) ────────────────────────────────

    async def _handle_autonomous_speech(self, intent: str) -> None:
        """Brain decided to speak — generate actual words via conv model."""
        if self.bot_speaking.is_set():
            return

        # Truncate long intents to save tokens
        mc = config.CONV_INTENT_MAX_CHARS
        short_intent = intent[:mc] if len(intent) > mc else intent
        prompt = (
            f"（你想主动说点什么。想法: {short_intent}。"
            f"自然简短地说，一两句。）"
        )

        # Build messages — prompt is appended as extra user message
        messages = self._build_messages(extra_user_msg=prompt)

        async with self._speech_lock:
            full = await self._stream_and_speak(messages, label="auto")

        if full:
            # Record in history as a natural utterance (not the meta-prompt)
            self.history.append({"role": "assistant", "content": full})
            if self.brain:
                self.brain.record_bot_speech(full)

    # ── history management ────────────────────────────────────────────────────

    def _trim_history(self, max_messages: int = config.CONV_MAX_HISTORY) -> None:
        """Keep system prompt + last N messages."""
        if len(self.history) <= max_messages + 1:
            return
        system = self.history[0]
        self.history = [system] + self.history[-(max_messages):]

    # ── startup ───────────────────────────────────────────────────────────────

    async def run(self) -> None:
        loop = asyncio.get_running_loop()

        print("Setting up PipeWire AEC…")
        _setup_aec()

        # Whisper CUDA JIT warmup
        dummy = np.zeros(16000, dtype=np.float32)
        t0 = time.perf_counter()
        print("Warming up Whisper…")
        await loop.run_in_executor(
            None, lambda: list(self.asr.transcribe(dummy)[0])
        )
        print(f"  Whisper ready  | {_C}{time.perf_counter() - t0:.2f}s{_R}")

        # ChatTTS warmup
        t0 = time.perf_counter()
        print("Warming up ChatTTS…")
        await loop.run_in_executor(
            None,
            lambda: self.tts.infer(
                ["你好世界"],
                skip_refine_text=False,
                params_infer_code={
                    "spk_emb": self._tts_spk,
                    "temperature": config.CHATTTS_TEMPERATURE,
                    "top_P": 0.7,
                    "top_K": 20,
                },
            ),
        )
        print(f"  ChatTTS ready  | {_C}{time.perf_counter() - t0:.2f}s{_R}")

        # Pin model in VRAM (keep_alive=-1, single model with parallel slots)
        t0 = time.perf_counter()
        print(f"Loading model ({config.CONV_MODEL}, "
              f"ctx={config.MODEL_NUM_CTX})…")
        await _ollama_load(config.CONV_MODEL, -1, config.MODEL_NUM_CTX)
        print(f"  Model pinned | {_C}{time.perf_counter() - t0:.2f}s{_R}")

        # Start camera
        if self.camera:
            print("Starting camera…")
            self.camera.start()

        # Start screen capture
        if self.screen:
            print("Starting screen capture…")
            await self.screen.start()

        # Load memories
        if self.memory:
            removed = self.memory.consolidate()
            print(f"  Memories loaded: {self.memory.count}"
                  + (f" ({removed} old entries pruned)" if removed else ""))

        # Synthesize dynamic background from memories (before first think)
        if self.brain:
            await self.brain.synthesize_dynamic_background()

        # Initial brain think (so it has context before first conversation)
        if self.brain:
            t0 = time.perf_counter()
            print("Brain initial observation…")
            await self.brain.think_once()
            print(f"  Brain ready | {_C}{time.perf_counter() - t0:.2f}s{_R}")

        print("\nAll systems ready.\n")

        print(f"Conv detail logs: {_CONV_DETAIL_DIR.resolve()}/")

        # Gather all async loops with graceful Ctrl+C shutdown
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop.set)

        coros = [self.mic_loop(), self.process_loop()]
        if self.brain:
            coros.append(self.brain.brain_loop())
        running = [asyncio.create_task(c) for c in coros]

        # Wait for stop signal, then cancel tasks
        await stop.wait()
        print("\nShutting down…")
        for t in running:
            t.cancel()
        await asyncio.gather(*running, return_exceptions=True)

        # Release resources
        if self.camera:
            self.camera.stop()
        if self.screen:
            await self.screen.stop()

        # Unload model from VRAM
        print("Releasing model…")
        await _ollama_unload(config.CONV_MODEL)
        print("Model released. Bye!")


# ── entry point ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    asyncio.run(VoiceBot().run())
