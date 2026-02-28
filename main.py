"""
VoiceBot — dual-system voice chat with vision, memory, and autonomous speech.

  System 1 (7b) : fast real-time conversation — mouth & reflexes
  System 2 (72b): background thinker — vision, memory, context generation

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
from brain import BrainEngine
from memory import MemoryManager
from vision import CameraCapture

# ANSI color for timing output
_C = "\033[36m"   # cyan
_R = "\033[0m"    # reset


# ── conversation logger ─────────────────────────────────────────────────────

_CONV_LOG_DIR = Path("logs/conversation")
_CONV_LOG_DIR.mkdir(parents=True, exist_ok=True)
_conv_log_path = _CONV_LOG_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"


def _log_conv(role: str, text: str, **extra):
    """Append a conversation event to the session log file."""
    entry = {
        "time": datetime.now().isoformat(),
        "role": role,
        "text": text,
        **extra,
    }
    with open(_conv_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


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

        # ── Vision ────────────────────────────────────────────────────────────
        self.camera: CameraCapture | None = None
        if config.CAMERA_ENABLED:
            self.camera = CameraCapture(
                device_index=config.CAMERA_DEVICE,
                width=config.CAMERA_WIDTH,
                height=config.CAMERA_HEIGHT,
                interval=config.CAMERA_INTERVAL,
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
            segs, _ = self.asr.transcribe(audio, beam_size=5, vad_filter=True)
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

    def _build_messages(self, current_text: str) -> list[dict]:
        """Build message list for the 7b model, enriched with brain context."""
        # Start with base system prompt
        system_content = config.SYSTEM_PROMPT

        # Inject brain's context brief (vision + memory + mood)
        if self.brain:
            brief = self.brain.get_context_brief()
            parts: list[str] = []
            if brief.scene:
                parts.append(f"你看到: {brief.scene}")
            if brief.memories:
                parts.append(f"你记得的事:\n{brief.memories}")
            if brief.mood_hint:
                parts.append(f"你的判断: {brief.mood_hint}")
            if brief.suggested_topics:
                parts.append(f"可以聊: {brief.suggested_topics}")
            if parts:
                system_content += (
                    "\n\n【你现在的感知】\n"
                    + "\n".join(parts)
                    + "\n自然地融入这些信息，不要刻意说'我看到'或'我记得'。"
                )

        messages: list[dict] = [{"role": "system", "content": system_content}]
        # Append conversation history (skip the original system prompt entry)
        messages.extend(self.history[1:])
        return messages

    # ── stream LLM + TTS (shared by process_loop and autonomous speech) ──────

    async def _stream_and_speak(
        self, messages: list[dict], label: str = "bot"
    ) -> str:
        """Stream tokens from 7b via native Ollama API, split into sentences, pipe to TTS.
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
                "options": {"num_ctx": config.CONV_NUM_CTX},
                "stream": True,
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
            _log_conv("user", text, stt_s=round(stt_elapsed, 2))

            # Notify brain of user speech
            if self.brain:
                self.brain.record_user_speech(text)

            # Relevance check: is this speech directed at 小悠?
            if self.brain and not self.brain.should_respond(text):
                print("  (not directed at bot, ignoring)")
                _log_conv("system", "ignored (not directed at bot)")
                continue

            # Append to conversation history
            self.history.append({"role": "user", "content": text})

            # Build messages with brain context and generate response
            messages = self._build_messages(text)

            async with self._speech_lock:
                full = await self._stream_and_speak(messages, label="bot")

            if full:
                self.history.append({"role": "assistant", "content": full})
                _log_conv("bot", full)
                if self.brain:
                    self.brain.record_bot_speech(full)

            # Trim history to prevent unbounded growth
            self._trim_history()

    # ── autonomous speech (triggered by brain) ────────────────────────────────

    async def _handle_autonomous_speech(self, intent: str) -> None:
        """Brain decided to speak — generate actual words via 7b."""
        if self.bot_speaking.is_set():
            return

        _log_conv("system", f"autonomous speech triggered: {intent}")

        prompt = (
            f"（你注意到了一些事情，想主动说点什么。\n"
            f"你的想法: {intent}\n"
            f"自然地说出来，像在教研室随口说话一样。简短一两句。）"
        )

        # Build messages with brain context
        messages = self._build_messages(prompt)

        async with self._speech_lock:
            full = await self._stream_and_speak(messages, label="auto")

        if full:
            # Record in history as a natural utterance (not the meta-prompt)
            self.history.append({"role": "assistant", "content": full})
            _log_conv("auto", full)
            if self.brain:
                self.brain.record_bot_speech(full)

    # ── history management ────────────────────────────────────────────────────

    def _trim_history(self, max_messages: int = 20) -> None:
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

        # Pin models in VRAM (keep_alive=-1, limited context to save VRAM)
        t0 = time.perf_counter()
        print(f"Loading conversation model ({config.CONV_MODEL}, "
              f"ctx={config.CONV_NUM_CTX})…")
        await _ollama_load(config.CONV_MODEL, -1, config.CONV_NUM_CTX)
        print(f"  Conv model pinned | {_C}{time.perf_counter() - t0:.2f}s{_R}")

        if config.BRAIN_ENABLED:
            t0 = time.perf_counter()
            print(f"Loading brain model ({config.BRAIN_MODEL}, "
                  f"ctx={config.BRAIN_NUM_CTX})…")
            await _ollama_load(config.BRAIN_MODEL, -1, config.BRAIN_NUM_CTX)
            print(f"  Brain model pinned | {_C}{time.perf_counter() - t0:.2f}s{_R}")

        # Start camera
        if self.camera:
            print("Starting camera…")
            self.camera.start()

        # Load memories
        if self.memory:
            removed = self.memory.consolidate()
            print(f"  Memories loaded: {self.memory.count}"
                  + (f" ({removed} old entries pruned)" if removed else ""))

        # Initial brain think (so it has context before first conversation)
        if self.brain:
            t0 = time.perf_counter()
            print("Brain initial observation…")
            await self.brain.think_once()
            print(f"  Brain ready | {_C}{time.perf_counter() - t0:.2f}s{_R}")

        print("\nAll systems ready.\n")

        print(f"Conversation log: {_conv_log_path}")

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

        # Unpin models — restore default keep_alive so Ollama can reclaim VRAM
        print("Releasing models…")
        await _ollama_load(config.CONV_MODEL, "5m")
        if config.BRAIN_ENABLED:
            await _ollama_load(config.BRAIN_MODEL, "5m")
        print("Models released. Bye!")


# ── entry point ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    asyncio.run(VoiceBot().run())
