"""
VoiceBot — local STT → LLM → TTS voice chat
  STT : faster-whisper  (CUDA)
  LLM : Ollama / qwen3.5 (OpenAI-compatible API)
  TTS : ChatTTS          (CUDA, 中英混合)
  VAD : Silero VAD
"""
import asyncio
import logging
import queue
import re
import subprocess
import threading
import time
import os

os.environ['TQDM_DISABLE'] = '1'

# Silence noisy third-party loggers (keep WARNING and above)
for _name in ("faster_whisper", "httpx", "ChatTTS", "onnxruntime"):
    logging.getLogger(_name).setLevel(logging.WARNING)

import ChatTTS
import numpy as np
import sounddevice as sd
import torch
from faster_whisper import WhisperModel
from openai import AsyncOpenAI

import config

# ANSI color for timing output
_C = "\033[36m"   # cyan
_R = "\033[0m"    # reset


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
        print("  ⚠ wpctl not found — PipeWire AEC not configured")
    except Exception as e:
        print(f"  ⚠ AEC setup failed: {e}")


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
            self.tts.load_models(compile=False)   # downloads from HuggingFace on first run
        finally:
            torch.load = _orig_load
        self.tts_sr = 24000                   # ChatTTS always outputs 24 kHz
        self._tts_spk = self.tts.sample_random_speaker(seed=config.CHATTTS_SEED)

        print("Connecting to LLM…")
        self.llm = AsyncOpenAI(
            base_url=config.LLM_BASE_URL,
            api_key=config.LLM_API_KEY,
        )

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
        interrupt_count = 0  # consecutive speech frames while bot is speaking
        bot_was_speaking = False
        cooldown_n = 0  # frames to skip after bot stops speaking
        COOLDOWN_FRAMES = int(0.5 * config.MIC_SAMPLE_RATE / CHUNK)  # 0.5s

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
            print("\n🎙️  Ready — speak now.\n")
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
                            # Require consecutive frames before triggering interrupt
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
        """Pull text from _synth_q, synthesize, push audio to _audio_q.
        Runs in a thread-pool thread. Stops on None sentinel or interrupt."""
        while True:
            text = self._synth_q.get()
            if text is None:
                self._audio_q.put(None)
                break
            if self.interrupt.is_set():
                self._drain_queue(self._synth_q)
                self._audio_q.put(None)
                break

            # Fresh dict each call — ChatTTS mutates params_infer_code internally
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
        """Pull audio from _audio_q, play sequentially.
        Runs in a thread-pool thread. Stops on None sentinel or interrupt."""
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
        """Discard all remaining items in a queue."""
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break

    # ── main pipeline ─────────────────────────────────────────────────────────

    async def process_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            audio = await self.speech_q.get()

            # Wait for any interrupted TTS to fully stop
            while self.bot_speaking.is_set():
                await asyncio.sleep(0.02)
            self.interrupt.clear()

            # STT
            t_stt = time.perf_counter()
            text = await self._transcribe(audio)
            if not text:
                continue
            stt_elapsed = time.perf_counter() - t_stt
            print(f"👤 {text}  ⏱ STT {_C}{stt_elapsed:.2f}s{_R}")

            # LLM stream → pipelined TTS
            self.history.append({"role": "user", "content": text})
            full = ""
            buf  = ""
            t_llm = time.perf_counter()
            t_first_tok = None
            t_tts_start = None
            print("🤖 ", end="", flush=True)

            # Launch synthesis and playback workers
            self.bot_speaking.set()
            synth_future = loop.run_in_executor(None, self._synthesize_worker)
            play_future  = loop.run_in_executor(None, self._playback_worker)

            try:
                stream = await self.llm.chat.completions.create(
                    model=config.LLM_MODEL,
                    messages=self.history,
                    stream=True,
                    max_tokens=512,
                )
                async for event in stream:
                    tok = event.choices[0].delta.content or ""
                    if tok and t_first_tok is None:
                        t_first_tok = time.perf_counter()
                    print(tok, end="", flush=True)
                    full += tok
                    buf  += tok

                    sent, buf = pop_sentence(buf)
                    if sent:
                        if t_tts_start is None:
                            t_tts_start = time.perf_counter()
                        self._synth_q.put(sent)

                    if self.interrupt.is_set():
                        await stream.close()
                        break

                # Flush any trailing text
                if buf.strip() and not self.interrupt.is_set():
                    if t_tts_start is None:
                        t_tts_start = time.perf_counter()
                    self._synth_q.put(buf.strip())

            except Exception as e:
                print(f"\n[LLM error] {e}")
            finally:
                # Signal end of turn → workers will exit
                self._synth_q.put(None)
                await synth_future
                await play_future
                self.bot_speaking.clear()

                # Defensive drain in case of interrupt leftovers
                self._drain_queue(self._synth_q)
                self._drain_queue(self._audio_q)

                # Timing
                ttft = (t_first_tok - t_llm) if t_first_tok else 0
                tts_elapsed = (time.perf_counter() - t_tts_start) if t_tts_start else 0
                print(f"  ⏱ LLM TTFT {_C}{ttft:.2f}s{_R} | TTS+play {_C}{tts_elapsed:.2f}s{_R}")
                if full:
                    self.history.append({"role": "assistant", "content": full})
                self.interrupt.clear()

    # ── startup ───────────────────────────────────────────────────────────────

    async def run(self) -> None:
        loop  = asyncio.get_running_loop()

        print("Setting up PipeWire AEC…")
        _setup_aec()

        # Whisper CUDA JIT warmup
        dummy = np.zeros(16000, dtype=np.float32)
        t0 = time.perf_counter()
        print("Warming up Whisper…")
        await loop.run_in_executor(
            None, lambda: list(self.asr.transcribe(dummy)[0])
        )
        print(f"  Whisper ready  ⏱ {_C}{time.perf_counter() - t0:.2f}s{_R}")

        # ChatTTS warmup — must match actual inference params (skip_refine_text=False)
        # to exercise the refine_text pipeline and avoid garbled first output
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
        print(f"  ChatTTS ready  ⏱ {_C}{time.perf_counter() - t0:.2f}s{_R}")

        # LLM warmup — preload model params to GPU on Ollama side
        t0 = time.perf_counter()
        print("Warming up LLM…")
        await self.llm.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
        )
        print(f"  LLM ready      ⏱ {_C}{time.perf_counter() - t0:.2f}s{_R}")

        print("\nAll models ready.\n")
        await asyncio.gather(self.mic_loop(), self.process_loop())


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    asyncio.run(VoiceBot().run())
