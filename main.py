"""
VoiceBot — local STT → LLM → TTS voice chat
  STT : faster-whisper  (CUDA)
  LLM : Ollama / qwen3.5 (OpenAI-compatible API)
  TTS : ChatTTS          (CUDA, 中英混合)
  VAD : Silero VAD
"""
import asyncio
import re
import threading
import time

import ChatTTS
import numpy as np
import sounddevice as sd
import torch
from faster_whisper import WhisperModel
from openai import AsyncOpenAI

import config


# ── sentence splitter ─────────────────────────────────────────────────────────

_SENT_RE = re.compile(r'[。！？\n]|(?<=[.!?])[ \t]')
_MAX_BUF = 60   # force-flush if no punctuation found within this many chars


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
        self._tts_spk = self.tts.sample_random_speaker(seed=42)  # fixed voice for session

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
                is_speech = self._vad_prob(chunk) > config.VAD_THRESHOLD

                if is_speech:
                    if not in_speech:
                        in_speech = True
                        # Interrupt bot mid-sentence if it's speaking
                        if self.bot_speaking.is_set():
                            self.interrupt.set()
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

    # ── STT ───────────────────────────────────────────────────────────────────

    async def _transcribe(self, audio: np.ndarray) -> str:
        loop = asyncio.get_running_loop()

        def _run() -> str:
            segs, _ = self.asr.transcribe(audio, beam_size=5, vad_filter=True)
            return "".join(s.text for s in segs).strip()

        return await loop.run_in_executor(None, _run)

    # ── TTS + playback ────────────────────────────────────────────────────────

    def _play_blocking(self, text: str) -> None:
        """Synthesize one sentence and play it. Runs in a thread-pool worker."""
        if self.interrupt.is_set():
            return

        params = {
            "spk_emb": self._tts_spk,
            "temperature": config.CHATTTS_TEMPERATURE,
            "top_P": 0.7,
            "top_K": 20,
        }
        wavs = self.tts.infer([text], skip_refine_text=True, params_infer_code=params)

        if not wavs or wavs[0] is None or self.interrupt.is_set():
            return

        audio = np.squeeze(wavs[0]).astype(np.float32)
        if len(audio) == 0 or self.interrupt.is_set():
            return

        idx   = [0]
        done  = threading.Event()

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

    async def _speak(self, text: str) -> None:
        if not text:
            return
        loop = asyncio.get_running_loop()
        self.bot_speaking.set()
        try:
            await loop.run_in_executor(None, self._play_blocking, text)
        finally:
            self.bot_speaking.clear()

    # ── main pipeline ─────────────────────────────────────────────────────────

    async def process_loop(self) -> None:
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
            print(f"👤 {text}  ⏱ STT {stt_elapsed:.2f}s")

            # LLM stream → sentence-level TTS
            self.history.append({"role": "user", "content": text})
            full = ""
            buf  = ""
            tts_total = 0.0
            t_llm = time.perf_counter()
            t_first_tok = None
            print("🤖 ", end="", flush=True)
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
                        t_tts = time.perf_counter()
                        await self._speak(sent)
                        tts_total += time.perf_counter() - t_tts

                    if self.interrupt.is_set():
                        await stream.close()
                        break

                # Flush any trailing text
                if buf.strip() and not self.interrupt.is_set():
                    t_tts = time.perf_counter()
                    await self._speak(buf.strip())
                    tts_total += time.perf_counter() - t_tts

            except Exception as e:
                print(f"\n[LLM error] {e}")
            finally:
                llm_elapsed = time.perf_counter() - t_llm - tts_total
                ttft = (t_first_tok - t_llm) if t_first_tok else 0
                print(f"  ⏱ LLM {llm_elapsed:.2f}s (TTFT {ttft:.2f}s) | TTS {tts_total:.2f}s")
                if full:
                    self.history.append({"role": "assistant", "content": full})
                self.interrupt.clear()

    # ── startup ───────────────────────────────────────────────────────────────

    async def run(self) -> None:
        loop  = asyncio.get_running_loop()

        # Whisper CUDA JIT warmup
        dummy = np.zeros(16000, dtype=np.float32)
        t0 = time.perf_counter()
        print("Warming up Whisper…")
        await loop.run_in_executor(
            None, lambda: list(self.asr.transcribe(dummy)[0])
        )
        print(f"  Whisper ready  ⏱ {time.perf_counter() - t0:.2f}s")

        # ChatTTS warmup — triggers internal lazy init
        t0 = time.perf_counter()
        print("Warming up ChatTTS…")
        await loop.run_in_executor(
            None,
            lambda: self.tts.infer(
                ["warmup"],
                skip_refine_text=True,
                params_infer_code={
                    "spk_emb": self._tts_spk,
                    "temperature": config.CHATTTS_TEMPERATURE,
                    "top_P": 0.7,
                    "top_K": 20,
                },
            ),
        )
        print(f"  ChatTTS ready  ⏱ {time.perf_counter() - t0:.2f}s")

        # LLM warmup — preload model params to GPU on Ollama side
        t0 = time.perf_counter()
        print("Warming up LLM…")
        await self.llm.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
        )
        print(f"  LLM ready      ⏱ {time.perf_counter() - t0:.2f}s")

        print("\nAll models ready.\n")
        await asyncio.gather(self.mic_loop(), self.process_loop())


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    asyncio.run(VoiceBot().run())
