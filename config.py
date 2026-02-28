# ── LLM ──────────────────────────────────────────────────────────────────────
LLM_BASE_URL = "http://10.0.0.190:11434/v1"
LLM_API_KEY  = "ollama"
LLM_MODEL    = "qwen2.5vl:72b"
# /no_think skips Qwen3 chain-of-thought, keeps responses snappy
SYSTEM_PROMPT = (
    "/no_think\n"
    "你是一个友好、简洁的AI语音助手。"
    "回答要简洁，适合口头表达，不要使用markdown格式，不要用列表或标题。"
)

# ── STT (faster-whisper) ──────────────────────────────────────────────────────
WHISPER_MODEL        = "large-v3"   # best accuracy; use "medium" for lower VRAM
WHISPER_DEVICE       = "cuda"
WHISPER_COMPUTE_TYPE = "float16"

# ── TTS (ChatTTS) ─────────────────────────────────────────────────────────────
CHATTTS_DEVICE      = "cuda"   # "cuda" or "cpu"
CHATTTS_TEMPERATURE = 0.3      # lower = more stable output, less random artifacts

# ── Audio I/O ─────────────────────────────────────────────────────────────────
MIC_SAMPLE_RATE = 16000
# MIC_DEVICE: None = system default microphone
# For PipeWire AEC (speaker mode), set to "Echo Cancellation Source"
# Run `python -c "import sounddevice; print(sounddevice.query_devices())"` to list devices.
MIC_DEVICE     = None
SPEAKER_DEVICE = None   # None = system default speaker

# ── VAD (Silero) ──────────────────────────────────────────────────────────────
VAD_THRESHOLD           = 0.5    # speech probability threshold [0, 1]
VAD_THRESHOLD_INTERRUPT = 0.80   # higher threshold during bot speech to reject residual echo
INTERRUPT_MIN_FRAMES    = 3      # consecutive speech frames to trigger interrupt (each ~32ms)
SILENCE_S               = 0.8    # seconds of silence before treating utterance as complete
