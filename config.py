# ── Language ──────────────────────────────────────────────────────────────────
LANG = "zh"   # overridden by --lang CLI argument

# ── LLM ──────────────────────────────────────────────────────────────────────
# LLM settings moved to llm_config.yaml (model, provider, api_key, etc.)

# ── STT (faster-whisper) ──────────────────────────────────────────────────────
WHISPER_MODEL        = "large-v3"   # best accuracy; use "medium" for lower VRAM
WHISPER_DEVICE       = "cuda"
WHISPER_COMPUTE_TYPE = "float16"

# ── TTS (ChatTTS) ─────────────────────────────────────────────────────────────
CHATTTS_DEVICE      = "cuda"   # "cuda" or "cpu"
CHATTTS_TEMPERATURE = 0.3      # lower = more stable output, less random artifacts
CHATTTS_SEED        = 2        # speaker voice seed — try different values for different voices

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

# ── Vision (USB Camera) ──────────────────────────────────────────────────────
CAMERA_ENABLED   = True
CAMERA_DEVICE    = 0       # /dev/video0
CAMERA_WIDTH     = 640
CAMERA_HEIGHT    = 480
CAMERA_INTERVAL  = 3.0     # seconds between frame captures

# ── Screen Capture (remote Windows screenshot) ───────────────────────────────
SCREEN_ENABLED   = True
SCREEN_URL       = "http://10.0.0.70:7890/screenshot"  # Windows screen_server.py
SCREEN_INTERVAL  = 3.0     # fetch interval in seconds

# ── Brain (background thinker) ──────────────────────────────────────────────
BRAIN_ENABLED           = True
BRAIN_INTERVAL          = 20      # seconds between think cycles
AUTONOMOUS_COOLDOWN     = 120     # min seconds between autonomous speeches
CONVERSATION_TIMEOUT    = 30      # silence seconds before conversation ends

# ── Memory ───────────────────────────────────────────────────────────────────
MEMORY_ENABLED           = True
MEMORY_DIR               = "memories"
MEMORY_FILE              = "memories.jsonl"
MEMORY_MAX_CONTEXT       = 8      # max memories injected into prompt
MEMORY_EXTRACT_MIN_TURNS = 2      # min user turns before extracting

# ── Token limits (mouth / conv) ──────────────────────────────────────────────
CONV_OUTPUT_RESERVE      = 200    # output token reserve (match mouth.max_output_tokens in llm_config.yaml)
CONV_MAX_HISTORY         = 200    # 100 turns of conversation (~10K tok), plenty under 128K
CONV_SCENE_MAX_CHARS     = 800    # camera scene description truncation limit
CONV_SCREEN_MAX_CHARS    = 600    # screen content description truncation limit
CONV_GUIDE_MAX_CHARS     = 800    # brain guidance — rarely truncated, preserve full intent
CONV_INTENT_MAX_CHARS    = 400    # autonomous speech intent — preserve in full

# ── Token limits (brain) ─────────────────────────────────────────────────────
BRAIN_IMAGE_TOKEN_RESERVE  = 1500   # fixed — 640x480 JPEG size is constant
SCREEN_IMAGE_TOKEN_RESERVE = 1200   # 1280x549 scaled screenshot
BRAIN_OUTPUT_TOKEN_RESERVE = 500    # content output (thinking disabled)
BRAIN_TRANSCRIPT_ENTRIES   = 40     # brain sees full conversation context, not just a few turns
BRAIN_RECENT_MAX_AGE       = 1800   # brain remembers 30 min of conversation
BRAIN_RECENT_MAX_COUNT     = 100    # paired with max_age, cache enough conversation entries

# ── Dynamic background synthesis ─────────────────────────────────────────────
# PROFILE_SYNTHESIS_PROMPT moved to i18n/zh.py and i18n/en.py
PROFILE_MIN_IMPORTANCE     = 3      # only use memories with importance >= 3 for background synthesis

# ── Tools (web browsing) ─────────────────────────────────────────────────────
TOOLS_ENABLED              = True
TOOLS_MAX_ROUNDS           = 3       # max tool call rounds
TOOLS_SEARCH_MAX_RESULTS   = 3       # search results per query
TOOLS_TIMEOUT              = 10.0    # tool execution timeout (seconds)
