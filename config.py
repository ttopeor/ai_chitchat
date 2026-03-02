# ── Language ──────────────────────────────────────────────────────────────────
LANG = "zh"   # overridden by --lang CLI argument

# ── LLM ──────────────────────────────────────────────────────────────────────
LLM_BASE_URL = "http://10.0.0.190:11434/v1"
LLM_API_KEY  = "ollama"

# 单模型双通道: qwen3.5:122b 同时用于实时对话(嘴)和后台思考(脑)
# Ollama 服务端需设置: OLLAMA_NUM_PARALLEL=2 (允许同一模型两路并发)
# main.py 启动时 pin 一次模型 (keep_alive=-1)，退出时释放
CONV_MODEL    = "qwen3.5:122b"     # 实时对话 (嘴)
BRAIN_MODEL   = "qwen3.5:122b"    # 后台大脑 (脑) — 同一模型
MODEL_NUM_CTX = 128000              # 统一 context 窗口 (每个 parallel slot 独立分配)
# /no_think skips Qwen3 chain-of-thought, keeps responses snappy

# ── STT (faster-whisper) ──────────────────────────────────────────────────────
WHISPER_MODEL        = "large-v3"   # best accuracy; use "medium" for lower VRAM
WHISPER_DEVICE       = "cuda"
WHISPER_COMPUTE_TYPE = "float16"

# ── TTS (ChatTTS) ─────────────────────────────────────────────────────────────
CHATTTS_DEVICE      = "cuda"   # "cuda" or "cpu"
CHATTTS_TEMPERATURE = 0.3      # lower = more stable output, less random artifacts
CHATTTS_SEED        = 2     # speaker voice seed — try different values for different voices
                               # Good seeds for "小悠" voice: 2
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

# ── Screen Capture（远程 Windows 屏幕截图）──────────────────────────────────
SCREEN_ENABLED   = True
SCREEN_URL       = "http://10.0.0.70:7890/screenshot"  # Windows screen_server.py
SCREEN_INTERVAL  = 3.0     # fetch 间隔秒数

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

# ── Token 限制 (嘴 conv 侧) ────────────────────────────────────────────────
CONV_NUM_PREDICT         = 200    # 唯一硬约束：TTS 延迟。prompt 要求"最多两句话"，200 tok 已是 3-4 句的空间
CONV_OUTPUT_RESERVE      = 200    # 和 num_predict 保持一致
CONV_MAX_HISTORY         = 200    # 100轮对话记忆 (~10K tok)，128K下毫无压力
CONV_SCENE_MAX_CHARS     = 800    # 摄像头场景描述截断上限
CONV_SCREEN_MAX_CHARS    = 600    # 屏幕内容描述截断上限
CONV_GUIDE_MAX_CHARS     = 800    # 脑的指导几乎不截断，完整传达脑的意图
CONV_INTENT_MAX_CHARS    = 400    # 自主发言意图完整保留

# ── Token 限制 (脑 brain 侧) ───────────────────────────────────────────────
BRAIN_IMAGE_TOKEN_RESERVE  = 1500   # 不变 — 640x480 JPEG 大小固定
SCREEN_IMAGE_TOKEN_RESERVE = 1200   # 1280x549 缩放后的屏幕截图
BRAIN_OUTPUT_TOKEN_RESERVE = 500    # content 输出（thinking 已关闭）
BRAIN_TRANSCRIPT_ENTRIES   = 40     # 脑看完整对话脉络，不再只看几轮就下判断
BRAIN_RECENT_MAX_AGE       = 1800   # 脑记住30分钟内的对话
BRAIN_RECENT_MAX_COUNT     = 100    # 配合 max_age，缓存足够多的对话

# ── 动态背景合成 ──────────────────────────────────────────────────────────
# PROFILE_SYNTHESIS_PROMPT moved to i18n/zh.py and i18n/en.py
PROFILE_MIN_IMPORTANCE     = 3      # 只用 importance >= 3 的记忆来合成背景

# ── Tools (网络浏览工具) ──────────────────────────────────────────────────
TOOLS_ENABLED              = True
TOOLS_MAX_ROUNDS           = 3       # 最多工具调用轮数
TOOLS_SEARCH_MAX_RESULTS   = 3       # 每次搜索返回结果数
TOOLS_TIMEOUT              = 10.0    # 工具执行超时（秒）
