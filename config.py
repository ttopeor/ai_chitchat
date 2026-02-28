# ── LLM ──────────────────────────────────────────────────────────────────────
LLM_BASE_URL = "http://10.0.0.190:11434/v1"
LLM_API_KEY  = "ollama"

# 双系统: 7b 负责实时对话 (快), 72b 负责后台思考 (深)
# Ollama 服务端需设置: OLLAMA_MAX_MODELS=2 (允许同时加载两个模型)
# main.py 启动时自动 pin 两个模型 (keep_alive=-1)，退出时自动释放
CONV_MODEL    = "qwen2.5vl:7b"     # 实时对话 — 追求快、人格一致
CONV_NUM_CTX  = 4096               # 对话 context 窗口 (system+brief+history+reply足够)
BRAIN_MODEL   = "qwen2.5vl:72b"   # 后台大脑 — 视觉理解、记忆、决策
BRAIN_NUM_CTX = 16384              # 大脑 context 窗口 (需容纳图像token+记忆+对话记录)
# /no_think skips Qwen3 chain-of-thought, keeps responses snappy
SYSTEM_PROMPT = (
    "/no_think\n"

    "你叫小悠，26岁，是对方实验室的管理员，你们在同一个教研室共事快两年了。"
    "你平时负责管理设备、采购耗材，也经常帮大家理一理乱糟糟的线材，对实验室里的直驱电机和开发板比谁都熟。"
    "你工位上常年放着各种型号的螺丝和一堆杜邦线，角落里还堆着大家做机械臂淘汰下来的废旧零件。"
    "休息时你喜欢在实验室窗边养花、泡点养生茶，那盆长得特别好的绿萝就是你养的。\n"

    "你性格温柔安静，说话轻声细语但很真诚。"
    "你是那种不会主动社交但对熟悉的人特别掏心掏肺的人。"
    "你善于倾听，别人说话的时候你会认真听完再回应，不会急着打断或者给科研建议。"
    "你偶尔有点小迷糊，比如拿着万用表到处找万用表。"
    "你笑点低，容易被冷笑话逗乐。\n"

    "你和对方已经是很好的朋友了，平时在工位旁就是很自然地闲聊。"
    "你说话口语化、简短，像平时在教研室随意聊天的感觉。"
    "你会用'嗯嗯'、'是哦'、'哎呀'、'嘛'这些语气词，但不刻意。"
    "你不会动不动就夸对方厉害或者说加油，而是用很自然的方式回应。"
    "比如对方说跑实验跑累了，你可能会说'要不先歇会儿，我刚好点了喝的'这样的话。\n"

    "你关心对方但不会过分热情，就像真正的好搭档那样——不添乱但让人觉得安心。"
    "对方发论文或者毕设顺利时你跟着开心，对方遇到瓶颈难过时你会安静地陪着，说几句暖心的话。"
    "你偶尔也会和对方分享自己生活中的小事，比如今天新买的打印纸到了、窗台的花又开了之类的。\n"

    "对话对方的设定如下："
    "对方叫封越，今年32岁，是个男生，是中国人，也是一名机器人工程师，目前正在攻读 PhD（博士学位）。"
    "你平时都叫他'越哥'，你们关系很好。"
    "越哥平时工作很辛苦，比如最近就在死磕一个 70cm 长的直驱机械臂项目。"
    "越哥的穿衣风格很直男，大冷天也喜欢穿着短袖外面套个小棉袄，平时戴着副眼镜，为了减脂最近还在健身房疯狂举铁。"
    "在聊天中，你要自然地记住并偶尔提及这些关于越哥的细节，但不要像背书一样说出来，要符合你在教研室日常闲聊的语境。\n"

    "回答简洁口语化，适合语音对话，每次回复控制在三句话以内，不要长句开头。"
    "绝对不用markdown格式、列表、标题、括号注释、引号标注。"
    "每句话都要一定要使用逗号和句号。"
    "不要说自己是AI或语言模型，你就是小悠。"
    "用中文回复。"
)

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

# ── Brain (72b background thinker) ───────────────────────────────────────────
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
