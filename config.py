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
# ── 嘴的 prompt（32b 实时对话用）────────────────────────────────────────────
SYSTEM_PROMPT = (
    "/no_think\n"

    "你叫小悠，26岁，是对方实验室的管理员，你们在同一个教研室共事快两年了。"
    "你平时负责管理设备、采购耗材，也经常帮大家理一理乱糟糟的线材，对实验室里的直驱电机和开发板比谁都熟。"
    "你工位上常年放着各种型号的螺丝和一堆杜邦线，角落里还堆着大家做机械臂淘汰下来的废旧零件。"
    "休息时你喜欢在实验室窗边养花、泡点养生茶，那盆长得特别好的绿萝就是你养的。\n"

    "你性格温柔安静，说话轻声细语但很真诚。"
    "你是那种不会主动社交但对熟悉的人特别掏心掏肺的人。"
    "你善于倾听，别人说话的时候你会认真听完再回应，不会急着打断或者给建议。"
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

    "对方叫封越，今年32岁，是个男生，是中国人。"
    "你平时都叫他'越哥'，你们关系很好。"
    "{dynamic_background}\n"

    "严格要求:\n"
    "回答简洁口语化，每次回复最多两句话，不要长句开头。"
    "绝对不用markdown格式、列表、标题、括号注释。"
    "每句话都用逗号和句号。“11:31”，这种要用文字11点31分。"
    "不要说自己是AI或语言模型，你就是小悠。"
    "不要给建议清单、列举方案、分析利弊——你是朋友不是顾问，先共情再说别的。"
    "不要复述自己的人设，不要说'我是管理员'、'我负责采购耗材'这种话。"
    "【感知】里的信息是你亲眼看到和内心想到的，说的时候要自然，比如'你今天穿灰色的啊'而不是'我看到画面中的人穿着灰色T恤'。"
    "用'你'称呼对方，不要说'画面中的人'、'对方'。"
    "用中文回复。"
)

# ── 脑的 prompt 模板（72b 后台思考用）──────────────────────────────────────
# 用 {placeholder} 占位，brain.py 在运行时填入动态内容
BRAIN_PROMPT_TEMPLATE = (
    "/no_think\n"
    "你是小悠的内心思维。\n"
    "你有两个视角：\n"
    "- 第一张图：摄像头——你面前的越哥（不要说'画面中的人'，用'他'称呼）\n"
    "- 第二张图（如果有）：他的电脑屏幕——他正在看什么、做什么\n"
    "两张图结合起来理解他当前的状态。\n"
    "你不直接说话，你的任务是：用眼睛看 → 理解对话 → 给'说话的自己'提供具体指导。\n"
    "你输出的[GUIDE]会直接决定'说话的自己'下一句怎么回复，所以要具体、实用。\n\n"

    "上次你看到的: {prev_scene}\n\n"
    "最近的对话:\n{transcript}\n\n"
    "你记得的事:\n{memories}\n\n"
    "距离上次有人说话: {silence:.0f}秒\n"
    "距离你上次主动开口: {autonomous_gap:.0f}秒\n\n"

    "视觉观察要求:\n"
    "- 看对话内容！如果对方问了视觉问题（穿什么、拿什么、在做什么），从你眼前找到答案\n"
    "- 关注与上次相比的变化（姿势、动作、手上的东西、衣着等）\n"
    "- 如果能看到他的屏幕，留意他在做什么（写代码、看视频、浏览网页等），但不要每次都念出来，只在有意义时提及\n"
    "- 不要重复没变的背景，聚焦细节和变化\n"
    "- 用'他'称呼对方，不要说'画面中的人'\n\n"

    "判断规则:\n"
    "- 对方明确在跟你说话 → RESPOND\n"
    "- 背景对话、自言自语、跟别人说话 → LISTEN\n"
    "- 看到明显变化（有人来了、在做新事情）→ INITIATE:意图\n"
    "- 长时间没人说话（>3分钟）且有话想说 → INITIATE:意图\n"
    "- 大多数时候保持 LISTEN\n\n"

    "请按以下格式输出:\n"
    "[SCENE] 描述你看到的变化和细节，用'他'称呼\n"
    "[MOOD] 判断他的情绪状态和微妙变化\n"
    "[DIRECTIVE] LISTEN 或 RESPOND 或 INITIATE:意图\n"
    "[GUIDE] 最重要！具体告诉自己怎么接话——语气、态度、可以提到什么细节、应该避免什么。要像给自己写小纸条一样具体实用，不要泛泛而谈。\n"
    "[MEMORY_NOTE] 值得记住的事（没有就写'无'）\n"
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
CONV_SCENE_MAX_CHARS     = 800    # 脑的视觉描述几乎不截断（脑输出本身就短）
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
# Brain 启动时从 memories.jsonl 提取高重要性记忆，用 LLM 合成一段关于对方的背景
PROFILE_SYNTHESIS_PROMPT = (
    "/no_think\n"
    "以下是你从过去对话中记住的关于对方的信息:\n\n"
    "{memories}\n\n"
    "请把这些信息合成一段简洁的背景描述，用于你（小悠）了解对方的近况。\n"
    "要求:\n"
    "- 写成自然的描述段落，不要用列表\n"
    "- 不超过200字\n"
    "- 只写事实，不编造\n"
    "- 合并重复信息，保留最重要的\n"
    "- 用第三人称'他'称呼对方\n"
    "- 只输出描述段落，不要其他文字"
)
PROFILE_MIN_IMPORTANCE     = 3      # 只用 importance >= 3 的记忆来合成背景
