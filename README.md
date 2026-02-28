# ai_chitchat — 小悠

一个有视觉、记忆和自主意识的本地语音聊天伙伴。

小悠是一个 26 岁的实验室管理员角色，通过麦克风听你说话、用摄像头观察环境、用声音回复你。她能记住跨会话的重要事情，也能在合适的时机主动开口。

## 双系统架构

灵感来自人脑的 System 1 / System 2 理论：

```
                     USB Camera
                         |
                         v
    =============== BRAIN (72b VLM) ==============
    ||                                           ||
    ||   camera frame --> scene description      ||
    ||   conversation --> memory extraction      ||
    ||   all context  --> speak directive        ||
    ||   visual Q&A   --> conversation guide     ||
    ||                                           ||
    ==============|==========|====================
                  |          |
           ContextBrief   INITIATE / follow-up
                  |          |
                  v          v
    =============== MOUTH (32b) ==================
    ||                                           ||
    ||   mic --> VAD --> STT                     ||
    ||                    \                      ||
    ||      brief -------> 32b stream --> TTS --+--> speaker
    ||                                           ||
    ==============================================
```

- **32b（嘴 / System 1）**：负责实时对话，追求低延迟和人格一致性。接收大脑消化好的 context，直接生成回复
- **72b（脑 / System 2）**：在后台持续观察（视觉）、思考、记忆，把消化好的 ContextBrief 喂给 32b

两个模型通过 Ollama 并行加载在同一张 GPU 上，互不阻塞。

## 技术栈

| 组件 | 技术 | 运行位置 |
|------|------|----------|
| STT | faster-whisper (large-v3) | 本地 GPU |
| TTS | ChatTTS | 本地 GPU |
| VAD | Silero VAD | 本地 GPU |
| 对话 LLM | qwen2.5vl:32b (Ollama, Q4_K_M) | Ollama 服务器 |
| 大脑 LLM | qwen2.5vl:72b (Ollama, Q4_K_M) | Ollama 服务器 |
| 视觉 | OpenCV (USB 摄像头) | 本地 |
| 记忆 | JSONL + keyword/importance 检索 | 本地文件 |

## 前置条件

- Linux (PipeWire 音频)
- NVIDIA GPU，CUDA 支持
- Ollama 服务器，96GB+ 显存（双模型共约 86GB）
- USB 麦克风（或系统默认音频输入）
- USB 摄像头（可选，`config.py` 中可关闭）

### Ollama 服务端配置

双模型同时运行需要 Ollama 允许加载多个模型：

```bash
# /etc/systemd/system/ollama.service 的 [Service] 段加:
Environment="OLLAMA_MAX_MODELS=2"

sudo systemctl daemon-reload
sudo systemctl restart ollama
```

拉取模型：

```bash
ollama pull qwen2.5vl:32b     # 32b, 实时对话（嘴）
ollama pull qwen2.5vl:72b     # 72b, 后台大脑
```

### 显存预算

```
32b @ ctx 2048   ≈ 28 GB
72b @ ctx 8192   ≈ 58 GB
STT + TTS + VAD  ≈  8 GB
─────────────────────────
Total            ≈ 94 GB  (fits in 96 GB)
```

## 安装

```bash
cd ~/Workspace/ai_chitchat
bash setup.sh
```

安装内容：
- Python venv + pip 依赖
- PyTorch (CUDA 12.8, Blackwell 兼容)
- ChatTTS 模型（首次运行时自动下载 ~1GB）
- PipeWire AEC（可选，用于扬声器模式回声消除）

## 运行

```bash
source .venv/bin/activate
python main.py
```

启动流程：
1. 加载 Whisper STT、Silero VAD、ChatTTS
2. 连接 Ollama，pin 两个模型到显存（`keep_alive=-1`，指定 `num_ctx` 控制 KV cache 大小）
3. 启动 USB 摄像头后台采集
4. 加载持久记忆，清理过期条目
5. 大脑初始观察（启动时拍一张照，建立初始 context）
6. 启动三个并发循环：麦克风监听、对话处理、大脑思考

退出时自动释放模型显存（`keep_alive` 恢复默认）。

## 项目结构

```
ai_chitchat/
├── main.py          # 主入口，VoiceBot 类，音频管线，双系统集成
├── config.py        # 所有配置：模型、Prompt、音频、摄像头、大脑、记忆
├── brain.py         # BrainEngine — 72b 后台大脑，ContextBrief 生成
├── vision.py        # CameraCapture — USB 摄像头后台线程
├── memory.py        # MemoryManager — JSONL 持久记忆，提取 & 检索
├── setup.sh         # 一键安装脚本
├── requirements.txt # Python 依赖
├── logs/            # 运行时自动创建
│   ├── brain/       # 每次大脑思考的完整日志（输入 prompt + 输出 + 摄像头截图）
│   └── conversation/# 每次会话的对话记录 (JSONL)
└── memories/        # 运行时自动创建
    └── memories.jsonl
```

## 配置

所有配置在 `config.py` 中，关键项：

```python
# 双模型
CONV_MODEL    = "qwen2.5vl:32b"    # 实时对话（嘴）
CONV_NUM_CTX  = 2048               # 对话 context 窗口
BRAIN_MODEL   = "qwen2.5vl:72b"   # 后台大脑
BRAIN_NUM_CTX = 8192               # 大脑 context 窗口

# 功能开关（可单独关闭）
CAMERA_ENABLED = True
BRAIN_ENABLED  = True
MEMORY_ENABLED = True

# 大脑节奏
BRAIN_INTERVAL      = 20    # 多久思考一次（秒），对话中自动加速到 0.5s
AUTONOMOUS_COOLDOWN = 120   # 两次主动说话的最小间隔（秒）
CONVERSATION_TIMEOUT = 30   # 沉默多久算一段对话结束
```

## 核心机制

### 大脑思考循环

72b 大脑每轮思考：
1. 抓取最新摄像头帧
2. 收集近期对话记录（最多 6 条）
3. 检索相关记忆
4. 送入 72b VLM（含图像），一次性完成场景理解 + 对话指导
5. 解析结构化输出 → 更新 ContextBrief
6. 根据 directive 决定是否触发自主说话或跟进补充

大脑输出格式：
```
[SCENE] 一句话描述场景变化
[MOOD] 判断对方状态
[DIRECTIVE] LISTEN / RESPOND / INITIATE:意图
[GUIDE] 给嘴的接话指导（限 30 字）
[MEMORY_NOTE] 值得记住的事
```

思考频率自动调整：
- 对话中：连续推理（~0.5s 间隔）
- 对话刚结束：正常间隔（20s）
- 长时间静默（>2min）：慢速（60s）

### 视觉

后台线程每 3 秒从 USB 摄像头抓一帧（640x480 JPEG），供大脑场景理解使用。摄像头断开时自动降级为纯语音模式。

大脑通过摄像头能回答视觉问题（"我穿的什么颜色"、"你看我在干嘛"），也能注意到场景变化主动评论。

### 记忆

- **提取**：大脑检测到值得记住的事时，用 72b 从对话中提取重要信息（事实、偏好、事件），存入 `memories.jsonl`
- **检索**：每次大脑思考时，按 keyword 匹配 + importance 权重检索相关记忆，注入 ContextBrief
- **防遗忘**：importance 权重远大于时间权重，重要的旧记忆不会被新记忆淹没
- **去重**：keyword 重叠 > 60% 的同类记忆自动合并更新
- **维护**：启动时清理 90 天以上的低重要性（<3）记忆

### 语音过滤

不是所有检测到的语音都会触发回复：
- 叫名字（"小悠"及 STT 常见误识变体）→ 一定回复
- 30s 内有过对话 → 视为连续对话，回复
- 大脑 directive 为 RESPOND → 回复
- 其他情况 → 忽略（背景对话、自言自语）

### 自主对话

大脑每轮思考输出一个 directive：
- `LISTEN` — 保持安静（大多数时候）
- `RESPOND` — 有人在跟我说话，应该回复
- `INITIATE:意图` — 我想主动说点什么（如看到用户回来、长时间沉默想搭话）

跟进机制：当大脑发现新信息（如视觉细节）但嘴已经回复过了，会触发一次跟进补充。

### Token 预算管理

两个模型的 context 窗口都有限（32b: 2048, 72b: 8192），系统自动管理 token 预算：

- **嘴 (32b)**：`_build_messages()` 计算 system prompt + brain context + history 的 token 总量，超出时从最早的历史消息开始裁剪
- **脑 (72b)**：`_build_brain_prompt()` 按比例分配预算（transcript 50%, memories 25%, prev_scene 15%），超出时逐层压缩，支持图像 token 预留（~1500 tokens）
- Token 估算使用 Qwen2.5 tokenizer 特性校准的近似公式（CJK ~1.0 tok/char, ASCII ~0.3 tok/char，~1.3x 安全余量）

### 并发安全

- `asyncio.Lock` 互斥用户对话和自主说话
- `threading.Lock` 保护摄像头帧读写
- 两个模型分开调度，Ollama 并行处理（需 `OLLAMA_MAX_MODELS=2`）
- `bot_speaking` + cooldown 防止自己的声音被麦克风拾取
- Ollama 原生 API（`/api/chat`）始终传 `num_ctx`，防止模型以默认 128k context 重载导致显存爆炸

### 日志

- **对话日志**：`logs/conversation/YYYYMMDD_HHMMSS.jsonl` — 每条消息（user/bot/auto/system）带时间戳
- **大脑日志**：`logs/brain/YYYYMMDD_HHMMSS_NNNN_think.json` — 每次思考的完整输入输出，含对应的摄像头截图（`_input.jpg`）
