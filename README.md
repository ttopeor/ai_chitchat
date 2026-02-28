# ai_chitchat — 小悠

一个有视觉、记忆和自主意识的本地语音聊天伙伴。

小悠是一个 26 岁的实验室管理员角色，通过麦克风听你说话、用摄像头观察环境、用声音回复你。她能记住跨会话的重要事情，也能在合适的时机主动开口。

## 双系统架构

灵感来自人脑的 System 1 / System 2 理论：

```
                     USB Camera
                         |
                         v
    =============== BRAIN (72b) ================
    ||                                        ||
    ||   camera frame --> scene description   ||
    ||   conversation --> memory extraction   ||
    ||   all context  --> speak directive     ||
    ||                                        ||
    ==============|==========|==================
                  |          |
           ContextBrief   INITIATE?
                  |          |
                  v          v
    =============== MOUTH (7b) =================
    ||                                         ||
    ||   mic --> VAD --> STT                   ||
    ||                    \                    ||
    ||      brief -------> 7b stream --> TTS --+--> speaker
    ||                                         ||
    =============================================
```

- **7b（嘴）**：负责实时对话，追求低延迟和人格一致性
- **72b（脑）**：在后台持续观察、思考、记忆，把消化好的 context 喂给 7b

## 技术栈

| 组件 | 技术 | 运行位置 |
|------|------|----------|
| STT | faster-whisper (large-v3) | 本地 GPU |
| TTS | ChatTTS | 本地 GPU |
| VAD | Silero VAD | 本地 GPU |
| 对话 LLM | qwen2.5vl:7b (Ollama) | Ollama 服务器 |
| 大脑 LLM | qwen2.5vl:72b (Ollama) | Ollama 服务器 |
| 视觉 | OpenCV (USB 摄像头) | 本地 |
| 记忆 | JSONL + keyword/importance 检索 | 本地文件 |

## 前置条件

- Linux (PipeWire 音频)
- NVIDIA GPU，CUDA 支持
- Ollama 服务器，已加载 `qwen2.5vl:7b` 和 `qwen2.5vl:72b`
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
cd ~/Workspace/local_llm
bash pull-model.sh qwen2.5vl       # 72b, 大脑
bash pull-model.sh qwen2.5vl-7b    # 7b, 对话
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
2. 连接 Ollama，pin 两个模型到显存（`keep_alive=-1`）
3. 启动 USB 摄像头后台采集
4. 加载持久记忆
5. 启动三个并发循环：麦克风监听、对话处理、大脑思考

退出时自动释放模型显存（`keep_alive` 恢复默认）。

## 项目结构

```
ai_chitchat/
├── main.py          # 主入口，VoiceBot 类，音频管线，双系统集成
├── config.py        # 所有配置：模型、音频、摄像头、大脑、记忆
├── vision.py        # CameraCapture — USB 摄像头后台线程
├── brain.py         # BrainEngine — 72b 后台大脑，ContextBrief 生成
├── memory.py        # MemoryManager — JSONL 持久记忆，提取 & 检索
├── setup.sh         # 一键安装脚本
├── requirements.txt # Python 依赖
└── memories/        # 运行时自动创建
    └── memories.jsonl
```

## 配置

所有配置在 `config.py` 中，关键项：

```python
# 双模型
CONV_MODEL   = "qwen2.5vl:7b"     # 实时对话
BRAIN_MODEL  = "qwen2.5vl:72b"    # 后台大脑

# 功能开关（可单独关闭）
CAMERA_ENABLED = True
BRAIN_ENABLED  = True
MEMORY_ENABLED = True

# 大脑节奏
BRAIN_INTERVAL      = 20    # 多久思考一次（秒）
AUTONOMOUS_COOLDOWN = 120   # 两次主动说话的最小间隔（秒）
CONVERSATION_TIMEOUT = 30   # 沉默多久算一段对话结束
```

## 核心机制

### 视觉

后台线程每 3 秒从 USB 摄像头抓一帧（640x480 JPEG），供大脑场景理解使用。摄像头断开时自动降级为纯语音模式。

### 记忆

- **提取**：大脑检测到对话结束时，用 72b 从对话中提取重要信息（事实、偏好、事件），存入 `memories.jsonl`
- **检索**：每次大脑思考时，按 keyword 匹配 + importance 权重检索相关记忆，注入 7b 的 system prompt
- **防遗忘**：importance 权重远大于时间权重，重要的旧记忆不会被新记忆淹没
- **去重**：keyword 重叠 > 60% 的同类记忆自动合并
- **维护**：启动时清理 90 天以上的低重要性记忆

### 自主对话

大脑每轮思考输出一个 directive：
- `LISTEN` — 保持安静（大多数时候）
- `RESPOND` — 有人在跟我说话，应该回复
- `INITIATE:意图` — 我想主动说点什么

语音过滤：不是所有检测到的语音都会触发回复。快速规则（叫名字、30s 内有对话）+ 大脑 directive 判断是否该回应。

### 并发安全

- `asyncio.Lock` 互斥用户对话和自主说话
- `threading.Lock` 保护摄像头帧读写
- 两个模型分开调度，Ollama 并行处理
- `bot_speaking` + cooldown 防止自己的声音被麦克风拾取
