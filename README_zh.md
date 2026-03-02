[English](README.md)

# ai_chitchat — 小悠

一个有视觉、记忆和自主意识的本地语音聊天伙伴。

小悠通过麦克风听你说话、用摄像头观察环境、看你的电脑屏幕、用声音回复你。她能记住跨会话的重要事情，能上网搜索信息，也能在合适的时机主动开口——不只是一个被动回答问题的语音助手。

## 小悠有什么不一样

**能看** — 通过 USB 摄像头持续观察周围环境，还能远程获取你的电脑屏幕截图。你问"我穿的什么颜色"、"你看我在干嘛"，她能直接从画面里找到答案。注意到场景变化时也会主动评论。看到你屏幕上在写代码或看视频，也能自然地提起。

**能记** — 对话中提到的重要事情会被提取并持久化存储。重要性权重远大于时间权重，所以一个月前告诉她的重要事情不会被昨天的闲聊淹没。高重要性记忆会被合成为动态背景，让小悠对你的了解越来越深。

**能搜索** — 你让她查天气、查新闻、查价格，她会通过 DuckDuckGo 搜索网络，然后用口语自然地告诉你结果。

**会主动说话** — 不是只等你开口。看到你回来了、发现你在做新的事情、沉默太久想搭个话、搜索结果回来了，她会自己判断时机主动开口。

**双系统思考** — 灵感来自人脑的 System 1 / System 2 理论。一个快系统负责即时反应，一个慢系统在后台持续观察、思考、决策。快系统保证对话流畅，慢系统保证回复有深度。

**完全本地** — 所有模型推理在本地 GPU 上完成，语音、图像、对话内容不上传任何外部服务（网络搜索除外）。

## 双系统架构

```
                     USB Camera    Windows Screen
                         |              |
                         v              v
    ============== BRAIN (慢系统) ===============
    ||                                         ||
    ||   camera + screen --> scene description ||
    ||   conversation   --> memory extraction  ||
    ||   all context    --> speak directive     ||
    ||   user request   --> [SEARCH] web       ||
    ||                                         ||
    ==============|==========|==================
                  |          |
           ContextBrief   INITIATE
                  |          |
                  v          v
    ============== MOUTH (快系统) ===============
    ||                                         ||
    ||   mic --> VAD --> STT                   ||
    ||                    \                    ||
    ||      brief -------> LLM stream --> TTS --+--> speaker
    ||                                         ||
    =============================================
```

- **嘴（快系统）**：负责实时对话，接收大脑消化好的 ContextBrief（场景、情绪、指导、搜索结果），直接生成回复并语音输出
- **脑（慢系统）**：在后台持续观察摄像头和屏幕、分析对话、检索记忆、发起网络搜索，把消化好的场景理解和对话指导喂给嘴

两个系统通过 Ollama 的 `OLLAMA_NUM_PARALLEL=2` 实现同一模型双通道并行推理，互不阻塞。

## 实现特点

**流式语音管线** — LLM 流式输出的同时实时切句，切出一句立刻送 TTS 合成，合成完立刻播放。生成、合成、播放三级流水线并行，不用等整段回复生成完才开口。

**自适应思考频率** — 对话中大脑每 0.5 秒思考一次保持跟进；对话结束后降到 20 秒一次；长时间无人时降到 60 秒一次，自动节省 GPU 资源。

**智能语音过滤** — 不是听到声音就回复。叫名字一定回复，30 秒内有过对话视为连续对话会回复，大脑判断应该回复时回复，其他情况（背景对话、自言自语）自动忽略。

**媒体感知** — 大脑能识别屏幕上是否在播放视频/音乐，自动区分媒体声音和用户说话，避免把歌词或视频旁白当成对你说的话。媒体播放期间的语音内容会被缓冲并提取为记忆。

**回声抑制** — 播放语音时自动提高 VAD 阈值，播放结束后设置冷却期，防止把自己的声音当成用户输入。可选搭配 PipeWire WebRTC AEC 进一步消除回声。

**动态背景** — 启动时从持久记忆中提取高重要性信息，用 LLM 合成一段关于你的背景描述注入到对话 Prompt 中，让小悠的了解随对话积累而加深。

**优雅降级** — 摄像头、屏幕截图、记忆、大脑、网络搜索五个模块可在 `config.py` 中独立关闭。没有摄像头就是纯语音模式，关掉大脑就是纯反应式对话，各模块解耦互不影响。

## 技术栈

| 组件 | 技术 |
|------|------|
| LLM | qwen3.5:122b (Ollama, 双通道并行) |
| STT | faster-whisper (large-v3, CUDA) |
| TTS | ChatTTS (CUDA) |
| VAD | Silero VAD |
| 视觉 | OpenCV (USB 摄像头) + 远程屏幕截图 |
| 记忆 | JSONL + keyword/importance 检索 + 动态背景合成 |
| 搜索 | DuckDuckGo (ddgs) |

## 系统要求

- **GPU**：NVIDIA GPU，96GB+ 显存（双通道 122b 模型 + STT/TTS/VAD）
- **系统**：Linux，PipeWire 音频
- **CUDA**：需要 CUDA 支持（PyTorch CUDA 12.8）
- **Ollama**：已安装并运行 Ollama 服务
- **麦克风**：USB 麦克风或系统默认音频输入
- **摄像头**：USB 摄像头（可选，可在 `config.py` 中关闭）
- **Windows 机器**：运行 `screen_server.py` 提供屏幕截图（可选）

## 安装与设置

### 1. 配置 Ollama 服务端

双通道并行推理需要 Ollama 允许同一模型多路并发：

```bash
# /etc/systemd/system/ollama.service 的 [Service] 段加:
Environment="OLLAMA_NUM_PARALLEL=2"

sudo systemctl daemon-reload
sudo systemctl restart ollama
```

拉取模型：

```bash
ollama pull qwen3.5:122b
```

### 2. 安装项目

```bash
git clone <repo-url>
cd ai_chitchat
bash setup.sh
```

`setup.sh` 会自动完成：
- 创建 Python venv
- 安装 PyTorch（CUDA 12.8，Blackwell 兼容）
- 安装其余 pip 依赖
- 可选：配置 PipeWire WebRTC 回声消除（脚本末尾会交互式询问）

ChatTTS 模型（~1GB）会在首次运行时自动下载。

### 3. 配置（可选）

编辑 `config.py` 按需调整：

- `LLM_BASE_URL` — Ollama 服务地址（默认 `http://10.0.0.190:11434/v1`）
- `MIC_DEVICE` / `SPEAKER_DEVICE` — 音频设备（`None` = 系统默认）
- `CAMERA_ENABLED` / `SCREEN_ENABLED` / `BRAIN_ENABLED` / `MEMORY_ENABLED` / `TOOLS_ENABLED` — 功能开关

查看可用音频设备：

```bash
python -c "import sounddevice; print(sounddevice.query_devices())"
```

### 4. 设置远程屏幕截图（可选）

在 Windows 机器上运行屏幕截图服务，让小悠能看到你的电脑屏幕：

```bash
# Windows 端安装依赖
pip install mss Pillow

# 启动服务（默认端口 7890，自动选择最宽显示器）
python screen_server.py

# 或指定参数
python screen_server.py --port 7890 --monitor 1 --width 1920 --interval 2
```

然后在 `config.py` 中配置：

```python
SCREEN_ENABLED = True
SCREEN_URL = "http://<windows-ip>:7890/screenshot"
```

### 5. 运行

```bash
source .venv/bin/activate
python main.py
```

英语模式启动：

```bash
python main.py --lang en
```

退出时按 Ctrl+C，模型显存会自动释放。

## 项目结构

```
ai_chitchat/
├── main.py            # 主入口，VoiceBot 编排，音频管线
├── config.py          # 所有配置：模型、音频、视觉、大脑、记忆、工具
├── brain.py           # BrainEngine — 后台大脑，ContextBrief 生成，搜索调度
├── memory.py          # MemoryManager — 持久记忆，提取、检索与整合
├── vision.py          # CameraCapture — USB 摄像头后台采集
├── screen.py          # ScreenCapture — 远程 Windows 屏幕截图获取
├── screen_server.py   # Windows 端屏幕截图 HTTP 服务（在 Windows 上运行）
├── tools.py           # 工具定义与执行（时间查询、DuckDuckGo 搜索）
├── i18n/              # 国际化字符串（中文/英文）
├── setup.sh           # 一键安装脚本
├── requirements.txt   # Python 依赖
├── logs/              # 运行时日志（自动创建）
└── memories/          # 持久记忆存储（自动创建）
```
