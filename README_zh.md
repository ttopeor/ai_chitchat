[English](README.md)

# ai_chitchat — 小悠

一个有视觉、记忆和自主意识的语音聊天伙伴。

小悠通过麦克风听你说话、用摄像头观察环境、看你的电脑屏幕、用声音回复你。她能记住跨会话的重要事情，能上网搜索信息，也能在合适的时机主动开口——不只是一个被动回答问题的语音助手。

## 小悠有什么不一样

**能看** — 通过 USB 摄像头持续观察周围环境，还能远程获取你的电脑屏幕截图。你问"我穿的什么颜色"、"你看我在干嘛"，她能直接从画面里找到答案。注意到场景变化时也会主动评论。看到你屏幕上在写代码或看视频，也能自然地提起。

**能记** — 对话中提到的重要事情会被提取并持久化存储。重要性权重远大于时间权重，所以一个月前告诉她的重要事情不会被昨天的闲聊淹没。高重要性记忆会被合成为动态背景，让小悠对你的了解越来越深。

**能搜索** — 你让她查天气、查新闻、查价格，她会通过 DuckDuckGo 搜索网络，然后用口语自然地告诉你结果。

**会主动说话** — 不是只等你开口。看到你回来了、发现你在做新的事情、沉默太久想搭个话、搜索结果回来了，她会自己判断时机主动开口。

**双系统思考** — 灵感来自人脑的 System 1 / System 2 理论。一个快系统负责即时反应，一个慢系统在后台持续观察、思考、决策。快系统保证对话流畅，慢系统保证回复有深度。

**灵活的 LLM 后端** — 嘴（对话）和脑（思考）可以各自使用不同的 provider 和模型。全部用 Ollama 本地跑、脑用云端 API、或者全部走云端——在 `llm_config.yaml` 中自由搭配。

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

- **嘴（快系统）**：负责实时对话，接收大脑消化好的 ContextBrief（场景、情绪、指导、搜索结果），通过流式 LLM + TTS 管线生成语音回复
- **脑（慢系统）**：在后台持续观察摄像头和屏幕、分析对话、检索记忆、发起网络搜索，把消化好的场景理解和对话指导喂给嘴

嘴和脑是独立的 LLM 客户端。两者使用同一个 Ollama 模型时，通过 `OLLAMA_NUM_PARALLEL=2` 双通道并行推理互不阻塞。两者使用不同 provider 时（比如嘴用本地 Ollama，脑用云端 API），则完全独立运行。

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
| LLM | Ollama（本地）或 OpenAI 兼容 API（云端）— 按系统独立配置 |
| STT | faster-whisper (large-v3, CUDA) |
| TTS | ChatTTS (CUDA) |
| VAD | Silero VAD |
| 视觉 | OpenCV (USB 摄像头) + 远程屏幕截图 |
| 记忆 | JSONL + keyword/importance 检索 + 动态背景合成 |
| 搜索 | DuckDuckGo (ddgs) |

## 系统要求

- **GPU**：NVIDIA GPU，需要 CUDA 支持（PyTorch CUDA 12.8）。显存取决于模型选择——大型本地模型（如 122b）需要 96GB+；将脑卸载到云端 API 可显著降低显存需求。
- **系统**：Linux，PipeWire 音频
- **Ollama**：仅在使用 Ollama 作为 provider 时需要
- **麦克风**：USB 麦克风或系统默认音频输入
- **摄像头**：USB 摄像头（可选，可在 `config.py` 中关闭）
- **Windows 机器**：运行 `screen_server.py` 提供屏幕截图（可选）

## 安装与设置

### 1. 安装项目

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

### 2. 配置 LLM Provider

编辑 `llm_config.yaml`，分别设置嘴和脑：

```yaml
# 嘴：实时对话（流式输出，喂给 TTS）
mouth:
  provider: ollama                      # "ollama" 或 "openai"
  base_url: "http://localhost:11434"
  api_key: ""
  model: "qwen3:32b"
  context_window: 128000
  max_output_tokens: 200
  keep_alive: -1                        # Ollama 专用：常驻显存
  think: false                          # Ollama 专用：关闭思维链

# 脑：后台思考 + 记忆提取（非流式，带视觉）
brain:
  provider: openai
  base_url: "https://generativelanguage.googleapis.com/v1beta/openai"
  api_key: "${GEMINI_API_KEY}"
  model: "gemini-3-flash-preview"
  context_window: 1000000
  max_output_tokens: 500
  vision: true                          # 不支持图片的模型设为 false

token_estimator: simple   # "qwen" | "tiktoken" | "simple"
```

`openai` provider 兼容所有 OpenAI 格式的端点（OpenAI、Gemini、DeepSeek、Together、Groq、vLLM 等）。`base_url` 需要包含版本路径：

| Provider | base_url |
|----------|----------|
| Gemini | `https://generativelanguage.googleapis.com/v1beta/openai` |
| DeepSeek | `https://api.deepseek.com/v1` |
| OpenAI | `https://api.openai.com/v1` |

API key 支持 `${ENV_VAR}` 语法，启动时从环境变量解析：

```bash
export GEMINI_API_KEY="..."
```

**如果使用 Ollama**，请确保 Ollama 已安装并运行。如果嘴和脑都用同一个 Ollama 模型，需要启用双通道并行推理：

```bash
# /etc/systemd/system/ollama.service 的 [Service] 段加:
Environment="OLLAMA_NUM_PARALLEL=2"

sudo systemctl daemon-reload
sudo systemctl restart ollama
```

拉取模型：

```bash
ollama pull qwen3:32b
```

### 3. 配置（可选）

编辑 `config.py` 调整非 LLM 设置：

- `MIC_DEVICE` / `SPEAKER_DEVICE` — 音频设备（`None` = 系统默认）
- `CAMERA_ENABLED` / `SCREEN_ENABLED` / `BRAIN_ENABLED` / `MEMORY_ENABLED` / `TOOLS_ENABLED` — 功能开关
- VAD 阈值、大脑时序、记忆限制、token 预算等

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

退出时按 Ctrl+C，Ollama 模型显存会自动释放。

## 项目结构

```
ai_chitchat/
├── main.py            # 主入口，VoiceBot 编排，音频管线
├── llm.py             # LLM 抽象层（Ollama + OpenAI 兼容）
├── llm_config.yaml    # LLM provider/模型配置（嘴和脑独立）
├── config.py          # 非 LLM 配置：音频、视觉、大脑时序、记忆、工具
├── brain.py           # BrainEngine — 后台大脑，ContextBrief 生成，搜索调度
├── memory.py          # MemoryManager — 持久记忆，提取、检索与整合
├── vision.py          # CameraCapture — USB 摄像头后台采集
├── screen.py          # ScreenCapture — 远程 Windows 屏幕截图获取
├── screen_server.py   # Windows 端屏幕截图 HTTP 服务
├── tools.py           # 工具定义与执行（时间查询、DuckDuckGo 搜索）
├── i18n/              # 国际化字符串（中文/英文）
├── setup.sh           # 一键安装脚本
├── requirements.txt   # Python 依赖
├── logs/              # 运行时日志（自动创建）
└── memories/          # 持久记忆存储（自动创建）
```
