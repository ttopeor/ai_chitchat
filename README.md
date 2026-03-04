[中文版](README_zh.md)

# ai_chitchat — Xiao You

A voice chat companion with vision, memory, and autonomous speech.

Xiao You listens through the microphone, observes through a webcam, watches your computer screen, and talks back with her voice. She remembers important things across sessions, can search the web, and speaks up on her own when the moment is right — not just a passive voice assistant that waits to be asked.

## What Makes Xiao You Different

**She can see** — Continuously observes the environment via USB webcam and remotely captures your computer screen. Ask "what color am I wearing" or "what am I doing" and she finds the answer from the camera feed. She also comments when she notices something change, and can naturally bring up what she sees on your screen.

**She remembers** — Important things mentioned in conversations are extracted and stored persistently. Importance is weighted far more than recency, so something important you told her a month ago won't be buried by yesterday's small talk. High-importance memories are synthesized into a dynamic background profile that deepens over time.

**She can search** — Ask her to check the weather, news, or prices, and she'll search the web via DuckDuckGo and relay the results conversationally.

**She speaks up** — She doesn't just wait for you to talk. When she notices you're back, sees you doing something new, wants to break a long silence, or gets search results back, she decides the right moment and initiates on her own.

**Dual-system thinking** — Inspired by the brain's System 1 / System 2 theory. A fast system handles real-time responses while a slow system continuously observes, thinks, and decides in the background. The fast system keeps conversation fluid; the slow system keeps replies grounded.

**Flexible LLM backend** — Mouth (conversation) and brain (thinking) can each use a different provider and model. Run everything locally with Ollama, offload the brain to a cloud API, or use cloud for both — mix and match freely via `llm_config.yaml`.

## Dual-System Architecture

```
                     USB Camera    Windows Screen
                         |              |
                         v              v
    ============== BRAIN (slow system) ============
    ||                                            ||
    ||   camera + screen --> scene description    ||
    ||   conversation   --> memory extraction     ||
    ||   all context    --> speak directive       ||
    ||   user request   --> [SEARCH] web          ||
    ||                                            ||
    ==============|==========|=====================
                  |          |
           ContextBrief   INITIATE
                  |          |
                  v          v
    ============== MOUTH (fast system) ============
    ||                                            ||
    ||   mic --> VAD --> STT                      ||
    ||                    \                       ||
    ||      brief -------> LLM stream --> TTS ---+--> speaker
    ||                                            ||
    ================================================
```

- **Mouth (fast system)**: Handles real-time conversation. Receives the digested ContextBrief (scene, mood, guidance, search results) from the brain and generates spoken responses via streaming LLM + TTS pipeline.
- **Brain (slow system)**: Continuously observes the webcam and screen, analyzes conversation, retrieves memories, triggers web searches, and feeds digested scene understanding and conversation guidance to the mouth.

Mouth and brain are independent LLM clients. When both use the same Ollama model, dual-slot parallel inference (`OLLAMA_NUM_PARALLEL=2`) keeps them from blocking each other. When they use different providers (e.g., local Ollama for mouth, cloud API for brain), they run fully independently.

## Implementation Highlights

**Streaming voice pipeline** — LLM tokens stream out while sentences are split in real-time. Each sentence is immediately sent to TTS, and playback starts as soon as synthesis finishes. Three-stage pipeline (generation, synthesis, playback) runs in parallel — no waiting for the full response.

**Adaptive thinking rate** — During conversation, the brain thinks every few seconds to keep up. After conversation ends, it drops to every 20 seconds. After long idle, it slows to every 60 seconds, automatically saving GPU resources.

**Smart speech filtering** — Doesn't respond to every sound. Calling her name always triggers a response. Recent conversation (within 30s) counts as ongoing dialogue. The brain's judgment is trusted otherwise. Background conversations and talking to yourself are automatically ignored.

**Media awareness** — The brain detects when video/music is playing on screen and automatically distinguishes media audio from user speech. Prevents treating song lyrics or video narration as things said to her. Media audio during playback is buffered and extracted as memories.

**Echo suppression** — VAD threshold automatically increases during speech playback, with a cooldown period after, preventing self-echo. Optional PipeWire WebRTC AEC integration for further echo cancellation.

**Dynamic background** — At startup, high-importance memories are extracted and synthesized by LLM into a background description injected into the conversation prompt. Xiao You's understanding of you deepens with every conversation.

**Graceful degradation** — Camera, screen capture, memory, brain, and web search can each be independently disabled in `config.py`. No camera? Pure voice mode. No brain? Pure reactive conversation. Modules are decoupled.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Ollama (local) or OpenAI-compatible API (cloud) — configured per system |
| STT | faster-whisper (large-v3, CUDA) |
| TTS | ChatTTS (CUDA) |
| VAD | Silero VAD |
| Vision | OpenCV (USB webcam) + remote screen capture |
| Memory | JSONL + keyword/importance retrieval + dynamic background synthesis |
| Search | DuckDuckGo (ddgs) |

## System Requirements

- **GPU**: NVIDIA GPU with CUDA support (PyTorch CUDA 12.8). VRAM depends on model choice — a large local model (e.g., 122b) needs 96GB+; offloading brain to a cloud API reduces this significantly.
- **OS**: Linux with PipeWire audio
- **Ollama**: Required only if using Ollama as a provider
- **Microphone**: USB mic or system default audio input
- **Webcam**: USB webcam (optional, can be disabled in `config.py`)
- **Windows machine**: Running `screen_server.py` for screen capture (optional)

## Installation & Setup

### 1. Install the Project

```bash
git clone <repo-url>
cd ai_chitchat
bash setup.sh
```

`setup.sh` handles:
- Creating a Python venv
- Installing PyTorch (CUDA 12.8, Blackwell compatible)
- Installing remaining pip dependencies
- Optional: Configuring PipeWire WebRTC echo cancellation (interactive prompt at the end)

ChatTTS models (~1GB) are downloaded automatically on first run.

### 2. Configure LLM Providers

Edit `llm_config.yaml` to set up mouth and brain independently:

```yaml
# Mouth: real-time conversation (streaming, fed to TTS)
mouth:
  provider: ollama                      # "ollama" or "openai"
  base_url: "http://localhost:11434"
  api_key: ""
  model: "qwen3:32b"
  context_window: 128000
  max_output_tokens: 200
  keep_alive: -1                        # Ollama-only: pin model in VRAM
  think: false                          # Ollama-only: disable chain-of-thought

# Brain: background thinking + memory extraction (non-streaming)
brain:
  provider: openai                      # OpenAI-compatible API
  base_url: "https://api.deepseek.com"
  api_key: "${DEEPSEEK_API_KEY}"        # resolved from environment variable
  model: "deepseek-chat"
  context_window: 65536
  max_output_tokens: 500

token_estimator: simple   # "qwen" | "tiktoken" | "simple"
```

The `openai` provider works with any OpenAI-compatible endpoint (OpenAI, DeepSeek, Together, Groq, vLLM, etc.).

API keys support `${ENV_VAR}` syntax — the value is resolved from environment variables at startup. Set them in your `.bashrc` or `.env`:

```bash
export DEEPSEEK_API_KEY="sk-..."
```

**If using Ollama**, make sure Ollama is installed and running. If both mouth and brain use the same Ollama model, enable dual-slot parallel inference:

```bash
# Add to [Service] section of /etc/systemd/system/ollama.service:
Environment="OLLAMA_NUM_PARALLEL=2"

sudo systemctl daemon-reload
sudo systemctl restart ollama
```

Pull the model:

```bash
ollama pull qwen3:32b
```

### 3. Configuration (Optional)

Edit `config.py` for non-LLM settings:

- `MIC_DEVICE` / `SPEAKER_DEVICE` — Audio devices (`None` = system default)
- `CAMERA_ENABLED` / `SCREEN_ENABLED` / `BRAIN_ENABLED` / `MEMORY_ENABLED` / `TOOLS_ENABLED` — Feature toggles
- VAD thresholds, brain timing, memory limits, token budgets, etc.

List available audio devices:

```bash
python -c "import sounddevice; print(sounddevice.query_devices())"
```

### 4. Remote Screen Capture (Optional)

Run the screenshot server on a Windows machine so Xiao You can see your screen:

```bash
# Install dependencies on Windows
pip install mss Pillow

# Start the server (default port 7890, auto-selects widest monitor)
python screen_server.py

# Or specify parameters
python screen_server.py --port 7890 --monitor 1 --width 1920 --interval 2
```

Then configure in `config.py`:

```python
SCREEN_ENABLED = True
SCREEN_URL = "http://<windows-ip>:7890/screenshot"
```

### 5. Run

```bash
source .venv/bin/activate
python main.py
```

To run in English mode:

```bash
python main.py --lang en
```

Press Ctrl+C to exit. Ollama model VRAM is released automatically.

## Project Structure

```
ai_chitchat/
├── main.py            # Entry point, VoiceBot orchestration, audio pipeline
├── llm.py             # LLM abstraction layer (Ollama + OpenAI-compatible)
├── llm_config.yaml    # LLM provider/model configuration (mouth & brain)
├── config.py          # Non-LLM config: audio, vision, brain timing, memory, tools
├── brain.py           # BrainEngine — background thinker, ContextBrief, search dispatch
├── memory.py          # MemoryManager — persistent memory, extraction, retrieval
├── vision.py          # CameraCapture — USB webcam background capture
├── screen.py          # ScreenCapture — remote Windows screenshot fetching
├── screen_server.py   # Windows-side screenshot HTTP server
├── tools.py           # Tool definitions & execution (datetime, DuckDuckGo search)
├── i18n/              # Internationalization strings (Chinese/English)
├── setup.sh           # One-click install script
├── requirements.txt   # Python dependencies
├── logs/              # Runtime logs (auto-created)
└── memories/          # Persistent memory storage (auto-created)
```
