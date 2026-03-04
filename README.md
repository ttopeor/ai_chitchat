[中文版](README_zh.md)

# ai_chitchat — Xiao You

A local voice chat companion with vision, memory, and autonomous speech.

Xiao You listens through the microphone, observes through a webcam, watches your computer screen, and talks back with her voice. She remembers important things across sessions, can search the web, and speaks up on her own when the moment is right — not just a passive voice assistant that waits to be asked.

## What Makes Xiao You Different

**She can see** — Continuously observes the environment via USB webcam and remotely captures your computer screen. Ask "what color am I wearing" or "what am I doing" and she finds the answer from the camera feed. She also comments when she notices something change, and can naturally bring up what she sees on your screen.

**She remembers** — Important things mentioned in conversations are extracted and stored persistently. Importance is weighted far more than recency, so something important you told her a month ago won't be buried by yesterday's small talk. High-importance memories are synthesized into a dynamic background profile that deepens over time.

**She can search** — Ask her to check the weather, news, or prices, and she'll search the web via DuckDuckGo and relay the results conversationally.

**She speaks up** — She doesn't just wait for you to talk. When she notices you're back, sees you doing something new, wants to break a long silence, or gets search results back, she decides the right moment and initiates on her own.

**Dual-system thinking** — Inspired by the brain's System 1 / System 2 theory. A fast system handles real-time responses while a slow system continuously observes, thinks, and decides in the background. The fast system keeps conversation fluid; the slow system keeps replies grounded.

**Fully local** — All model inference runs on local GPUs. Voice, images, and conversation data never leave your machine (except for web searches).

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

- **Mouth (fast system)**: Handles real-time conversation. Receives the digested ContextBrief (scene, mood, guidance, search results) from the brain and generates spoken responses.
- **Brain (slow system)**: Continuously observes the webcam and screen, analyzes conversation, retrieves memories, triggers web searches, and feeds digested scene understanding and conversation guidance to the mouth.

Both systems run the same model via Ollama's `OLLAMA_NUM_PARALLEL=2` for dual-slot parallel inference, never blocking each other.

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
| LLM | qwen3.5:122b (Ollama, dual-slot parallel) |
| STT | faster-whisper (large-v3, CUDA) |
| TTS | ChatTTS (CUDA) |
| VAD | Silero VAD |
| Vision | OpenCV (USB webcam) + remote screen capture |
| Memory | JSONL + keyword/importance retrieval + dynamic background synthesis |
| Search | DuckDuckGo (ddgs) |

## System Requirements

- **GPU**: NVIDIA GPU, 96GB+ VRAM (dual-slot 122b model + STT/TTS/VAD)
- **OS**: Linux with PipeWire audio
- **CUDA**: CUDA support required (PyTorch CUDA 12.8)
- **Ollama**: Installed and running
- **Microphone**: USB mic or system default audio input
- **Webcam**: USB webcam (optional, can be disabled in `config.py`)
- **Windows machine**: Running `screen_server.py` for screen capture (optional)

## Installation & Setup

### 1. Configure Ollama Server

Dual-slot parallel inference requires Ollama to allow concurrent requests to the same model:

```bash
# Add to [Service] section of /etc/systemd/system/ollama.service:
Environment="OLLAMA_NUM_PARALLEL=2"

sudo systemctl daemon-reload
sudo systemctl restart ollama
```

Pull the model:

```bash
ollama pull qwen3.5:122b
```

### 2. Install the Project

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

### 3. Configuration (Optional)

Edit `config.py` as needed:

- `LLM_BASE_URL` — Ollama server address (default `http://10.0.0.190:11434/v1`)
- `MIC_DEVICE` / `SPEAKER_DEVICE` — Audio devices (`None` = system default)
- `CAMERA_ENABLED` / `SCREEN_ENABLED` / `BRAIN_ENABLED` / `MEMORY_ENABLED` / `TOOLS_ENABLED` — Feature toggles

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

Press Ctrl+C to exit. Model VRAM is released automatically.

## Project Structure

```
ai_chitchat/
├── main.py            # Entry point, VoiceBot orchestration, audio pipeline
├── config.py          # All config: models, audio, vision, brain, memory, tools
├── brain.py           # BrainEngine — background thinker, ContextBrief generation, search dispatch
├── memory.py          # MemoryManager — persistent memory, extraction, retrieval & consolidation
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
