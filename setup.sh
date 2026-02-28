#!/usr/bin/env bash
# One-time setup for ai_chitchat
#   - Python venv + pip deps (includes ChatTTS)
#   - PipeWire AEC for speaker use  (optional, see bottom)
set -e
cd "$(dirname "$0")"

# ── 1. Python venv ────────────────────────────────────────────────────────────
echo "[1/2] Setting up Python venv…"
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

pip install --upgrade pip setuptools wheel -q

# ── 2. Dependencies ───────────────────────────────────────────────────────────
echo "[2/2] Installing dependencies…"
# Install torch family from CUDA 12.4 index — must install together to guarantee
# version coherence (torchvision from PyPI can mismatch torch from CUDA index).
python3 -c "
import torch, torchvision
assert torch.cuda.is_available()
getattr(torch.library, 'register_fake')
torch.zeros(1, device='cuda')  # actually execute a CUDA kernel — catches sm_120 mismatch
" 2>/dev/null || {
    echo "  Installing PyTorch family with CUDA 12.8 support (Blackwell-compatible)…"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 -q
}
# Install remaining deps (torch family handled above)
grep -vE "^torch|^torchvision|^torchaudio" requirements.txt > /tmp/main_req.txt
pip install -r /tmp/main_req.txt -q
rm -f /tmp/main_req.txt

echo ""
echo "=========================================="
echo "  Setup complete!"
echo "  Activate venv:  source .venv/bin/activate"
echo "  Run:            python main.py"
echo "  Note: ChatTTS model (~1 GB) downloads on first run."
echo "=========================================="
echo ""

# ── PipeWire AEC (optional — for speaker use with interruption) ───────────────
read -rp "Set up PipeWire AEC for speaker echo cancellation? [y/N] " ans
if [[ "$ans" =~ ^[Yy]$ ]]; then
    echo ""
    echo "Configuring PipeWire WebRTC echo cancellation…"
    mkdir -p ~/.config/pipewire/pipewire.conf.d
    cat > ~/.config/pipewire/pipewire.conf.d/99-echo-cancel.conf <<'EOF'
context.modules = [
  {
    name = libpipewire-module-echo-cancel
    args = {
      library.name  = aec/libspa-aec-webrtc
      aec.args = {
        webrtc.gain_control = true
      }
      capture.props = {
        node.name = "capture.aec"
      }
      source.props = {
        node.name        = "aec-source"
        node.description = "Echo Cancellation Source"
      }
      sink.props = {
        node.name        = "aec-sink"
        node.description = "Echo Cancellation Sink"
      }
      playback.props = {
        node.name = "playback.aec"
      }
    }
  }
]
EOF
    systemctl --user restart pipewire
    echo ""
    echo "  PipeWire AEC enabled."
    echo "  In config.py, set:  MIC_DEVICE = \"Echo Cancellation Source\""
    echo "  (Run 'python -c \"import sounddevice; print(sounddevice.query_devices())\"'"
    echo "   to confirm the exact device name after restarting PipeWire.)"
fi
