"""
screen_server.py — Windows-side screenshot HTTP server.

Captures a specific monitor (default: widest = 21:9) and serves the latest
screenshot as JPEG via HTTP.  Designed to run on the user's Windows machine
so that 小悠 (on a remote Ubuntu box) can GET the screen each think cycle.

Usage:
    pip install mss Pillow
    python screen_server.py [--port 7890] [--monitor N] [--width 1280] [--interval 2]

Endpoints:
    GET /screenshot  → latest JPEG (image/jpeg)
    GET /health      → {"status": "ok", "monitor": ..., "size": ...}
"""

import argparse
import io
import json
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

import mss
from PIL import Image


# ── Globals (set by main, read by handler & capture thread) ──────────────────

_latest_jpeg: bytes = b""
_lock = threading.Lock()
_monitor_info: dict = {}


# ── Capture thread ───────────────────────────────────────────────────────────

def _capture_loop(
    monitor_index: int,
    target_width: int,
    interval: float,
) -> None:
    global _latest_jpeg

    with mss.mss() as sct:
        mon = sct.monitors[monitor_index]
        print(f"[Capture] Monitor {monitor_index}: "
              f"{mon['width']}x{mon['height']} at ({mon['left']}, {mon['top']})")

        while True:
            try:
                raw = sct.grab(mon)
                img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")

                # Scale down preserving aspect ratio
                if img.width > target_width:
                    ratio = target_width / img.width
                    new_h = int(img.height * ratio)
                    img = img.resize((target_width, new_h), Image.LANCZOS)

                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=70)
                jpeg_bytes = buf.getvalue()

                with _lock:
                    _latest_jpeg = jpeg_bytes

            except Exception as e:
                print(f"[Capture] Error: {e}")

            time.sleep(interval)


# ── HTTP handler ─────────────────────────────────────────────────────────────

class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/screenshot":
            with _lock:
                data = _latest_jpeg
            if not data:
                self.send_response(503)
                self.end_headers()
                self.wfile.write(b"No screenshot yet")
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        elif self.path == "/health":
            body = json.dumps({
                "status": "ok",
                "monitor": _monitor_info,
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Silence per-request logs
        pass


# ── Monitor selection ────────────────────────────────────────────────────────

def _pick_monitor(preferred: int | None) -> int:
    """Pick monitor index (1-based, as mss uses 0 for 'all screens')."""
    with mss.mss() as sct:
        monitors = sct.monitors[1:]  # skip index 0 (virtual all-screens)
        print(f"[Setup] Found {len(monitors)} monitor(s):")
        for i, m in enumerate(monitors, 1):
            label = "  → " if (preferred and i == preferred) else "    "
            print(f"{label}[{i}] {m['width']}x{m['height']} "
                  f"at ({m['left']}, {m['top']})")

        if preferred and 1 <= preferred <= len(monitors):
            return preferred

        # Auto-pick widest monitor (likely the 21:9)
        widest_idx = max(range(len(monitors)), key=lambda i: monitors[i]["width"])
        chosen = widest_idx + 1  # 1-based
        print(f"[Setup] Auto-selected monitor {chosen} "
              f"({monitors[widest_idx]['width']}x{monitors[widest_idx]['height']})")
        return chosen


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    global _monitor_info

    parser = argparse.ArgumentParser(description="Screenshot HTTP server")
    parser.add_argument("--port", type=int, default=7890)
    parser.add_argument("--monitor", type=int, default=None,
                        help="Monitor number (1-based). Default: widest.")
    parser.add_argument("--width", type=int, default=1280,
                        help="Scale target width in pixels.")
    parser.add_argument("--interval", type=float, default=2.0,
                        help="Capture interval in seconds.")
    args = parser.parse_args()

    monitor_index = _pick_monitor(args.monitor)

    with mss.mss() as sct:
        mon = sct.monitors[monitor_index]
        _monitor_info = {
            "index": monitor_index,
            "width": mon["width"],
            "height": mon["height"],
        }

    # Start capture thread
    t = threading.Thread(
        target=_capture_loop,
        args=(monitor_index, args.width, args.interval),
        daemon=True,
    )
    t.start()

    # Wait for first capture
    print("[Setup] Waiting for first capture…")
    while True:
        with _lock:
            if _latest_jpeg:
                break
        time.sleep(0.1)
    print(f"[Setup] First capture OK ({len(_latest_jpeg)} bytes)")

    # Start HTTP server
    server = HTTPServer(("0.0.0.0", args.port), _Handler)
    print(f"\n[Server] Listening on http://0.0.0.0:{args.port}/screenshot")
    print("[Server] Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Server] Shutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
