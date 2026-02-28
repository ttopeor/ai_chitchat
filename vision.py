"""
CameraCapture — background thread that continuously grabs frames from a USB webcam.

Provides the latest frame as a base64-encoded JPEG for VLM consumption.
Gracefully degrades (returns None) when the camera is unavailable.
"""

import base64
import threading
import time

import cv2
import numpy as np


class CameraCapture:
    def __init__(
        self,
        device_index: int = 0,
        width: int = 640,
        height: int = 480,
        interval: float = 3.0,
    ):
        self._device_index = device_index
        self._width = width
        self._height = height
        self._interval = interval

        self._lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._latest_ts: float = 0.0
        self._running = False
        self._thread: threading.Thread | None = None

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    # ── capture loop ──────────────────────────────────────────────────────────

    def _capture_loop(self) -> None:
        cap = cv2.VideoCapture(self._device_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)

        if not cap.isOpened():
            print("[Vision] WARNING: cannot open camera — vision disabled")
            return

        print(f"[Vision] Camera opened (device {self._device_index}, "
              f"{self._width}x{self._height}, interval {self._interval}s)")
        try:
            while self._running:
                ret, frame = cap.read()
                if ret:
                    with self._lock:
                        self._latest_frame = frame
                        self._latest_ts = time.monotonic()
                time.sleep(self._interval)
        finally:
            cap.release()
            print("[Vision] Camera released")

    # ── public API ────────────────────────────────────────────────────────────

    def get_latest_frame_b64(self, max_age: float = 10.0) -> str | None:
        """Return the latest frame as a base64 JPEG string, or None."""
        with self._lock:
            if self._latest_frame is None:
                return None
            if time.monotonic() - self._latest_ts > max_age:
                return None
            ok, buf = cv2.imencode(
                ".jpg", self._latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 70]
            )
            if not ok:
                return None
            return base64.b64encode(buf).decode("ascii")

    def get_latest_frame_raw(self) -> np.ndarray | None:
        """Return a copy of the latest frame as a raw BGR numpy array."""
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()
