"""
ScreenCapture — async fetcher for remote Windows screenshot.

Periodically GETs a JPEG from the Windows-side screen_server.py HTTP endpoint,
caches the latest frame as base64 for BrainEngine consumption.

Mirrors the CameraCapture interface (get_latest_frame_b64) so brain.py can
treat both sources uniformly.
"""

import asyncio
import base64
import time

import httpx


class ScreenCapture:
    def __init__(self, url: str, interval: float = 3.0):
        self._url = url
        self._interval = interval

        self._latest_b64: str | None = None
        self._latest_ts: float = 0.0
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._fetch_loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def get_latest_frame_b64(self, max_age: float = 10.0) -> str | None:
        if self._latest_b64 is None:
            return None
        if time.monotonic() - self._latest_ts > max_age:
            return None
        return self._latest_b64

    async def _fetch_loop(self) -> None:
        async with httpx.AsyncClient() as client:
            while True:
                try:
                    resp = await client.get(
                        self._url,
                        timeout=httpx.Timeout(connect=5, read=10, write=5, pool=5),
                    )
                    if resp.status_code == 200:
                        self._latest_b64 = base64.b64encode(resp.content).decode("ascii")
                        self._latest_ts = time.monotonic()
                except Exception as e:
                    print(f"[Screen] fetch error: {e}")

                await asyncio.sleep(self._interval)
