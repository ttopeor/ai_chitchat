"""
LLM abstraction layer — unified interface for Ollama and OpenAI-compatible providers.

Module-level clients: `mouth` and `brain`, configured from llm_config.yaml.
Memory extraction reuses `brain`.

Call sites:
  main.py   → llm.mouth.stream_chat(messages) → AsyncIterator[str]
  brain.py  → llm.brain.chat(messages, images) → (content, thinking)
  memory.py → llm.brain.chat(messages)         → (content, thinking)
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator

import httpx
import yaml


# ── Config dataclass ────────────────────────────────────────────────────────

@dataclass
class LLMConfig:
    provider: str           # "ollama" | "openai"
    base_url: str
    api_key: str
    model: str
    context_window: int
    max_output_tokens: int
    keep_alive: int | str = -1      # Ollama only
    think: bool = False              # Ollama only
    extra: dict = field(default_factory=dict)


# ── Abstract base ───────────────────────────────────────────────────────────

class LLMClient(ABC):
    """Abstract LLM client — one instance per role (mouth / brain)."""

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.model = cfg.model
        self.context_window = cfg.context_window
        self.max_output_tokens = cfg.max_output_tokens

    @abstractmethod
    async def chat(
        self,
        messages: list[dict],
        images: list[str] | None = None,
    ) -> tuple[str, str]:
        """Non-streaming chat.  Returns (content, thinking)."""
        ...

    @abstractmethod
    def stream_chat(
        self,
        messages: list[dict],
    ) -> AsyncIterator[str]:
        """Streaming chat.  Yields content tokens."""
        ...

    async def load_model(self) -> None:
        """Pin/preload model.  No-op for providers that don't support it."""

    async def unload_model(self) -> None:
        """Unpin/unload model.  No-op for providers that don't support it."""


# ── Ollama (native /api/chat) ──────────────────────────────────────────────

class OllamaClient(LLMClient):

    def __init__(self, cfg: LLMConfig):
        super().__init__(cfg)
        self._base = cfg.base_url.rstrip("/")

    # -- non-streaming ---------------------------------------------------------

    async def chat(
        self,
        messages: list[dict],
        images: list[str] | None = None,
    ) -> tuple[str, str]:
        if images and messages:
            # Inject images into the last user message (Ollama native format)
            msgs = [m.copy() for m in messages]
            for m in reversed(msgs):
                if m["role"] == "user":
                    m["images"] = images
                    break
            messages = msgs

        body: dict = {
            "model": self.model,
            "messages": messages,
            "options": {"num_ctx": self.context_window},
            "stream": False,
            "think": self.cfg.think,
        }

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self._base}/api/chat",
                json=body,
                timeout=httpx.Timeout(connect=10, read=120, write=10, pool=10),
            )
            resp.raise_for_status()
            data = resp.json()
            msg = data.get("message", {})
            content = msg.get("content", "").strip()
            thinking = msg.get("thinking", "").strip()
            return content, thinking

    # -- streaming -------------------------------------------------------------

    async def stream_chat(
        self,
        messages: list[dict],
    ) -> AsyncIterator[str]:
        body: dict = {
            "model": self.model,
            "messages": messages,
            "options": {
                "num_ctx": self.context_window,
                "num_predict": self.max_output_tokens,
            },
            "stream": True,
            "think": self.cfg.think,
        }

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{self._base}/api/chat",
                json=body,
                timeout=httpx.Timeout(connect=10, read=120, write=10, pool=10),
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    tok = data.get("message", {}).get("content", "")
                    if tok:
                        yield tok
                    if data.get("done", False):
                        return

    # -- model lifecycle -------------------------------------------------------

    async def load_model(self) -> None:
        body: dict = {
            "model": self.model,
            "keep_alive": self.cfg.keep_alive,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }
        body["options"] = {"num_ctx": self.context_window}
        try:
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST",
                    f"{self._base}/api/chat",
                    json=body,
                    timeout=httpx.Timeout(connect=10, read=300, write=10, pool=10),
                ) as resp:
                    resp.raise_for_status()
                    async for _ in resp.aiter_lines():
                        pass
        except Exception as e:
            print(f"  [Ollama] load({self.model}) failed: {e}")

    async def unload_model(self) -> None:
        body = {"model": self.model, "keep_alive": 0}
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{self._base}/api/generate",
                    json=body,
                    timeout=httpx.Timeout(connect=10, read=30, write=10, pool=10),
                )
                resp.raise_for_status()
        except Exception as e:
            print(f"  [Ollama] unload({self.model}) failed: {e}")


# ── OpenAI-compatible (/v1/chat/completions) ────────────────────────────────

class OpenAICompatClient(LLMClient):

    def __init__(self, cfg: LLMConfig):
        super().__init__(cfg)
        self._base = cfg.base_url.rstrip("/")
        self._headers = {"Content-Type": "application/json"}
        if cfg.api_key:
            self._headers["Authorization"] = f"Bearer {cfg.api_key}"

    @staticmethod
    def _convert_images(messages: list[dict], images: list[str]) -> list[dict]:
        """Convert images into OpenAI vision format on the last user message."""
        msgs = [m.copy() for m in messages]
        for m in reversed(msgs):
            if m["role"] == "user":
                content_parts: list[dict] = [
                    {"type": "text", "text": m["content"]}
                ]
                for img_b64 in images:
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                    })
                m["content"] = content_parts
                break
        return msgs

    # -- non-streaming ---------------------------------------------------------

    async def chat(
        self,
        messages: list[dict],
        images: list[str] | None = None,
    ) -> tuple[str, str]:
        if images and messages:
            messages = self._convert_images(messages, images)

        body: dict = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_output_tokens,
            "stream": False,
        }

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self._base}/v1/chat/completions",
                json=body,
                headers=self._headers,
                timeout=httpx.Timeout(connect=10, read=120, write=10, pool=10),
            )
            resp.raise_for_status()
            data = resp.json()
            choice = data.get("choices", [{}])[0]
            content = choice.get("message", {}).get("content", "").strip()
            return content, ""

    # -- streaming -------------------------------------------------------------

    async def stream_chat(
        self,
        messages: list[dict],
    ) -> AsyncIterator[str]:
        body: dict = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_output_tokens,
            "stream": True,
        }

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{self._base}/v1/chat/completions",
                json=body,
                headers=self._headers,
                timeout=httpx.Timeout(connect=10, read=120, write=10, pool=10),
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data: "):
                        continue
                    payload = line[6:]  # strip "data: "
                    if payload == "[DONE]":
                        return
                    try:
                        data = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    delta = (
                        data.get("choices", [{}])[0]
                        .get("delta", {})
                        .get("content", "")
                    )
                    if delta:
                        yield delta


# ── Token estimation ────────────────────────────────────────────────────────

_token_estimator: str = "qwen"


def estimate_tokens(text: str) -> int:
    """Estimate token count based on configured estimator."""
    if not text:
        return 0
    if _token_estimator == "qwen":
        # Qwen tokenizer: CJK ≈ 1.0 tok/char, ASCII ≈ 0.3 tok/char (1.3x safety)
        n_cjk = sum(1 for c in text if ord(c) > 0x2E7F)
        n_other = len(text) - n_cjk
        return int(n_cjk * 1.0 + n_other * 0.3) + 4
    else:  # "tiktoken" / "simple"
        return int(len(text) * 0.4) + 4


# ── Module-level init ───────────────────────────────────────────────────────

_CONFIG_PATH = Path(__file__).parent / "llm_config.yaml"

mouth: LLMClient | None = None
brain: LLMClient | None = None


def _parse_section(section: dict) -> LLMConfig:
    known_keys = {
        "provider", "base_url", "api_key", "model",
        "context_window", "max_output_tokens", "keep_alive", "think",
    }
    return LLMConfig(
        provider=section.get("provider", "ollama"),
        base_url=section.get("base_url", "").rstrip("/"),
        api_key=section.get("api_key", ""),
        model=section["model"],
        context_window=section.get("context_window", 128000),
        max_output_tokens=section.get("max_output_tokens", 500),
        keep_alive=section.get("keep_alive", -1),
        think=section.get("think", False),
        extra={k: v for k, v in section.items() if k not in known_keys},
    )


def _make_client(cfg: LLMConfig) -> LLMClient:
    if cfg.provider == "ollama":
        return OllamaClient(cfg)
    elif cfg.provider == "openai":
        return OpenAICompatClient(cfg)
    else:
        raise ValueError(f"Unknown LLM provider: {cfg.provider!r}")


def init() -> None:
    """Load llm_config.yaml and create module-level mouth / brain clients."""
    global mouth, brain, _token_estimator

    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    mouth = _make_client(_parse_section(raw["mouth"]))
    brain = _make_client(_parse_section(raw["brain"]))
    _token_estimator = raw.get("token_estimator", "qwen")

    print(f"  [LLM] mouth: {mouth.cfg.provider} / {mouth.model}")
    print(f"  [LLM] brain: {brain.cfg.provider} / {brain.model}")
