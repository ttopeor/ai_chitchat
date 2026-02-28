"""
MemoryManager — persistent long-term memory for 小悠.

Memories are stored as JSONL (one JSON object per line).  Extraction is done
by the 72b brain model; retrieval uses keyword + importance scoring to avoid
recency bias.

Uses Ollama native API (/api/chat) for extraction to ensure num_ctx is passed.
"""

import json
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta

import httpx

import config

# Ollama native API base
_OLLAMA_BASE = config.LLM_BASE_URL.removesuffix("/v1").removesuffix("/")


@dataclass
class MemoryEntry:
    id: str
    timestamp: str
    category: str          # fact / preference / event / emotion / routine
    content: str
    keywords: list[str]
    importance: int        # 1-5
    source: str            # e.g. "2月27日聊天"


class MemoryManager:
    def __init__(
        self,
        model: str,
        storage_dir: str = "memories",
        storage_file: str = "memories.jsonl",
        max_context: int = 8,
        min_turns: int = 2,
    ):
        self._model = model
        self._max_context = max_context
        self._min_turns = min_turns

        self._path = os.path.join(storage_dir, storage_file)
        os.makedirs(storage_dir, exist_ok=True)

        self._memories: list[MemoryEntry] = []
        self._load()

    # ── persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not os.path.exists(self._path):
            return
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        self._memories.append(MemoryEntry(**json.loads(line)))
                    except (json.JSONDecodeError, TypeError):
                        continue

    def _append(self, entries: list[MemoryEntry]) -> None:
        with open(self._path, "a", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(asdict(e), ensure_ascii=False) + "\n")

    def _save_all(self) -> None:
        with open(self._path, "w", encoding="utf-8") as f:
            for e in self._memories:
                f.write(json.dumps(asdict(e), ensure_ascii=False) + "\n")

    # ── extraction (called by brain after conversation ends) ──────────────────

    async def extract_memories(self, conversation: list[dict]) -> list[MemoryEntry]:
        user_msgs = [m for m in conversation if m["role"] == "user"]
        if len(user_msgs) < self._min_turns:
            return []

        conv_text = "\n".join(
            f"{'小悠' if m['role'] == 'assistant' else '对方'}: "
            + (m["content"] if isinstance(m["content"], str) else str(m["content"]))
            for m in conversation
            if m["role"] in ("user", "assistant")
        )

        prompt = (
            "/no_think\n"
            "你是一个记忆提取助手。分析以下对话，提取值得长期记住的信息。\n\n"
            "提取标准:\n"
            "- 对方的个人信息、喜好、习惯\n"
            "- 重要的事件、计划、约定\n"
            "- 情绪状态的重大变化\n"
            "- 对方反复提到的话题\n"
            "- 不要记录闲聊废话、打招呼、日常寒暄\n"
            "- 不要记录小悠自己说的信息，只记对方的\n\n"
            f"对话内容:\n{conv_text}\n\n"
            "用JSON数组格式回复，每条记忆包含:\n"
            '{"category": "fact/preference/event/emotion/routine", '
            '"content": "记忆内容", '
            '"keywords": ["关键词1", "关键词2"], '
            '"importance": 1到5的数字}\n\n'
            "如果没有值得记住的内容，回复空数组 []\n"
            "只回复JSON，不要其他文字。"
        )

        try:
            result = await self._call_ollama(prompt)
            if result is None:
                return []

            # Strip markdown code fences if present
            if result.startswith("```"):
                result = result.split("\n", 1)[1].rsplit("```", 1)[0]

            entries_data = json.loads(result)
            if not isinstance(entries_data, list):
                return []

            now_str = datetime.now().isoformat()
            source = f"对话于{datetime.now().strftime('%m月%d日%H:%M')}"

            new_entries: list[MemoryEntry] = []
            for d in entries_data:
                if not isinstance(d, dict) or "content" not in d:
                    continue
                new_entries.append(MemoryEntry(
                    id=uuid.uuid4().hex[:8],
                    timestamp=now_str,
                    category=d.get("category", "fact"),
                    content=d["content"],
                    keywords=d.get("keywords", []),
                    importance=min(5, max(1, int(d.get("importance", 3)))),
                    source=source,
                ))

            new_entries = self._deduplicate(new_entries)
            if new_entries:
                self._append(new_entries)
                self._memories.extend(new_entries)

            return new_entries

        except Exception as e:
            print(f"[Memory] extraction error: {e}")
            return []

    # ── Ollama native API call ────────────────────────────────────────────────

    async def _call_ollama(self, prompt: str) -> str | None:
        """Call the brain model via native Ollama /api/chat with num_ctx."""
        body = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"num_ctx": config.MODEL_NUM_CTX},
            "stream": False,
            "think": False,
        }

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{_OLLAMA_BASE}/api/chat",
                    json=body,
                    timeout=httpx.Timeout(connect=10, read=120, write=10, pool=10),
                )
                resp.raise_for_status()
                data = resp.json()
                return data.get("message", {}).get("content", "").strip()
        except Exception as e:
            print(f"[Memory] LLM error: {e}")
            return None

    # ── retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str = "",
        extra_keywords: list[str] | None = None,
    ) -> list[MemoryEntry]:
        if not self._memories:
            return []

        query_terms = set(self._extract_terms(query))
        if extra_keywords:
            query_terms.update(extra_keywords)

        scored: list[tuple[float, MemoryEntry]] = []
        now = datetime.now()

        for mem in self._memories:
            score = 0.0

            # keyword overlap
            mem_terms = set(mem.keywords) | set(self._extract_terms(mem.content))
            overlap = len(query_terms & mem_terms)
            score += overlap * 2.0

            # importance (dominant factor — avoids recency bias)
            score += mem.importance * 1.5

            # slight recency (max 1 point)
            try:
                age_days = (now - datetime.fromisoformat(mem.timestamp)).days
            except ValueError:
                age_days = 0
            score += max(0.0, 1.0 - age_days / 365)

            if score > 0:
                scored.append((score, mem))

        scored.sort(key=lambda x: x[0], reverse=True)

        if not scored:
            # fallback: highest-importance memories
            by_imp = sorted(self._memories, key=lambda m: m.importance, reverse=True)
            return by_imp[: min(3, len(by_imp))]

        return [m for _, m in scored[: self._max_context]]

    # ── deduplication ─────────────────────────────────────────────────────────

    def _deduplicate(self, new_entries: list[MemoryEntry]) -> list[MemoryEntry]:
        unique: list[MemoryEntry] = []
        for new in new_entries:
            is_dup = False
            for existing in self._memories:
                if existing.category != new.category:
                    continue
                e_kw = set(existing.keywords)
                n_kw = set(new.keywords)
                if e_kw and n_kw:
                    overlap = len(e_kw & n_kw) / min(len(e_kw), len(n_kw))
                    if overlap > 0.6:
                        if new.importance > existing.importance:
                            existing.content = new.content
                            existing.importance = new.importance
                            existing.timestamp = new.timestamp
                            self._save_all()
                        is_dup = True
                        break
            if not is_dup:
                unique.append(new)
        return unique

    # ── consolidation (run on startup) ────────────────────────────────────────

    def consolidate(self) -> int:
        if len(self._memories) < 10:
            return 0
        cutoff = datetime.now() - timedelta(days=90)
        before = len(self._memories)
        self._memories = [
            m for m in self._memories
            if m.importance >= 3
            or self._parse_ts(m.timestamp) > cutoff
        ]
        removed = before - len(self._memories)
        if removed > 0:
            self._save_all()
        return removed

    # ── profile memories (for dynamic background synthesis) ───────────────────

    def get_profile_memories(self) -> list[MemoryEntry]:
        """Return high-importance memories sorted by importance desc.

        Used by BrainEngine to synthesize a dynamic user background
        at startup and after new memories are extracted.
        """
        min_imp = config.PROFILE_MIN_IMPORTANCE
        profile = [m for m in self._memories if m.importance >= min_imp]
        profile.sort(key=lambda m: m.importance, reverse=True)
        return profile

    # ── context injection ─────────────────────────────────────────────────────

    def format_for_prompt(self, query: str = "") -> str:
        memories = self.retrieve(query)
        if not memories:
            return ""
        lines = [
            f"- [{m.category}] {m.content}（{m.source}）"
            for m in memories
        ]
        return "\n".join(lines)

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_terms(text: str) -> list[str]:
        # Simple: 2-char and 3-char sliding windows for Chinese, plus whole words
        terms: list[str] = []
        for i in range(len(text) - 1):
            terms.append(text[i: i + 2])
        for i in range(len(text) - 2):
            terms.append(text[i: i + 3])
        return terms

    @staticmethod
    def _parse_ts(ts: str) -> datetime:
        try:
            return datetime.fromisoformat(ts)
        except ValueError:
            return datetime.now()

    @property
    def count(self) -> int:
        return len(self._memories)
