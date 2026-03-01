"""
Tool definitions and execution for 小悠's web browsing capabilities.

Tools are defined in Ollama's native format and executed when the model
requests them via tool_calls in the streaming response.
"""

import asyncio
import json
import logging
from datetime import datetime

from ddgs import DDGS

import config

logger = logging.getLogger(__name__)

# ── Tool definitions (Ollama native format) ─────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_datetime",
            "description": "获取当前的日期和时间，包括星期几",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "搜索网络获取实时信息，如天气、新闻、本地商家、价格等",
            "parameters": {
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索查询词",
                    },
                },
            },
        },
    },
]

# ── Tool implementations ────────────────────────────────────────────────────

_WEEKDAYS = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]


def get_current_datetime() -> str:
    """Return current date, time, and weekday in Chinese."""
    now = datetime.now()
    weekday = _WEEKDAYS[now.weekday()]
    hour = now.hour
    minute = now.minute
    if hour < 6:
        period = "凌晨"
    elif hour < 12:
        period = "上午"
    elif hour == 12:
        period = "中午"
    elif hour < 18:
        period = "下午"
    else:
        period = "晚上"
    display_hour = hour if hour <= 12 else hour - 12
    return (
        f"{now.year}年{now.month}月{now.day}日 {weekday} "
        f"{period}{display_hour}点{minute:02d}分"
    )


async def web_search(query: str) -> str:
    """Search DuckDuckGo and return formatted results."""
    max_results = config.TOOLS_SEARCH_MAX_RESULTS
    loop = asyncio.get_running_loop()

    def _search() -> str:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            if not results:
                return f"没有找到关于'{query}'的搜索结果。"
            lines = []
            for r in results:
                title = r.get("title", "")
                body = r.get("body", "")
                lines.append(f"- {title}: {body}")
            return "\n".join(lines)
        except Exception as e:
            logger.warning(f"web_search failed: {e}")
            return f"搜索'{query}'时出错，请稍后再试。"

    return await loop.run_in_executor(None, _search)


# ── Dispatcher ──────────────────────────────────────────────────────────────

async def execute_tool(name: str, arguments: dict) -> str:
    """Execute a tool by name and return its result as a string."""
    if name == "get_current_datetime":
        return get_current_datetime()
    elif name == "web_search":
        query = arguments.get("query", "")
        if not query:
            return "搜索查询不能为空。"
        return await asyncio.wait_for(
            web_search(query),
            timeout=config.TOOLS_TIMEOUT,
        )
    else:
        return f"未知工具: {name}"
