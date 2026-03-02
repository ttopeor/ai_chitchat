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
import i18n

logger = logging.getLogger(__name__)

# ── Tool definitions (Ollama native format) ─────────────────────────────────

def get_tool_definitions() -> list[dict]:
    """Return tool definitions with localized descriptions."""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_current_datetime",
                "description": i18n.T.TOOL_DATETIME_DESC,
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
                "description": i18n.T.TOOL_SEARCH_DESC,
                "parameters": {
                    "type": "object",
                    "required": ["query"],
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": i18n.T.TOOL_SEARCH_QUERY_DESC,
                        },
                    },
                },
            },
        },
    ]

# ── Tool implementations ────────────────────────────────────────────────────

def get_current_datetime() -> str:
    """Return current date, time, and weekday in the active locale."""
    return i18n.T.format_full_datetime(datetime.now())


async def web_search(query: str) -> str:
    """Search DuckDuckGo and return formatted results."""
    max_results = config.TOOLS_SEARCH_MAX_RESULTS
    loop = asyncio.get_running_loop()

    def _search() -> str:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            if not results:
                return i18n.T.TOOL_NO_RESULTS.format(query=query)
            lines = []
            for r in results:
                title = r.get("title", "")
                body = r.get("body", "")
                lines.append(f"- {title}: {body}")
            return "\n".join(lines)
        except Exception as e:
            logger.warning(f"web_search failed: {e}")
            return i18n.T.TOOL_SEARCH_ERROR.format(query=query)

    return await loop.run_in_executor(None, _search)


# ── Dispatcher ──────────────────────────────────────────────────────────────

async def execute_tool(name: str, arguments: dict) -> str:
    """Execute a tool by name and return its result as a string."""
    if name == "get_current_datetime":
        return get_current_datetime()
    elif name == "web_search":
        query = arguments.get("query", "")
        if not query:
            return i18n.T.TOOL_EMPTY_QUERY
        return await asyncio.wait_for(
            web_search(query),
            timeout=config.TOOLS_TIMEOUT,
        )
    else:
        return i18n.T.TOOL_UNKNOWN.format(name=name)
