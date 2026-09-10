from __future__ import annotations

from contextvars import ContextVar

raw_tool_output: ContextVar[bool] = ContextVar('raw_tool_output', default=False)


def truncate(text: str, limit: int = 12_000) -> str:
    if raw_tool_output.get() or len(text) <= limit:
        return text
    omitted = len(text) - limit
    return text[:limit] + f"\n... truncated {omitted} characters ..."
