from __future__ import annotations


def truncate(text: str, limit: int = 12_000) -> str:
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return text[:limit] + f"\n... truncated {omitted} characters ..."
