from __future__ import annotations

from collections.abc import Mapping
from typing import Any

def format_approval_summary(
    action: Mapping[str, Any] | None,
    reason: str | None = None,
    *,
    max_length: int = 140,
) -> str:
    tool_name = str((action or {}).get("tool") or (action or {}).get("name") or "unknown")
    args = dict((action or {}).get("args") or {})
    detail = _action_detail(tool_name, args) or _clean_reason(reason) or "requires approval"
    return _truncate_single_line(f"{tool_name}: {detail}", max_length)


def format_tool_call_summary(
    tool_name: str,
    args: Mapping[str, Any] | None,
    *,
    max_length: int = 140,
) -> str:
    detail = _action_detail(tool_name, dict(args or {}))
    if detail:
        return _truncate_single_line(f"{tool_name}: {detail}", max_length)
    return _truncate_single_line(tool_name, max_length)


def _action_detail(tool_name: str, args: Mapping[str, Any]) -> str | None:
    path = args.get("path")
    if path:
        return _single_line(path)

    if args:
        return _single_line(", ".join(sorted(str(key) for key in args)))

    return None


def _clean_reason(reason: str | None) -> str | None:
    if not reason:
        return None
    return _single_line(reason)


def _single_line(value: Any) -> str:
    return " ".join(str(value or "").split())


def _truncate_single_line(value: str, max_length: int) -> str:
    line = _single_line(value)
    if len(line) <= max_length:
        return line
    if max_length <= 3:
        return "." * max_length
    return line[: max_length - 3] + "..."
