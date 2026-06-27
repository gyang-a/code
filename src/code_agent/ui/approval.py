from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from code_agent.services.summarizer import truncate


def format_approval_summary(
    action: Mapping[str, Any] | None,
    reason: str | None = None,
    *,
    max_length: int = 140,
) -> str:
    tool_name = str((action or {}).get("tool") or "unknown")
    args = dict((action or {}).get("args") or {})
    detail = _action_detail(tool_name, args) or _clean_reason(reason) or "requires approval"
    return truncate(f"{tool_name}: {detail}", max_length)


def _action_detail(tool_name: str, args: Mapping[str, Any]) -> str | None:
    if tool_name in {"run_shell", "run_command"}:
        return _single_line(args.get("command"))

    path = args.get("path")
    if path:
        return _single_line(path)

    if args:
        return _single_line(", ".join(sorted(str(key) for key in args)))

    return None


def _clean_reason(reason: str | None) -> str | None:
    if not reason:
        return None
    cleaned = str(reason)
    if ":" in cleaned and cleaned.startswith("APPROVAL_REQUIRED"):
        cleaned = cleaned.split(":", 1)[1]
    return _single_line(cleaned)


def _single_line(value: Any) -> str:
    return " ".join(str(value or "").split())
