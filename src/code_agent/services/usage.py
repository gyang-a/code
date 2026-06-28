from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    actual: bool = False

    def add(self, other: "TokenUsage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_tokens += other.total_tokens
        self.actual = self.actual or other.actual


def estimate_message_tokens(messages: list[Any]) -> int:
    chars = 0
    for message in messages:
        chars += len(message.type) + len(_stringify_content(getattr(message, "content", "")))
        tool_calls = getattr(message, "tool_calls", None) or []
        chars += len(str(tool_calls))
    return max(1, chars // 4) if chars else 0


def usage_from_messages(messages: list[Any]) -> TokenUsage:
    usage = TokenUsage()
    for message in messages:
        if getattr(message, "type", None) != "ai":
            continue
        usage.add(usage_from_message(message))
    return usage


def usage_from_message(message: Any) -> TokenUsage:
    actual = _usage_mapping(getattr(message, "usage_metadata", None))
    if actual:
        return actual

    response_metadata = getattr(message, "response_metadata", None)
    if isinstance(response_metadata, dict):
        for key in ("token_usage", "usage", "usage_metadata"):
            actual = _usage_mapping(response_metadata.get(key))
            if actual:
                return actual

    return TokenUsage(total_tokens=_estimate_single_message_tokens(message), actual=False)


def format_usage_line(
    *,
    context_tokens: int,
    model_usage: TokenUsage,
    compression_count: int,
) -> str:
    marker = "" if model_usage.actual else "~"
    return (
        f"usage: context~{context_tokens} tokens | "
        f"model {marker}{model_usage.total_tokens} tokens "
        f"(in {marker}{model_usage.input_tokens}, out {marker}{model_usage.output_tokens}) | "
        f"context compressions: {compression_count}"
    )


def _usage_mapping(value: Any) -> TokenUsage | None:
    if not isinstance(value, dict):
        return None

    input_tokens = _int_value(value, "input_tokens", "prompt_tokens")
    output_tokens = _int_value(value, "output_tokens", "completion_tokens")
    total_tokens = _int_value(value, "total_tokens")

    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens

    if input_tokens == 0 and output_tokens == 0 and total_tokens == 0:
        return None

    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        actual=True,
    )


def _int_value(value: dict[str, Any], *keys: str) -> int:
    for key in keys:
        raw = value.get(key)
        if isinstance(raw, int):
            return raw
        if isinstance(raw, float):
            return int(raw)
    return 0


def _estimate_single_message_tokens(message: Any) -> int:
    chars = len(message.type) + len(_stringify_content(getattr(message, "content", "")))
    tool_calls = getattr(message, "tool_calls", None) or []
    chars += len(str(tool_calls))
    return max(1, chars // 4) if chars else 0


def _stringify_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    return str(content)
