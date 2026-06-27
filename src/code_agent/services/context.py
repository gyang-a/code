from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.messages import BaseMessage, RemoveMessage

from code_agent.services.summarizer import truncate


@dataclass(frozen=True)
class CompactionResult:
    messages: list[BaseMessage | RemoveMessage]
    context_summary: str
    recent_files: list[str]
    compacted: bool
    removed_count: int


def should_compact_messages(
    messages: list[BaseMessage],
    *,
    max_messages: int,
    max_chars: int,
) -> bool:
    if len(messages) > max_messages:
        return True
    return _messages_char_count(messages) > max_chars


def compact_messages(
    messages: list[BaseMessage],
    *,
    existing_summary: str | None,
    changed_files: list[str],
    test_result: str | None,
    keep_recent: int,
    max_summary_chars: int = 8000,
) -> CompactionResult:
    if len(messages) <= keep_recent:
        return CompactionResult(
            messages=[],
            context_summary=existing_summary or "",
            recent_files=_extract_recent_files(messages),
            compacted=False,
            removed_count=0,
        )

    removable, retained = _split_messages(messages, keep_recent=keep_recent)
    if not removable:
        return CompactionResult(
            messages=[],
            context_summary=existing_summary or "",
            recent_files=_extract_recent_files(retained),
            compacted=False,
            removed_count=0,
        )

    summary = build_context_summary(
        removable,
        existing_summary=existing_summary,
        changed_files=changed_files,
        test_result=test_result,
        max_chars=max_summary_chars,
    )
    removals = [RemoveMessage(id=message.id) for message in removable if message.id]
    return CompactionResult(
        messages=removals,
        context_summary=summary,
        recent_files=_extract_recent_files(retained),
        compacted=True,
        removed_count=len(removals),
    )


def build_context_summary(
    messages: list[BaseMessage],
    *,
    existing_summary: str | None,
    changed_files: list[str],
    test_result: str | None,
    max_chars: int,
) -> str:
    sections = []
    if existing_summary:
        sections.append("既有摘要:\n" + truncate(existing_summary, max_chars // 3))

    if changed_files:
        sections.append("本轮变更文件:\n" + "\n".join(f"- {path}" for path in changed_files))

    if test_result:
        sections.append("最近验证结果:\n" + truncate(test_result, 1200))

    ledger = []
    for message in messages:
        ledger.append(_summarize_message(message))
    if ledger:
        sections.append("已压缩的历史消息:\n" + "\n".join(ledger))

    return truncate("\n\n".join(sections), max_chars)


def _split_messages(messages: list[BaseMessage], *, keep_recent: int) -> tuple[list[BaseMessage], list[BaseMessage]]:
    retained_recent = messages[-keep_recent:]
    retained_ids = {id(message) for message in retained_recent}
    removable = [
        message
        for message in messages
        if id(message) not in retained_ids and _can_remove_message(message)
    ]
    retained = [
        message
        for message in messages
        if not _can_remove_message(message) or id(message) in retained_ids
    ]
    return removable, retained


def _can_remove_message(message: BaseMessage) -> bool:
    if message.type != "system":
        return True
    return bool(message.additional_kwargs.get("code_agent_context_summary"))


def _summarize_message(message: BaseMessage) -> str:
    role = message.type
    content = _content_to_text(message.content)
    tool_calls = getattr(message, "tool_calls", None) or []
    if tool_calls:
        calls = ", ".join(_format_tool_call(call) for call in tool_calls)
        return f"- {role}: tool_calls=[{calls}]"
    return f"- {role}: {truncate(content.replace(chr(10), ' '), 500)}"


def _format_tool_call(tool_call: dict[str, Any]) -> str:
    name = tool_call.get("name", "tool")
    args = tool_call.get("args") or {}
    interesting = {}
    for key in ("path", "command", "query"):
        if key in args:
            interesting[key] = args[key]
    return f"{name}({interesting or args})"


def _extract_recent_files(messages: list[BaseMessage]) -> list[str]:
    files: list[str] = []
    for message in messages:
        tool_calls = getattr(message, "tool_calls", None) or []
        for tool_call in tool_calls:
            args = tool_call.get("args") or {}
            path = args.get("path")
            if isinstance(path, str) and path not in files:
                files.append(path)
    return files[-20:]


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return str(content)


def _messages_char_count(messages: list[BaseMessage]) -> int:
    total = 0
    for message in messages:
        total += len(_content_to_text(message.content))
        for tool_call in getattr(message, "tool_calls", None) or []:
            total += len(str(tool_call))
    return total
