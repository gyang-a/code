from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from rich.markup import escape

from code_agent.services.summarizer import truncate
from code_agent.ui.approval import format_tool_call_summary
from code_agent.ui.console import console


NODE_LABELS = {
    "agent": "Agent is thinking",
    "model": "Agent is thinking",
    "middleware": "Updated agent context",
    "execute": "Handled tool calls",
    "tools": "Handled tool calls",
    "tool_result_router": "Processed tool results",
}


def render_stream_chunk(chunk: Mapping[str, Any]) -> None:
    if "__interrupt__" in chunk:
        return

    for node_name, update in chunk.items():
        if not _should_render_update(node_name, update):
            continue
        label = NODE_LABELS.get(node_name, node_name)
        console.print(f"[dim]> {label}[/dim]")
        if isinstance(update, Mapping):
            _render_node_update(update)


def final_answer_from_chunk(chunk: Mapping[str, Any]) -> str | None:
    for update in chunk.values():
        if isinstance(update, Mapping) and update.get("final_answer"):
            return str(update["final_answer"])
    return None


def interrupt_from_chunk(chunk: Mapping[str, Any]) -> dict[str, Any] | None:
    interrupts = chunk.get("__interrupt__")
    if not interrupts:
        return None
    value = interrupts[0].value
    if isinstance(value, Mapping):
        return dict(value)
    action_requests = getattr(value, "action_requests", None)
    review_configs = getattr(value, "review_configs", None)
    if action_requests is not None or review_configs is not None:
        return {
            "action_requests": [_as_dict(item) for item in action_requests or []],
            "review_configs": [_as_dict(item) for item in review_configs or []],
        }
    return {"reason": str(value)}


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    data: dict[str, Any] = {}
    for key in ("name", "args", "description", "action_name", "allowed_decisions"):
        if hasattr(value, key):
            data[key] = getattr(value, key)
    return data


def _render_node_update(update: Mapping[str, Any]) -> None:
    if update.get("messages"):
        _render_messages(update["messages"])

    if update.get("test_command"):
        summary = format_tool_call_summary("run_shell", {"command": update["test_command"]}, max_length=100)
        console.print(f"[dim]  validation command: {escape(summary)}[/dim]")

    if update.get("test_result"):
        first_line = str(update["test_result"]).splitlines()[0] if str(update["test_result"]).strip() else "validation complete"
        console.print(f"[dim]  {escape(first_line)}[/dim]")


def _should_render_update(node_name: str, update: Any) -> bool:
    if not isinstance(update, Mapping):
        return True
    if not update:
        return False
    renderable_keys = {"messages", "final_answer", "test_command", "test_result"}
    if any(update.get(key) for key in renderable_keys):
        return True
    if ".before_" in node_name or ".after_" in node_name:
        return False
    return False


def _render_messages(messages: list[BaseMessage]) -> None:
    for message in messages:
        if isinstance(message, AIMessage):
            tool_calls = getattr(message, "tool_calls", None) or []
            for tool_call in tool_calls:
                name = tool_call.get("name", "tool")
                args = tool_call.get("args") or {}
                summary = format_tool_call_summary(name, args, max_length=100)
                console.print(f"[cyan]  tool call:[/cyan] {escape(summary)}")
            if message.content and not tool_calls:
                console.print("[dim]  Agent drafted a response[/dim]")
        elif isinstance(message, ToolMessage):
            first_line = str(message.content).splitlines()[0] if str(message.content).strip() else "empty tool result"
            style = "red" if first_line.startswith("REJECTED[") else "dim"
            console.print(f"[{style}]  tool result:[/{style}] {escape(_tool_result_summary(first_line))}")


def _tool_result_summary(first_line: str) -> str:
    return truncate(first_line, 180)
