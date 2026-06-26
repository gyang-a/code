from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from code_agent.services.summarizer import truncate
from code_agent.ui.console import console


NODE_LABELS = {
    "load_project_context": "loaded project context",
    "route_input": "routed input",
    "command_handler": "handled slash command",
    "plan_node": "created task plan",
    "agent_loop": "agent thinking",
    "execute": "executed tool call",
    "tool_result_router": "classified tool result",
    "approval": "waiting for approval",
    "reject": "handled rejected action",
    "observe": "observed result",
    "validation_node": "ran validation",
    "review_diff_node": "reviewed diff",
    "final_summary": "prepared final summary",
}


def render_stream_chunk(chunk: Mapping[str, Any]) -> None:
    if "__interrupt__" in chunk:
        return

    for node_name, update in chunk.items():
        label = NODE_LABELS.get(node_name, node_name)
        console.print(f"[dim]> {label}[/dim]")
        if isinstance(update, Mapping):
            _render_node_update(node_name, update)


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
    return value if isinstance(value, dict) else {"reason": str(value)}


def _render_node_update(node_name: str, update: Mapping[str, Any]) -> None:
    if node_name == "plan_node" and update.get("plan"):
        for index, item in enumerate(update["plan"], start=1):
            console.print(f"[dim]  {index}. {item}[/dim]")

    if update.get("messages"):
        _render_messages(update["messages"])

    if update.get("approval_reason"):
        console.print(f"[yellow]  approval required:[/yellow] {update['approval_reason']}")

    if update.get("rejected_reason"):
        console.print(f"[red]  rejected:[/red] {update['rejected_reason']}")

    if update.get("test_command"):
        console.print(f"[dim]  validation command: {update['test_command']}[/dim]")

    if update.get("test_result"):
        first_line = str(update["test_result"]).splitlines()[0] if str(update["test_result"]).strip() else "validation complete"
        console.print(f"[dim]  {first_line}[/dim]")

    if update.get("diff_summary"):
        console.print(f"[dim]  {update['diff_summary']}[/dim]")


def _render_messages(messages: list[BaseMessage]) -> None:
    for message in messages:
        if isinstance(message, AIMessage):
            tool_calls = getattr(message, "tool_calls", None) or []
            for tool_call in tool_calls:
                name = tool_call.get("name", "tool")
                args = tool_call.get("args") or {}
                console.print(f"[cyan]  tool call:[/cyan] {name}({_format_args(args)})")
            if message.content and not tool_calls:
                console.print(f"[dim]  agent drafted response[/dim]")
        elif isinstance(message, ToolMessage):
            first_line = str(message.content).splitlines()[0] if str(message.content).strip() else "empty tool result"
            style = "yellow" if "APPROVAL_REQUIRED" in first_line else "red" if "REJECTED" in first_line else "dim"
            console.print(f"[{style}]  tool result:[/{style}] {truncate(first_line, 180)}")


def _format_args(args: Mapping[str, Any]) -> str:
    parts = []
    for key, value in args.items():
        rendered = repr(value)
        if len(rendered) > 80:
            rendered = rendered[:77] + "..."
        parts.append(f"{key}={rendered}")
    return ", ".join(parts)
