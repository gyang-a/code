from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from code_agent.services.summarizer import truncate
from code_agent.ui.approval import format_approval_summary
from code_agent.ui.console import console


NODE_LABELS = {
    "agent": "Agent 正在思考",
    "context_manager": "已压缩历史上下文",
    "execute": "已执行工具调用",
    "tool_result_router": "已分类工具结果",
    "approval": "等待审批",
    "reject": "已处理拒绝操作",
}


def render_stream_chunk(chunk: Mapping[str, Any]) -> None:
    if "__interrupt__" in chunk:
        return

    for node_name, update in chunk.items():
        if node_name == "context_manager" and not _context_update_compacted(update):
            continue
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
    if update.get("messages"):
        _render_messages(update["messages"])

    if update.get("approval_reason"):
        console.print(
            f"[yellow]  需要审批:[/yellow] "
            f"{format_approval_summary(update.get('pending_approval'), str(update['approval_reason']))}"
        )

    if update.get("rejected_reason"):
        console.print(f"[red]  已拒绝:[/red] {update['rejected_reason']}")

    if update.get("test_command"):
        console.print(f"[dim]  验证命令: {update['test_command']}[/dim]")

    if update.get("test_result"):
        first_line = str(update["test_result"]).splitlines()[0] if str(update["test_result"]).strip() else "验证完成"
        console.print(f"[dim]  {first_line}[/dim]")

    if update.get("compaction_count"):
        console.print(f"[dim]  上下文压缩次数: {update['compaction_count']}[/dim]")


def _context_update_compacted(update: Any) -> bool:
    return isinstance(update, Mapping) and bool(update.get("compaction_count"))


def _render_messages(messages: list[BaseMessage]) -> None:
    for message in messages:
        if isinstance(message, AIMessage):
            tool_calls = getattr(message, "tool_calls", None) or []
            for tool_call in tool_calls:
                name = tool_call.get("name", "tool")
                args = tool_call.get("args") or {}
                console.print(f"[cyan]  工具调用:[/cyan] {name}({_format_args(args)})")
            if message.content and not tool_calls:
                console.print(f"[dim]  Agent 已生成回复草稿[/dim]")
        elif isinstance(message, ToolMessage):
            first_line = str(message.content).splitlines()[0] if str(message.content).strip() else "空工具结果"
            style = "yellow" if "APPROVAL_REQUIRED" in first_line else "red" if "REJECTED" in first_line else "dim"
            console.print(f"[{style}]  工具结果:[/{style}] {truncate(first_line, 180)}")


def _format_args(args: Mapping[str, Any]) -> str:
    parts = []
    for key, value in args.items():
        rendered = repr(value)
        if len(rendered) > 80:
            rendered = rendered[:77] + "..."
        parts.append(f"{key}={rendered}")
    return ", ".join(parts)
