from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

    workspace: str
    user_goal: str
    input_kind: str | None
    project_context: str | None
    context_summary: str | None
    recent_files: list[str]
    compaction_count: int

    plan: list[str]
    current_step: str | None
    iteration_count: int
    max_iterations: int

    changed_files: list[str]
    did_write: bool
    last_diff: str | None

    test_command: str | None
    test_result: str | None
    validation_requested: bool

    needs_approval: bool
    approval_reason: str | None
    pending_approval: dict | None
    rejected_reason: str | None

    tool_errors: list[str]
    diff_summary: str | None
    final_answer: str | None
