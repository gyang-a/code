from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

    workspace: str
    user_goal: str
    context_summary: str | None
    recent_files: list[str]
    compaction_count: int

    iteration_count: int
    max_iterations: int

    changed_files: list[str]
    did_write: bool

    test_command: str | None
    test_result: str | None

    tool_errors: list[str]
    final_answer: str | None
