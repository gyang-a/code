from __future__ import annotations

from typing import Annotated, NotRequired, Required, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    messages: Required[Annotated[list[BaseMessage], add_messages]]

    workspace: NotRequired[str]
    thread_id: NotRequired[str]
    user_goal: NotRequired[str]

    changed_files: NotRequired[list[str]]
    did_write: NotRequired[bool]

    tool_errors: NotRequired[list[str]]
    final_answer: NotRequired[str | None]

    skill_reviewed_tool_count: NotRequired[int]
    skill_review_status: NotRequired[str]
    skill_review_message: NotRequired[str]
    skill_review_pending_id: NotRequired[str | None]
    skill_review_trigger: NotRequired[str | None]
