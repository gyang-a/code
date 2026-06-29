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
