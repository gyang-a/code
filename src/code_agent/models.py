from __future__ import annotations

from enum import Enum
from typing import Any, Literal, TypedDict

from pydantic import BaseModel


class RiskLevel(str, Enum):
    level_0 = "level_0"
    level_1 = "level_1"
    level_2 = "level_2"
    level_3 = "level_3"


class SandboxMode(str, Enum):
    read_only = "read-only"
    workspace_write = "workspace-write"
    danger_full_access = "danger-full-access"


class PermissionDecision(BaseModel):
    risk: RiskLevel
    allowed: bool
    requires_approval: bool = False
    reason: str


class AgentError(TypedDict):
    source: str
    category: Literal[
        "validation",
        "permission",
        "sandbox",
        "timeout",
        "network",
        "tool_execution",
        "model",
    ]
    code: str
    message: str
    retryable: bool
    attempt: int
    tool_call_id: str | None
    details: dict[str, Any]
