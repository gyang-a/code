from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    level_0 = "level_0"
    level_1 = "level_1"
    level_2 = "level_2"
    level_3 = "level_3"


class CommandSpec(BaseModel):
    name: str = Field(description="Stable command name exposed to the model.")
    argv: list[str] = Field(description="Command argv executed with shell disabled.")
    risk: RiskLevel = RiskLevel.level_1
    description: str = ""


class PermissionDecision(BaseModel):
    risk: RiskLevel
    allowed: bool
    requires_approval: bool = False
    reason: str


class ToolFailure(BaseModel):
    error: str
    hint: str | None = None
