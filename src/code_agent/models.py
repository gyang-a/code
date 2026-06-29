from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class RiskLevel(str, Enum):
    level_0 = "level_0"
    level_1 = "level_1"
    level_2 = "level_2"
    level_3 = "level_3"


class PermissionDecision(BaseModel):
    risk: RiskLevel
    allowed: bool
    requires_approval: bool = False
    reason: str
