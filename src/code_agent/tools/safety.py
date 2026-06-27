from __future__ import annotations

import subprocess
from pathlib import Path

from code_agent.models import PermissionDecision, RiskLevel
from code_agent.services.permissions import (
    LEVEL_2_COMMAND_PREFIXES,
    LEVEL_2_FILES,
    LEVEL_3_COMMAND_MARKERS,
    allowed,
    classify_command,
    classify_file_operation,
    classify_tool_call,
    describe_permission_policy,
    rejected,
)
from code_agent.services.sandbox import ShellSandbox
from code_agent.services.workspace import Workspace


def run_argv(argv: list[str], cwd: Path, *, timeout: int) -> subprocess.CompletedProcess[str]:
    """Backward-compatible shim for older tests/call sites.

    New code should use ShellSandbox directly so runtime isolation is not mixed
    into permission rules.
    """
    workspace = Workspace(cwd)
    return ShellSandbox(workspace).run(argv, timeout=timeout)


__all__ = [
    "LEVEL_2_COMMAND_PREFIXES",
    "LEVEL_2_FILES",
    "LEVEL_3_COMMAND_MARKERS",
    "PermissionDecision",
    "RiskLevel",
    "allowed",
    "classify_command",
    "classify_file_operation",
    "classify_tool_call",
    "describe_permission_policy",
    "rejected",
    "run_argv",
]
