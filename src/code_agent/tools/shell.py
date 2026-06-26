from __future__ import annotations

import subprocess

from langchain_core.tools import tool

from code_agent.services.summarizer import truncate
from code_agent.services.workspace import Workspace
from code_agent.tools.safety import approval_required, classify_command, rejected, run_argv


def build_run_command_tool(workspace: Workspace):
    @tool
    def run_command(command: str, timeout_seconds: int = 60) -> str:
        """Run an approved validation command. Level 2 commands require confirmation; Level 3 commands are rejected."""
        decision, argv = classify_command(command, workspace)
        if decision.risk.value == "level_3":
            return rejected(decision.risk, decision.reason)
        if decision.requires_approval or argv is None:
            return approval_required(decision.risk, decision.reason)
        timeout_seconds = min(max(timeout_seconds, 1), 180)
        try:
            result = run_argv(argv, workspace.root, timeout=timeout_seconds)
        except FileNotFoundError as exc:
            return f"ERROR: {exc}"
        except subprocess.TimeoutExpired:
            return f"ERROR: Command timed out after {timeout_seconds}s"

        output = result.stdout + result.stderr
        return truncate(
            f"ALLOWED[{decision.risk.value}]: {decision.reason}\n"
            f"$ {' '.join(argv)}\n"
            f"exit_code={result.returncode}\n"
            f"{output}"
        )

    return run_command
