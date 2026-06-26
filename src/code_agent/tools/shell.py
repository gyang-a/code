from __future__ import annotations

import subprocess

from langchain_core.tools import tool

from code_agent.services.summarizer import truncate
from code_agent.services.workspace import Workspace
from code_agent.services.sandbox import ShellSandbox
from code_agent.tools.safety import approval_required, classify_command, rejected

APPROVAL_TOKEN = "approved"


def build_run_command_tool(workspace: Workspace):
    sandbox = ShellSandbox(workspace)

    @tool
    def run_shell(command: str, timeout_seconds: int = 60, approval_token: str | None = None) -> str:
        """在工作区沙箱内运行受限 shell 命令；Level 2 命令需要确认，Level 3 命令会被拒绝。"""
        decision, argv = classify_command(command, workspace)
        if decision.risk.value == "level_3":
            return rejected(decision.risk, decision.reason)
        if (decision.requires_approval or argv is None) and approval_token != APPROVAL_TOKEN:
            return approval_required(decision.risk, decision.reason)
        if argv is None:
            return f"ERROR: 已审批命令没有可执行 argv: {command}"
        timeout_seconds = min(max(timeout_seconds, 1), 180)
        try:
            result = sandbox.run(argv, timeout=timeout_seconds)
        except FileNotFoundError as exc:
            return f"ERROR: {exc}"
        except subprocess.TimeoutExpired:
            return f"ERROR: 命令在 {timeout_seconds}s 后超时"

        output = result.stdout + result.stderr
        return truncate(
            f"ALLOWED[{decision.risk.value}]: {decision.reason}\n"
            f"$ {' '.join(argv)}\n"
            f"exit_code={result.returncode}\n"
            f"{output}"
        )

    return run_shell
