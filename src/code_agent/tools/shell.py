from __future__ import annotations

import subprocess

from langchain_core.tools import tool

from code_agent.services.sandbox import build_shell_sandbox
from code_agent.services.summarizer import truncate
from code_agent.services.workspace import Workspace, WorkspaceError
from code_agent.tools.safety import classify_command, rejected
from code_agent.tools.schemas import RunShellInput


DEFAULT_SHELL_OUTPUT_LIMIT = 12_000
MAX_TIMEOUT_SECONDS = 180


def build_run_command_tool(
    workspace: Workspace,
    *,
    allow_requires_approval: bool = False,
):
    sandbox = build_shell_sandbox(workspace)

    @tool(args_schema=RunShellInput)
    def run_shell(command: str, timeout_seconds: int = 180) -> str:
        """
        Run validation or project setup commands inside the workspace sandbox.

        Use this only for:
        - tests
        - builds
        - linters/type checks
        - package installs
        - project scaffolding
        - short temporary scripts

        Do not use this for:
        - listing files
        - reading files
        - searching text
        - showing git diff/status
        - editing files
        - deleting files
        """
        decision, argv = classify_command(command, workspace)

        if argv is None:
            return f"ERROR: Command could not be parsed safely: {command}"

        if decision.risk.value == "level_3":
            return rejected(decision.risk, decision.reason)

        if decision.risk.value == "level_2" and not allow_requires_approval:
            return rejected(decision.risk, decision.reason)

        timeout_seconds = min(max(timeout_seconds, 1), MAX_TIMEOUT_SECONDS)

        try:
            # 注意：argv 目前只用于 classify_command 的安全判断。
            # 真正执行仍然交给 sandbox.run_shell(command)。
            # 所以 classify_command 必须完整检查 command，包括 ; && || | > >> $() `...` 等 shell 组合。
            result = sandbox.run_shell(command, timeout=timeout_seconds)

        except FileNotFoundError as exc:
            return f"ERROR: {exc}"
        except WorkspaceError as exc:
            return f"ERROR: {exc}"
        except subprocess.TimeoutExpired:
            return f"ERROR: Command timed out after {timeout_seconds}s"

        output = (result.stdout + result.stderr).strip()

        if result.returncode != 0:
            return truncate(
                f"exit_code={result.returncode}\n{output}",
                DEFAULT_SHELL_OUTPUT_LIMIT,
            )

        return truncate(
            output or "Command completed successfully.",
            DEFAULT_SHELL_OUTPUT_LIMIT,
        )

    return run_shell
