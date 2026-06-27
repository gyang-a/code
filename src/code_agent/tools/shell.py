from __future__ import annotations

import subprocess

from langchain_core.tools import tool

from code_agent.services.sandbox import SandboxPolicy, build_shell_sandbox
from code_agent.services.summarizer import truncate
from code_agent.services.workspace import Workspace, WorkspaceError
from code_agent.tools.safety import classify_command, rejected
from code_agent.tools.schemas import RunShellInput


def build_run_command_tool(workspace: Workspace, *, sandbox_policy: SandboxPolicy | None = None):
    sandbox = build_shell_sandbox(workspace, sandbox_policy)

    @tool(args_schema=RunShellInput)
    def run_shell(command: str, timeout_seconds: int = 60) -> str:
        """Run a shell command in the workspace sandbox."""
        decision, argv = classify_command(command, workspace)
        if decision.risk.value == "level_3":
            return rejected(decision.risk, decision.reason)
        if argv is None:
            return f"ERROR: Command could not be parsed safely: {command}"

        timeout_seconds = min(max(timeout_seconds, 1), 180)
        try:
            result = sandbox.run_shell(command, timeout=timeout_seconds)
        except FileNotFoundError as exc:
            return f"ERROR: {exc}"
        except WorkspaceError as exc:
            return f"ERROR: {exc}"
        except subprocess.TimeoutExpired:
            return f"ERROR: Command timed out after {timeout_seconds}s"

        output = result.stdout + result.stderr
        if result.returncode != 0:
            return truncate(f"exit_code={result.returncode}\n{output}".strip())
        return truncate(output.strip() or "Command completed successfully.")

    return run_shell
