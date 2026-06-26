from __future__ import annotations

import subprocess

from langchain_core.tools import tool

from code_agent.services.summarizer import truncate
from code_agent.services.workspace import Workspace


def _run_git(workspace: Workspace, args: list[str], timeout: int = 10) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=workspace.root,
        capture_output=True,
        text=True,
        timeout=timeout,
        shell=False,
    )
    output = result.stdout + result.stderr
    return truncate(output)


def build_git_status_tool(workspace: Workspace):
    @tool
    def git_status() -> str:
        """Show git status in short format for the workspace."""
        try:
            return _run_git(workspace, ["status", "--short"])
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            return f"ERROR: {exc}"

    return git_status


def build_git_diff_tool(workspace: Workspace):
    @tool
    def git_diff(path: str = ".") -> str:
        """Show git diff for the workspace or one path."""
        try:
            resolved = workspace.resolve(path)
            rel = "." if resolved == workspace.root else workspace.relative(resolved)
            return _run_git(workspace, ["diff", "--", rel], timeout=20)
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError) as exc:
            return f"ERROR: {exc}"

    return git_diff
