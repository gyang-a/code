from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from code_agent.services.workspace import Workspace, WorkspaceError


class ShellSandbox:
    """Runtime boundary for shell commands and child processes.

    Shell commands run through the host shell with a fixed workspace cwd,
    closed stdin, bounded timeouts, and path escape checks.
    """

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    def run(self, argv: list[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
        if not argv:
            raise ValueError("sandbox run requires argv")

        executable = shutil.which(argv[0])
        if executable is None:
            raise FileNotFoundError(f"找不到可执行文件: {argv[0]}")

        cwd = self.workspace.root.resolve()
        try:
            cwd.relative_to(self.workspace.root)
        except ValueError as exc:
            raise WorkspaceError(f"Shell sandbox cwd escapes workspace: {cwd}") from exc

        self._assert_argv_paths_stay_in_workspace(argv)

        return subprocess.run(
            argv,
            cwd=cwd,
            env=self._build_env(),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            shell=False,
        )

    def run_shell(self, command: str, *, timeout: int) -> subprocess.CompletedProcess[str]:
        if not command.strip():
            raise ValueError("sandbox run_shell requires a command")

        cwd = self.workspace.root.resolve()
        try:
            cwd.relative_to(self.workspace.root)
        except ValueError as exc:
            raise WorkspaceError(f"Shell sandbox cwd escapes workspace: {cwd}") from exc

        self._assert_shell_paths_stay_in_workspace(command)

        return subprocess.run(
            _platform_shell_argv(command),
            cwd=cwd,
            env=self._build_env(),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            shell=False,
        )

    def _build_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.setdefault("CODE_AGENT_NETWORK_DISABLED", "1")
        return env

    def _assert_argv_paths_stay_in_workspace(self, argv: list[str]) -> None:
        for arg in argv[1:]:
            if _looks_like_option(arg):
                continue
            if not _looks_like_path(arg):
                continue
            candidate = Path(arg)
            if candidate.is_absolute() or ".." in candidate.parts:
                resolved = candidate.resolve() if candidate.is_absolute() else (self.workspace.root / candidate).resolve()
                try:
                    resolved.relative_to(self.workspace.root)
                except ValueError as exc:
                    raise WorkspaceError(f"Shell sandbox rejected path outside workspace: {arg}") from exc

    def _assert_shell_paths_stay_in_workspace(self, command: str) -> None:
        for arg in _split_command_for_path_scan(command):
            if _looks_like_option(arg):
                continue
            if not _looks_like_path(arg):
                continue
            candidate = Path(arg.strip("\"'"))
            if candidate.is_absolute() or ".." in candidate.parts:
                resolved = candidate.resolve() if candidate.is_absolute() else (self.workspace.root / candidate).resolve()
                try:
                    resolved.relative_to(self.workspace.root)
                except ValueError as exc:
                    raise WorkspaceError(f"Shell sandbox rejected path outside workspace: {arg}") from exc


def build_shell_sandbox(workspace: Workspace) -> ShellSandbox:
    return ShellSandbox(workspace)


def describe_sandbox_policy() -> str:
    return "local"


def _looks_like_option(value: str) -> bool:
    return value.startswith("-") and value not in {"-", "--"}


def _looks_like_path(value: str) -> bool:
    return (
        "/" in value
        or "\\" in value
        or value.startswith(".")
        or value.startswith("~")
        or ":" in value
    )


def _platform_shell_argv(command: str) -> list[str]:
    if sys.platform == "win32":
        return ["cmd.exe", "/d", "/s", "/c", command]
    shell = shutil.which("bash") or shutil.which("sh")
    if shell is None:
        raise FileNotFoundError("No shell executable found: bash or sh")
    return [shell, "-lc", command]


def _split_command_for_path_scan(command: str) -> list[str]:
    import shlex

    try:
        return shlex.split(command, posix=sys.platform != "win32")
    except ValueError:
        return command.split()
