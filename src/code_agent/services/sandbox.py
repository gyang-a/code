from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from code_agent.services.workspace import Workspace, WorkspaceError


@dataclass(frozen=True)
class SandboxPolicy:
    allow_network: bool = False
    inherit_environment: bool = True


class ShellSandbox:
    """Runtime boundary for shell commands and child processes.

    This MVP uses shell=False, a fixed cwd, a scrubbed environment hook, and
    workspace checks. The class is intentionally isolated so a future backend
    can swap in a real OS/container sandbox without touching permission rules.
    """

    def __init__(self, workspace: Workspace, policy: SandboxPolicy | None = None) -> None:
        self.workspace = workspace
        self.policy = policy or SandboxPolicy()

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
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            shell=False,
        )

    def _build_env(self) -> dict[str, str]:
        if self.policy.inherit_environment:
            env = dict(os.environ)
        else:
            env = {}

        if not self.policy.allow_network:
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
