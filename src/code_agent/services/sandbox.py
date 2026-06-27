from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from code_agent.services.workspace import Workspace, WorkspaceError


@dataclass(frozen=True)
class SandboxPolicy:
    allow_network: bool = False
    inherit_environment: bool = True
    backend: str = "local"
    docker_image: str = "python:3.12-slim"
    docker_workspace: str = "/workspace"


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


class DockerSandbox(ShellSandbox):
    """Run shell commands inside a disposable Docker container.

    The host workspace is mounted as the container working directory. Permission
    rules still run before this backend, so Docker is an execution boundary, not
    a replacement for command classification.
    """

    def run(self, argv: list[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
        if not argv:
            raise ValueError("sandbox run requires argv")

        if shutil.which("docker") is None:
            raise FileNotFoundError("找不到 docker；请先安装 Docker，或改用 CODE_AGENT_SHELL_SANDBOX=local")

        self._assert_argv_paths_stay_in_workspace(argv)
        docker_argv = self._docker_argv(argv)
        return subprocess.run(
            docker_argv,
            cwd=self.workspace.root.resolve(),
            env=self._build_env(),
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

        if shutil.which("docker") is None:
            raise FileNotFoundError("找不到 docker；请先安装 Docker，或改用 CODE_AGENT_SHELL_SANDBOX=local")

        self._assert_shell_paths_stay_in_workspace(command)
        docker_argv = self._docker_argv(["sh", "-lc", command])
        return subprocess.run(
            docker_argv,
            cwd=self.workspace.root.resolve(),
            env=self._build_env(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            shell=False,
        )

    def _docker_argv(self, argv: list[str]) -> list[str]:
        workspace_root = str(self.workspace.root.resolve())
        container_workspace = self.policy.docker_workspace
        command = [
            "docker",
            "run",
            "--rm",
            "--workdir",
            container_workspace,
            "--volume",
            f"{workspace_root}:{container_workspace}",
        ]
        if not self.policy.allow_network:
            command.extend(["--network", "none"])
        command.append(self.policy.docker_image)
        command.extend(argv)
        return command


def build_shell_sandbox(workspace: Workspace, policy: SandboxPolicy | None = None) -> ShellSandbox:
    policy = policy or SandboxPolicy()
    if policy.backend == "docker":
        return DockerSandbox(workspace, policy)
    if policy.backend == "local":
        return ShellSandbox(workspace, policy)
    raise ValueError(f"未知 shell sandbox backend: {policy.backend}")


def describe_sandbox_policy(policy: SandboxPolicy) -> str:
    if policy.backend == "docker":
        network = "enabled" if policy.allow_network else "none"
        return f"docker image={policy.docker_image} network={network}"
    return policy.backend


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
