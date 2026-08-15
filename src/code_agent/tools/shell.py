from __future__ import annotations

import hashlib
import threading
import time
from pathlib import Path

from langchain_core.tools import tool

from code_agent.models import SandboxMode
from code_agent.services.trace import (
    capture_current_workspace_write_snapshots,
    record_current_shell_file_changes,
)
from code_agent.services.windows_sandbox import (
    SandboxUnavailableError,
    ShellExecutionSpec,
    WindowsRestrictedTokenSandbox,
    WindowsSandboxExecutor,
)
from code_agent.services.workspace import Workspace, WorkspaceError
from code_agent.tools.schemas import ShellCommandInput


class ShellDenialRegistry:
    """Remember the exact command and next escalation allowed by a real denial."""

    def __init__(self) -> None:
        self._latest: tuple[str, str, float] | None = None
        self._lock = threading.Lock()

    def remember(self, command: str, workdir: Path, permission: str) -> None:
        with self._lock:
            self._latest = (_fingerprint(command, workdir), permission, time.monotonic())

    def consume(self, command: str, workdir: Path, permission: str) -> bool:
        fingerprint = _fingerprint(command, workdir)
        with self._lock:
            latest = self._latest
            if latest is None:
                return False
            latest_fingerprint, allowed_permission, created_at = latest
            valid = (
                latest_fingerprint == fingerprint
                and allowed_permission == permission
                and time.monotonic() - created_at <= 300
            )
            if valid or time.monotonic() - created_at > 300:
                self._latest = None
            return valid


def build_shell_command_tool(
    workspace: Workspace,
    *,
    executor: WindowsSandboxExecutor | None = None,
    denial_registry: ShellDenialRegistry | None = None,
    default_timeout_ms: int = 10_000,
    max_timeout_ms: int = 120_000,
    output_limit: int = 12_000,
):
    sandbox = executor or WindowsRestrictedTokenSandbox(workspace.root)
    registry = denial_registry or ShellDenialRegistry()

    @tool(args_schema=ShellCommandInput)
    def shell_command(
        command: str,
        description: str,
        workdir: str = ".",
        timeout_ms: int | None = None,
        sandbox_permissions: str | None = None,
        justification: str | None = None,
    ) -> str:
        """Execute a PowerShell command (`pwsh -Command`) and return stdout/stderr.

        Every call starts a fresh pwsh process, so cwd, variables, and functions
        do not persist between calls. Use `workdir` instead of `cd`. Use native
        Windows paths and `$env:NAME` for environment variables. Results include
        `[exit code: N]`; timeouts also include `[timed out]`. Long stdout/stderr
        is truncated to its tail.

        Read-only and workspace-write commands run under a Windows file sandbox
        and PowerShell ConstrainedLanguage mode. Prefer cmdlets, properties, formatting, and
        core types; Add-Type, COM, reflection, and non-core .NET static calls may
        fail. Restricted processes cannot capture child-process output through
        named-pipe stdio: Node.js spawn/exec with `stdio: 'pipe'` may fail with
        `spawn EPERM`. After a real process-pipe denial, retry the exact same
        command once with `sandbox_permissions='danger-full-access'`. This
        requires separate user approval and runs with the user's normal Windows
        file permissions so tools such as Vite/esbuild can start pipe-based
        child processes. Never request it speculatively or change the command.

        Calls default to read-only. Attempt the command and inspect its sandbox
        marker rather than predicting denial. If a real file-access denial occurs
        and workspace writes would make it succeed, retry the exact same command
        and workdir once with `sandbox_permissions='workspace-write'` and a
        one-sentence justification. The retry triggers user approval and widens
        only that execution. Never escalate speculatively or work around a
        rejected escalation.
        """
        del description, justification
        try:
            resolved_workdir = workspace.resolve(workdir)
            if not resolved_workdir.exists():
                return f"ERROR: Shell working directory does not exist: {workdir}"
            if not resolved_workdir.is_dir():
                return f"ERROR: Shell working directory is not a directory: {workdir}"

            mode = {
                SandboxMode.workspace_write.value: SandboxMode.workspace_write,
                SandboxMode.danger_full_access.value: SandboxMode.danger_full_access,
            }.get(sandbox_permissions, SandboxMode.read_only)
            if mode != SandboxMode.read_only:
                if not registry.consume(command, resolved_workdir, mode.value):
                    return (
                        f"REJECTED[level_2]: {mode.value} is only valid for the exact command "
                        "after the corresponding real sandbox denial."
                    )
                capture_current_workspace_write_snapshots("shell_command")

            requested_timeout = default_timeout_ms if timeout_ms is None else timeout_ms
            effective_timeout = min(max(requested_timeout, 1), max_timeout_ms)
            result = sandbox.run(
                ShellExecutionSpec(
                    command=command,
                    workdir=resolved_workdir,
                    timeout_ms=effective_timeout,
                    mode=mode,
                )
            )
            if mode != SandboxMode.read_only:
                record_current_shell_file_changes()
            if (
                mode == SandboxMode.read_only
                and result.denial_kind == "file-access"
            ):
                registry.remember(command, resolved_workdir, SandboxMode.workspace_write.value)
            elif result.denial_kind == "process-pipe":
                registry.remember(command, resolved_workdir, SandboxMode.danger_full_access.value)

            return _render_result(result, output_limit=output_limit)
        except WorkspaceError as exc:
            return f"REJECTED[level_3]: {exc}"
        except SandboxUnavailableError as exc:
            return f"ERROR: SANDBOX_UNAVAILABLE: {exc}"

    return shell_command


def _fingerprint(command: str, workdir: Path) -> str:
    payload = f"{workdir.resolve()}\0{command}".encode("utf-8", errors="replace")
    return hashlib.sha256(payload).hexdigest()


def _render_result(result, *, output_limit: int) -> str:
    lines = [
        (
            f"[sandbox: mode={result.mode.value} enforcement={result.enforcement} "
            f"denied={'true' if result.sandbox_denied else 'false'}]"
        )
    ]
    if result.sandbox_denied:
        denial_label = (
            "process pipe access denied"
            if result.denial_kind == "process-pipe"
            else "file access denied"
        )
        lines.append(f"[sandbox: {denial_label} under {result.mode.value} mode]")
    if result.stdout:
        lines.extend(["stdout:", _truncate_tail(result.stdout.rstrip(), output_limit)])
    if result.stderr:
        lines.extend(["stderr:", _truncate_tail(result.stderr.rstrip(), output_limit)])
    if result.timed_out:
        lines.append("[timed out]")
    lines.append(f"[exit code: {result.exit_code if result.exit_code is not None else 'unknown'}]")
    if (
        result.sandbox_denied
        and result.denial_kind == "file-access"
        and result.mode == SandboxMode.read_only
    ):
        lines.append(
            "The sandbox blocked a write. Retry this exact command once with "
            "sandbox_permissions='workspace-write' and a justification if the write is necessary."
        )
    elif result.denial_kind == "process-pipe":
        lines.append(
            "The Windows restricted token blocked child-process pipe capture. Retry this exact "
            "command once with sandbox_permissions='danger-full-access' and a justification. "
            "That approval runs the command with normal Windows file permissions; do not change "
            "the command or try an alias/wrapper."
        )
    return "\n".join(lines)


def _truncate_tail(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"... truncated {omitted} leading characters ...\n{text[-limit:]}"
