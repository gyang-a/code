from __future__ import annotations
import hashlib
import threading
from collections import Counter
from langchain_core.tools import tool
from code_agent.models import SandboxMode, RiskLevel
from code_agent.services.shell_policy import classify_shell
from code_agent.services.trace import capture_current_workspace_write_snapshots, record_current_shell_file_changes
from code_agent.services.windows_sandbox import SandboxUnavailableError, ShellExecutionSpec, WindowsRestrictedTokenSandbox
from code_agent.services.workspace import Workspace, WorkspaceError
from code_agent.tools.schemas import ShellCommandInput
from code_agent.services.summarizer import raw_tool_output


def build_shell_command_tool(workspace: Workspace, *, executor=None,
        default_timeout_ms=10_000, max_timeout_ms=120_000, output_limit=12_000):
    sandbox = executor or WindowsRestrictedTokenSandbox(workspace.root)
    approvals = Counter()
    lock = threading.Lock()

    def fingerprint(args):
        payload = repr((args.get('command'), str(workspace.resolve(args.get('workdir', '.'))),
                        args.get('timeout_ms'), workspace.shell_mode,
                        workspace.shell_approval_policy, workspace.shell_allowed_commands))
        return hashlib.sha256(payload.encode()).hexdigest()

    def authorize(args):
        result = classify_shell(workspace, args)
        if result.requires_approval:
            with lock:
                approvals[fingerprint(args)] += 1

    @tool(args_schema=ShellCommandInput)
    def shell_command(command: str, description: str, workdir: str = '.',
                      timeout_ms: int | None = None) -> str:
        """Run PowerShell inside the host-selected restricted-token sandbox.

        Known reads and host-authorized commands run directly. Deletions and
        unknown scripts require approval; forbidden paths/actions are rejected.
        Approval never grants full access. Each call starts a fresh process; use
        workdir. Inspect exit code, timeout and denial markers.
        """
        args = dict(command=command, workdir=workdir, timeout_ms=timeout_ms)
        try:
            result = classify_shell(workspace, args)
            if result.risk == RiskLevel.level_3:
                return f'REJECTED[level_3]: {result.reason}'
            if result.requires_approval:
                key = fingerprint(args)
                with lock:
                    if not approvals[key]:
                        return f'REJECTED[level_2]: Approval required. {result.reason}'
                    approvals[key] -= 1
            cwd = workspace.resolve(workdir)
            if not cwd.is_dir():
                return 'ERROR: Shell working directory does not exist.'
            mode = SandboxMode(workspace.shell_mode)
            if mode == SandboxMode.workspace_write:
                capture_current_workspace_write_snapshots('shell_command')
            try:
                execution = sandbox.run(ShellExecutionSpec(command, cwd,
                    min(max(timeout_ms or default_timeout_ms, 1), max_timeout_ms), mode))
            finally:
                if mode == SandboxMode.workspace_write:
                    record_current_shell_file_changes()
            return _render_result(execution, output_limit=output_limit)
        except WorkspaceError as exc:
            return f'REJECTED[level_3]: {exc}'
        except SandboxUnavailableError as exc:
            return f'ERROR: SANDBOX_UNAVAILABLE: {exc}'

    object.__setattr__(shell_command, '_authorize', authorize)
    return shell_command


def _render_result(result, *, output_limit):
    lines = [f'[sandbox: mode={result.mode.value} enforcement={result.enforcement} '
             f'denied={str(result.sandbox_denied).lower()}]']
    for label, value in [('stdout', result.stdout), ('stderr', result.stderr)]:
        if value:
            lines.extend([label + ':', _truncate_tail(value.rstrip(), output_limit)])
    if result.timed_out:
        lines.append('[timed out]')
    lines.append(f'[exit code: {result.exit_code if result.exit_code is not None else "unknown"}]')
    if result.sandbox_denied:
        lines.append('[possible access denial detected in output; full-access retry is disabled]')
    return '\n'.join(lines)


def _truncate_tail(text, limit):
    return text if raw_tool_output.get() or len(text) <= limit else f'... truncated {len(text)-limit} leading characters ...\n{text[-limit:]}'
