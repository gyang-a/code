from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from code_agent.models import SandboxMode


DENIAL_SIGNATURES = (
    "access to the path",
    "access is denied",
    "permission denied",
    "operation not permitted",
    "unauthorizedaccessexception",
    "requested registry access is not allowed",
    "eperm",
    "eacces",
    "拒绝访问",
    "访问被拒绝",
    "winerror 5",
)
PROCESS_PIPE_DENIAL_SIGNATURES = (
    "spawn eperm",
    "spawn eacces",
)
SENSITIVE_ENV_RE = re.compile(
    r"(api[_-]?key|access[_-]?token|auth[_-]?token|secret|password|passwd|credential)",
    re.IGNORECASE,
)


class SandboxUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class ShellExecutionSpec:
    command: str
    workdir: Path
    timeout_ms: int
    mode: SandboxMode


@dataclass(frozen=True)
class ShellExecutionResult:
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool
    sandbox_denied: bool
    mode: SandboxMode
    enforcement: str = "partial"
    denial_kind: str | None = None


class WindowsSandboxExecutor(Protocol):
    def run(self, spec: ShellExecutionSpec) -> ShellExecutionResult: ...


class WindowsRestrictedTokenSandbox:
    """Run PowerShell with a Windows WRITE_RESTRICTED token.

    The restricted SID list is the capability boundary. Read-only tokens receive
    only a deterministic workspace read capability. Workspace-write tokens also
    receive deterministic workspace-write and per-call temporary-directory
    capabilities. Windows ACL and hard-link semantics make this a partial, not
    absolute, write boundary.
    """

    _acl_lock = threading.Lock()

    def __init__(self, workspace_root: str | Path) -> None:
        if sys.platform != "win32":
            raise SandboxUnavailableError("The shell sandbox is available only on Windows.")

        self.workspace_root = Path(workspace_root).expanduser().resolve()
        self.powershell = _find_powershell()

        try:
            import win32api  # noqa: F401
            import win32job  # noqa: F401
            import win32process  # noqa: F401
            import win32security  # noqa: F401
        except ImportError as exc:
            raise SandboxUnavailableError(
                "pywin32 is required for the Windows restricted-token sandbox."
            ) from exc

    def run(self, spec: ShellExecutionSpec) -> ShellExecutionResult:
        if spec.workdir != self.workspace_root and self.workspace_root not in spec.workdir.parents:
            raise SandboxUnavailableError("Shell working directory escapes the workspace.")

        temp_dir = Path(tempfile.mkdtemp(prefix="code-agent-shell-"))
        stdout_path = temp_dir / "stdout.txt"
        stderr_path = temp_dir / "stderr.txt"

        try:
            token = self._create_token(spec.mode, temp_dir)
            exit_code, timed_out = self._spawn_and_wait(
                token,
                spec,
                temp_dir=temp_dir,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
            )
            stdout = _read_output(stdout_path)
            stderr = _read_output(stderr_path)
            denial_kind = _classify_denial(stdout, stderr)
            return ShellExecutionResult(
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                timed_out=timed_out,
                sandbox_denied=denial_kind is not None,
                mode=spec.mode,
                enforcement=(
                    "unrestricted"
                    if spec.mode == SandboxMode.danger_full_access
                    else "partial"
                ),
                denial_kind=denial_kind,
            )
        except SandboxUnavailableError:
            raise
        except Exception as exc:
            raise SandboxUnavailableError(f"Windows sandbox initialization failed: {exc}") from exc
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _create_token(self, mode: SandboxMode, temp_dir: Path):
        import win32api
        import win32con
        import win32security

        source = win32security.OpenProcessToken(
            win32api.GetCurrentProcess(),
            win32con.TOKEN_ALL_ACCESS,
        )
        if mode == SandboxMode.danger_full_access:
            # This mode is reachable only through the host approval gate after
            # a real process-pipe denial. It deliberately uses the caller's
            # normal token so Node/esbuild can create child-process stdio pipes.
            return source

        groups = win32security.GetTokenInformation(source, win32security.TokenGroups)
        logon_sids = [sid for sid, attrs in groups if attrs & win32con.SE_GROUP_LOGON_ID]
        if len(logon_sids) != 1:
            raise SandboxUnavailableError("Could not resolve the current Windows logon SID.")

        everyone = win32security.CreateWellKnownSid(win32security.WinWorldSid)
        read_sid = _capability_sid("workspace-read", self.workspace_root)
        with self._acl_lock:
            _grant_directory_access(self.workspace_root, read_sid, writable=False)
        restricting_sids = [(logon_sids[0], 0), (everyone, 0), (read_sid, 0)]

        if mode == SandboxMode.workspace_write:
            workspace_sid = _capability_sid("workspace-write", self.workspace_root)
            temp_sid = _capability_sid("temp", temp_dir)
            with self._acl_lock:
                _grant_directory_access(self.workspace_root, workspace_sid, writable=True)
                _grant_directory_access(temp_dir, temp_sid, writable=True)
            restricting_sids.extend([(workspace_sid, 0), (temp_sid, 0)])

        # DISABLE_MAX_PRIVILEGE | LUA_TOKEN | WRITE_RESTRICTED. pywin32 does
        # not publish names for the latter two flags on all supported builds.
        flags = 0x1 | 0x4 | 0x8
        return win32security.CreateRestrictedToken(
            source,
            flags,
            [],
            [],
            restricting_sids,
        )

    def _spawn_and_wait(
        self,
        token,
        spec: ShellExecutionSpec,
        *,
        temp_dir: Path,
        stdout_path: Path,
        stderr_path: Path,
    ) -> tuple[int | None, bool]:
        import msvcrt

        import win32api
        import win32con
        import win32event
        import win32job
        import win32process

        command_line = subprocess.list2cmdline(
            [
                str(self.powershell),
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                spec.command,
            ]
        )
        environment = _sanitized_environment(temp_dir)
        creation_flags = (
            win32con.CREATE_UNICODE_ENVIRONMENT
            | win32con.CREATE_NEW_PROCESS_GROUP
            | win32con.CREATE_SUSPENDED
        )

        with open(os.devnull, "rb") as stdin_handle, open(stdout_path, "w+b") as stdout_handle, open(
            stderr_path, "w+b"
        ) as stderr_handle:
            raw_handles = [
                msvcrt.get_osfhandle(stdin_handle.fileno()),
                msvcrt.get_osfhandle(stdout_handle.fileno()),
                msvcrt.get_osfhandle(stderr_handle.fileno()),
            ]
            for raw_handle in raw_handles:
                os.set_handle_inheritable(raw_handle, True)

            startup = win32process.STARTUPINFO()
            startup.dwFlags |= win32process.STARTF_USESTDHANDLES
            startup.hStdInput, startup.hStdOutput, startup.hStdError = raw_handles

            process_info = win32process.CreateProcessAsUser(
                token,
                str(self.powershell),
                command_line,
                None,
                None,
                True,
                creation_flags,
                environment,
                str(spec.workdir),
                startup,
            )
            process_handle, thread_handle, _process_id, _thread_id = process_info
            job = None

            try:
                # The child starts suspended so it cannot spawn an uncontained
                # descendant before assignment to the kill-on-close job.
                job = win32job.CreateJobObject(None, "")
                info = win32job.QueryInformationJobObject(
                    job,
                    win32job.JobObjectExtendedLimitInformation,
                )
                info["BasicLimitInformation"]["LimitFlags"] |= win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
                win32job.SetInformationJobObject(
                    job,
                    win32job.JobObjectExtendedLimitInformation,
                    info,
                )
                win32job.AssignProcessToJobObject(job, process_handle)
                win32process.ResumeThread(thread_handle)

                wait_result = win32event.WaitForSingleObject(process_handle, spec.timeout_ms)
                timed_out = wait_result == win32con.WAIT_TIMEOUT
                if timed_out:
                    win32job.TerminateJobObject(job, 124)
                    win32event.WaitForSingleObject(process_handle, 5_000)
                exit_code = win32process.GetExitCodeProcess(process_handle)
                if exit_code == win32con.STILL_ACTIVE:
                    exit_code = None
                return exit_code, timed_out
            except Exception:
                try:
                    win32process.TerminateProcess(process_handle, 125)
                except Exception:
                    pass
                raise
            finally:
                win32api.CloseHandle(thread_handle)
                win32api.CloseHandle(process_handle)
                if job is not None:
                    win32api.CloseHandle(job)
                for raw_handle in raw_handles:
                    os.set_handle_inheritable(raw_handle, False)


def _find_powershell() -> Path:
    executable = shutil.which("pwsh") or shutil.which("powershell")
    if not executable:
        raise SandboxUnavailableError("Neither pwsh nor Windows PowerShell is installed.")
    return Path(executable).resolve()


def _capability_sid(domain: str, path: Path):
    import win32security

    canonical = os.path.normcase(str(path.expanduser().resolve())).encode("utf-8")
    digest = hashlib.sha256(domain.encode("ascii") + b"\0" + canonical).digest()
    first = int.from_bytes(digest[:4], "little") or 1
    second = int.from_bytes(digest[4:8], "little") or 1
    return win32security.ConvertStringSidToSid(f"S-1-4-{first}-{second}")


def _grant_directory_access(path: Path, sid, *, writable: bool) -> None:
    import ntsecuritycon
    import win32security

    flags = win32security.DACL_SECURITY_INFORMATION
    descriptor = win32security.GetNamedSecurityInfo(
        str(path),
        win32security.SE_FILE_OBJECT,
        flags,
    )
    dacl = descriptor.GetSecurityDescriptorDacl()
    if dacl is None:
        raise SandboxUnavailableError(f"Refusing to modify a NULL DACL: {path}")

    sid_text = win32security.ConvertSidToStringSid(sid)
    access_mask = ntsecuritycon.FILE_GENERIC_READ | ntsecuritycon.FILE_GENERIC_EXECUTE
    if writable:
        access_mask |= (
            ntsecuritycon.FILE_GENERIC_WRITE
            | ntsecuritycon.DELETE
            | ntsecuritycon.FILE_DELETE_CHILD
        )
    for index in range(dacl.GetAceCount()):
        _header, existing_mask, existing_sid = dacl.GetAce(index)
        if (
            win32security.ConvertSidToStringSid(existing_sid) == sid_text
            and existing_mask == access_mask
        ):
            return

    inheritance = win32security.OBJECT_INHERIT_ACE | win32security.CONTAINER_INHERIT_ACE
    dacl.AddAccessAllowedAceEx(win32security.ACL_REVISION_DS, inheritance, access_mask, sid)
    win32security.SetNamedSecurityInfo(
        str(path),
        win32security.SE_FILE_OBJECT,
        flags,
        None,
        None,
        dacl,
        None,
    )


def _sanitized_environment(temp_dir: Path) -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith("DSH_") and not SENSITIVE_ENV_RE.search(key)
    }
    environment["TEMP"] = str(temp_dir)
    environment["TMP"] = str(temp_dir)
    environment["CODE_AGENT_SANDBOX"] = "windows-restricted-token"
    return environment


def _read_output(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def _looks_like_denial(stdout: str, stderr: str) -> bool:
    return _classify_denial(stdout, stderr) is not None


def _classify_denial(stdout: str, stderr: str) -> str | None:
    lowered = f"{stdout}\n{stderr}".lower()
    if any(signature in lowered for signature in PROCESS_PIPE_DENIAL_SIGNATURES):
        return "process-pipe"
    if any(signature in lowered for signature in DENIAL_SIGNATURES):
        return "file-access"
    return None
