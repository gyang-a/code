from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from code_agent.models import SandboxMode
from code_agent.services.windows_sandbox import (
    ShellExecutionResult,
    _classify_denial,
    _looks_like_denial,
    _sanitized_environment,
)
from code_agent.services.workspace import Workspace
from code_agent.tools.schemas import ShellCommandInput
from code_agent.tools.shell import ShellDenialRegistry, build_shell_command_tool


class FakeSandbox:
    def __init__(self, results: list[ShellExecutionResult]) -> None:
        self.results = list(results)
        self.calls = []

    def run(self, spec):
        self.calls.append(spec)
        return self.results.pop(0)


class ShellToolTests(unittest.TestCase):
    def test_defaults_to_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sandbox = FakeSandbox([_result(mode=SandboxMode.read_only, stdout="ok")])
            tool = build_shell_command_tool(Workspace(tmp), executor=sandbox)

            output = tool.invoke({"command": "Get-ChildItem", "description": "List files"})

        self.assertEqual(sandbox.calls[0].mode, SandboxMode.read_only)
        self.assertIn("mode=read-only", output)
        self.assertIn("stdout:\nok", output)

    def test_workspace_write_requires_a_prior_exact_denial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sandbox = FakeSandbox([_result(mode=SandboxMode.workspace_write)])
            tool = build_shell_command_tool(Workspace(tmp), executor=sandbox)

            output = tool.invoke(
                {
                    "command": "pytest",
                    "description": "Run tests",
                    "sandbox_permissions": "workspace-write",
                    "justification": "Tests create cache files.",
                }
            )

        self.assertTrue(output.startswith("REJECTED[level_2]"))
        self.assertEqual(sandbox.calls, [])

    def test_denied_command_can_be_retried_once_with_workspace_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sandbox = FakeSandbox(
                [
                    _result(mode=SandboxMode.read_only, denied=True, stderr="Access is denied"),
                    _result(mode=SandboxMode.workspace_write, stdout="passed"),
                ]
            )
            registry = ShellDenialRegistry()
            tool = build_shell_command_tool(
                Workspace(tmp),
                executor=sandbox,
                denial_registry=registry,
            )
            base = {"command": "pytest", "description": "Run tests", "workdir": "."}

            denied = tool.invoke(base)
            approved = tool.invoke(
                {
                    **base,
                    "sandbox_permissions": "workspace-write",
                    "justification": "Tests create cache files.",
                }
            )
            second_retry = tool.invoke(
                {
                    **base,
                    "sandbox_permissions": "workspace-write",
                    "justification": "Try the same write again.",
                }
            )

        self.assertIn("denied=true", denied)
        self.assertIn("mode=workspace-write", approved)
        self.assertTrue(second_retry.startswith("REJECTED[level_2]"))
        self.assertEqual([call.mode for call in sandbox.calls], [SandboxMode.read_only, SandboxMode.workspace_write])

    def test_changed_command_does_not_match_denial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sandbox = FakeSandbox(
                [_result(mode=SandboxMode.read_only, denied=True, stderr="Access is denied")]
            )
            tool = build_shell_command_tool(Workspace(tmp), executor=sandbox)
            tool.invoke({"command": "pytest", "description": "Run tests"})

            output = tool.invoke(
                {
                    "command": "Remove-Item important.txt",
                    "description": "Delete file",
                    "sandbox_permissions": "workspace-write",
                    "justification": "This command needs to delete a file.",
                }
            )

        self.assertTrue(output.startswith("REJECTED[level_2]"))
        self.assertEqual(len(sandbox.calls), 1)

    def test_workdir_cannot_escape_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sandbox = FakeSandbox([])
            tool = build_shell_command_tool(Workspace(tmp), executor=sandbox)

            output = tool.invoke(
                {"command": "Get-ChildItem", "description": "List parent", "workdir": ".."}
            )

        self.assertTrue(output.startswith("REJECTED[level_3]"))
        self.assertEqual(sandbox.calls, [])

    def test_escalation_fields_must_be_paired(self) -> None:
        with self.assertRaises(ValidationError):
            ShellCommandInput(
                command="pytest",
                description="Run tests",
                sandbox_permissions="workspace-write",
            )

    def test_shell_timeout_is_capped_by_host(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sandbox = FakeSandbox([_result(mode=SandboxMode.read_only)])
            tool = build_shell_command_tool(
                Workspace(tmp),
                executor=sandbox,
                max_timeout_ms=5_000,
            )

            tool.invoke(
                {
                    "command": "Start-Sleep -Seconds 10",
                    "description": "Wait briefly",
                    "timeout_ms": 60_000,
                }
            )

        self.assertEqual(sandbox.calls[0].timeout_ms, 5_000)

    def test_sensitive_environment_variables_are_removed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            "os.environ",
            {
                "DEEPSEEK_API_KEY": "secret",
                "SERVICE_ACCESS_TOKEN": "secret",
                "SAFE_SETTING": "visible",
            },
            clear=True,
        ):
            environment = _sanitized_environment(Path(tmp))

        self.assertNotIn("DEEPSEEK_API_KEY", environment)
        self.assertNotIn("SERVICE_ACCESS_TOKEN", environment)
        self.assertEqual(environment["SAFE_SETTING"], "visible")

    def test_stdout_eperm_is_classified_as_sandbox_denial(self) -> None:
        self.assertTrue(
            _looks_like_denial(
                "Could not write tsconfig.tsbuildinfo: EPERM: operation not permitted",
                "",
            )
        )
        self.assertEqual(
            _classify_denial(
                "Could not write tsconfig.tsbuildinfo: EPERM: operation not permitted",
                "",
            ),
            "file-access",
        )

    def test_stderr_spawn_eperm_is_classified_as_sandbox_denial(self) -> None:
        self.assertTrue(_looks_like_denial("", "Error: spawn EPERM"))
        self.assertEqual(_classify_denial("", "Error: spawn EPERM"), "process-pipe")

    def test_normal_nonzero_error_is_not_classified_as_sandbox_denial(self) -> None:
        self.assertFalse(_looks_like_denial("", "TypeScript error TS2322"))

    def test_process_pipe_denial_allows_only_exact_danger_full_access_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sandbox = FakeSandbox(
                [
                    ShellExecutionResult(
                        exit_code=1,
                        stdout="",
                        stderr="Error: spawn EPERM",
                        timed_out=False,
                        sandbox_denied=True,
                        mode=SandboxMode.read_only,
                        denial_kind="process-pipe",
                    ),
                    _result(mode=SandboxMode.danger_full_access, stdout="built"),
                ]
            )
            tool = build_shell_command_tool(Workspace(tmp), executor=sandbox)
            base = {"command": "node build.js", "description": "Run build"}

            denied = tool.invoke(base)
            retry = tool.invoke(
                {
                    **base,
                    "sandbox_permissions": "workspace-write",
                    "justification": "Retry the build with workspace writes.",
                }
            )
            approved = tool.invoke(
                {
                    **base,
                    "sandbox_permissions": "danger-full-access",
                    "justification": "Vite requires pipe-based child processes.",
                }
            )

        self.assertIn("process pipe access denied", denied)
        self.assertIn("sandbox_permissions='danger-full-access'", denied)
        self.assertTrue(retry.startswith("REJECTED[level_2]"))
        self.assertIn("mode=danger-full-access", approved)
        self.assertEqual(len(sandbox.calls), 2)

    def test_danger_full_access_cannot_be_requested_speculatively(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sandbox = FakeSandbox([])
            tool = build_shell_command_tool(Workspace(tmp), executor=sandbox)

            output = tool.invoke(
                {
                    "command": "npm run build",
                    "description": "Build project",
                    "sandbox_permissions": "danger-full-access",
                    "justification": "Build might need child processes.",
                }
            )

        self.assertTrue(output.startswith("REJECTED[level_2]"))
        self.assertEqual(sandbox.calls, [])

    def test_tool_description_documents_supported_windows_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tool = build_shell_command_tool(
                Workspace(tmp),
                executor=FakeSandbox([]),
            )

        self.assertIn("fresh pwsh process", tool.description)
        self.assertIn("ConstrainedLanguage", tool.description)
        self.assertIn("named-pipe stdio", tool.description)
        self.assertIn("exact same command", tool.description)
        self.assertIn("danger-full-access", tool.description)


def _result(
    *,
    mode: SandboxMode,
    stdout: str = "",
    stderr: str = "",
    denied: bool = False,
) -> ShellExecutionResult:
    return ShellExecutionResult(
        exit_code=1 if denied else 0,
        stdout=stdout,
        stderr=stderr,
        timed_out=False,
        sandbox_denied=denied,
        mode=mode,
        denial_kind="file-access" if denied else None,
    )


if __name__ == "__main__":
    unittest.main()
