from __future__ import annotations

import subprocess
from pathlib import Path
import unittest
from unittest.mock import patch

from code_agent.services.sandbox import ShellSandbox, build_shell_sandbox
from code_agent.services.workspace import Workspace, WorkspaceError


class SandboxTests(unittest.TestCase):
    def test_rejects_obvious_path_escape_in_argv(self) -> None:
        with TemporaryWorkspace() as tmp_path:
            workspace = Workspace(tmp_path)
            sandbox = ShellSandbox(workspace)

            with self.assertRaises(WorkspaceError):
                sandbox.run(["python", "../outside.py"], timeout=5)

    def test_allows_workspace_relative_script_path(self) -> None:
        with TemporaryWorkspace() as tmp_path:
            workspace = Workspace(tmp_path)
            sandbox = ShellSandbox(workspace)

            result = sandbox.run(["python", "-c", "print('ok')"], timeout=5)

            self.assertEqual(result.returncode, 0)
            self.assertIn("ok", result.stdout)

    def test_run_shell_supports_shell_builtins(self) -> None:
        with TemporaryWorkspace() as tmp_path:
            workspace = Workspace(tmp_path)
            sandbox = ShellSandbox(workspace)

            result = sandbox.run_shell("echo ok", timeout=5)

            self.assertEqual(result.returncode, 0)
            self.assertIn("ok", result.stdout)

    def test_run_shell_closes_stdin_to_prevent_interactive_hangs(self) -> None:
        with TemporaryWorkspace() as tmp_path:
            workspace = Workspace(tmp_path)
            sandbox = ShellSandbox(workspace)

            with patch("code_agent.services.sandbox.subprocess.run") as run_mock:
                run_mock.return_value.returncode = 0
                run_mock.return_value.stdout = "ok\n"
                run_mock.return_value.stderr = ""

                sandbox.run_shell("echo ok", timeout=5)

            self.assertIs(run_mock.call_args.kwargs["stdin"], subprocess.DEVNULL)

    def test_run_shell_rejects_obvious_path_escape(self) -> None:
        with TemporaryWorkspace() as tmp_path:
            workspace = Workspace(tmp_path)
            sandbox = ShellSandbox(workspace)

            with self.assertRaises(WorkspaceError):
                sandbox.run_shell("cd ..", timeout=5)

    def test_build_shell_sandbox_returns_local_shell_sandbox(self) -> None:
        with TemporaryWorkspace() as tmp_path:
            workspace = Workspace(tmp_path)
            sandbox = build_shell_sandbox(workspace)

            self.assertIsInstance(sandbox, ShellSandbox)


class TemporaryWorkspace:
    def __enter__(self) -> Path:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        return Path(self._tmp.name)

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._tmp.cleanup()
