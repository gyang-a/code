from __future__ import annotations

from pathlib import Path
import unittest

from code_agent.services.sandbox import ShellSandbox
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


class TemporaryWorkspace:
    def __enter__(self) -> Path:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        return Path(self._tmp.name)

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._tmp.cleanup()
