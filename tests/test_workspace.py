from __future__ import annotations

from pathlib import Path
import unittest

from code_agent.services.workspace import Workspace, WorkspaceError


class WorkspaceTests(unittest.TestCase):
    def test_resolve_rejects_path_escape(self) -> None:
        with TemporaryWorkspace() as tmp_path:
            workspace = Workspace(tmp_path)

            with self.assertRaises(WorkspaceError):
                workspace.resolve("../outside.txt")

    def test_read_file_rejects_sensitive_file(self) -> None:
        with TemporaryWorkspace() as tmp_path:
            (tmp_path / ".env").write_text("SECRET=1", encoding="utf-8")
            workspace = Workspace(tmp_path)

            with self.assertRaises(WorkspaceError):
                workspace.read_text(".env")

    def test_read_file_rejects_large_file(self) -> None:
        with TemporaryWorkspace() as tmp_path:
            (tmp_path / "large.txt").write_text("x" * 11, encoding="utf-8")
            workspace = Workspace(tmp_path, read_limit=10)

            with self.assertRaises(WorkspaceError):
                workspace.read_text("large.txt")


class TemporaryWorkspace:
    def __enter__(self) -> Path:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        return Path(self._tmp.name)

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._tmp.cleanup()
