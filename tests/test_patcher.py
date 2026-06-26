from __future__ import annotations

from pathlib import Path
import unittest

from code_agent.services.patcher import replace_exact_once
from code_agent.services.workspace import Workspace


class PatcherTests(unittest.TestCase):
    def test_replace_exact_once_rejects_ambiguous_match(self) -> None:
        with TemporaryWorkspace() as tmp_path:
            (tmp_path / "app.py").write_text("print(1)\nprint(1)\n", encoding="utf-8")
            workspace = Workspace(tmp_path)

            result = replace_exact_once(workspace, "app.py", "print(1)", "print(2)")

            self.assertFalse(result.changed)
            self.assertEqual(result.old_count, 2)

    def test_replace_exact_once_updates_unique_match(self) -> None:
        with TemporaryWorkspace() as tmp_path:
            (tmp_path / "app.py").write_text("print(1)\n", encoding="utf-8")
            workspace = Workspace(tmp_path)

            result = replace_exact_once(workspace, "app.py", "print(1)", "print(2)")

            self.assertTrue(result.changed)
            self.assertEqual((tmp_path / "app.py").read_text(encoding="utf-8"), "print(2)\n")


class TemporaryWorkspace:
    def __enter__(self) -> Path:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        return Path(self._tmp.name)

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._tmp.cleanup()
