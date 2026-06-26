from __future__ import annotations

from pathlib import Path
import unittest

from code_agent.services.memory import format_project_memory, load_project_memory
from code_agent.services.workspace import Workspace


class MemoryTests(unittest.TestCase):
    def test_loads_project_memory_as_context(self) -> None:
        with TemporaryWorkspace() as tmp_path:
            (tmp_path / "CLAUDE.md").write_text("项目约定：先读文件再编辑。", encoding="utf-8")
            workspace = Workspace(tmp_path)

            entries = load_project_memory(workspace)
            rendered = format_project_memory(entries)

            self.assertEqual(len(entries), 1)
            self.assertIn("CLAUDE.md", rendered)
            self.assertIn("先读文件再编辑", rendered)


class TemporaryWorkspace:
    def __enter__(self) -> Path:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        return Path(self._tmp.name)

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._tmp.cleanup()
