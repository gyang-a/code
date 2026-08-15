from __future__ import annotations

from pathlib import Path
import unittest

from code_agent.services.memory import format_project_memory, load_project_memory
from code_agent.services.metadata import build_turn_metadata
from code_agent.services.workspace import Workspace


class MemoryTests(unittest.TestCase):
    def test_loads_project_memory_as_context(self) -> None:
        with TemporaryWorkspace() as tmp_path:
            (tmp_path / "AGENTS.md").write_text(
                "Project rule: read files before editing.",
                encoding="utf-8",
            )
            workspace = Workspace(tmp_path)

            entries = load_project_memory(workspace)
            rendered = format_project_memory(entries)

            self.assertEqual(len(entries), 1)
            self.assertIn("AGENTS.md", rendered)
            self.assertIn("read files before editing", rendered)

    def test_does_not_load_claude_memory_file(self) -> None:
        with TemporaryWorkspace() as tmp_path:
            (tmp_path / "CLAUDE.md").write_text("legacy claude memory", encoding="utf-8")
            workspace = Workspace(tmp_path)

            entries = load_project_memory(workspace)

            self.assertEqual(entries, [])

    def test_turn_metadata_is_bounded_and_not_a_full_tree(self) -> None:
        with TemporaryWorkspace() as tmp_path:
            (tmp_path / "src").mkdir()
            (tmp_path / "README.md").write_text("# demo", encoding="utf-8")
            workspace = Workspace(tmp_path)

            metadata = build_turn_metadata(workspace, max_entries=1)

            self.assertIn("current_folder_name:", metadata)
            self.assertIn("top_level_visible_entries:", metadata)
            self.assertIn("Use shell_command for Windows PowerShell commands", metadata)
            self.assertIn("direct danger-full-access for package managers", metadata)
            self.assertIn("Broad triage budget", metadata)
            self.assertIn("at most 6 file reads or 12 total tool calls", metadata)
            self.assertIn("... truncated ...", metadata)
            self.assertNotIn("/workspace", metadata)
            self.assertNotIn(str(tmp_path), metadata)


class TemporaryWorkspace:
    def __enter__(self) -> Path:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        return Path(self._tmp.name)

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._tmp.cleanup()
