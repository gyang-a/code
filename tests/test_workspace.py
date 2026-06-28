from __future__ import annotations

from pathlib import Path
import unittest

from code_agent.services.workspace import Workspace, WorkspaceError
from code_agent.tools.fs import build_create_file_tool, build_git_status_tool, build_read_file_tool


class WorkspaceTests(unittest.TestCase):
    def test_resolve_rejects_path_escape(self) -> None:
        with TemporaryWorkspace() as tmp_path:
            workspace = Workspace(tmp_path)

            with self.assertRaises(WorkspaceError) as raised:
                workspace.resolve("../outside.txt")

            self.assertIn("Path escapes current project folder", str(raised.exception))
            self.assertIn(str(tmp_path.resolve()), str(raised.exception))

    def test_resolve_accepts_absolute_path_inside_workspace(self) -> None:
        with TemporaryWorkspace() as tmp_path:
            workspace = Workspace(tmp_path)
            nested = tmp_path / "src" / "app.py"

            resolved = workspace.resolve(nested)

            self.assertEqual(resolved, nested.resolve())

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

    def test_read_text_window_can_read_large_file_by_lines(self) -> None:
        with TemporaryWorkspace() as tmp_path:
            (tmp_path / "large.txt").write_text("\n".join(f"line {index}" for index in range(1, 6)), encoding="utf-8")
            workspace = Workspace(tmp_path, read_limit=10, read_max_lines=2)

            content = workspace.read_text_window("large.txt")

            self.assertIn("LINES: 1-2", content)
            self.assertIn("1: line 1", content)
            self.assertIn("2: line 2", content)
            self.assertIn("start_line=3", content)

    def test_read_text_window_supports_start_line(self) -> None:
        with TemporaryWorkspace() as tmp_path:
            (tmp_path / "app.py").write_text("a\nb\nc\n", encoding="utf-8")
            workspace = Workspace(tmp_path, read_max_lines=2)

            content = workspace.read_text_window("app.py", start_line=2)

            self.assertIn("LINES: 2-3", content)
            self.assertIn("2: b", content)
            self.assertIn("3: c", content)

    def test_read_file_tool_limits_lines_and_chars(self) -> None:
        with TemporaryWorkspace() as tmp_path:
            (tmp_path / "data.txt").write_text("\n".join(["x" * 50, "y" * 50, "z" * 50]), encoding="utf-8")
            workspace = Workspace(tmp_path)
            read_file = build_read_file_tool(workspace, default_max_lines=3, output_limit=80)

            content = read_file.invoke({"path": "data.txt"})

            self.assertIn("FILE: data.txt", content)
            self.assertIn("truncated", content)

    def test_create_file_tool_does_not_append_git_fatal_outside_git_repo(self) -> None:
        with TemporaryWorkspace() as tmp_path:
            workspace = Workspace(tmp_path)
            create_file = build_create_file_tool(workspace)

            content = create_file.invoke({"path": "src/app.py", "content": "print('ok')\n"})

            self.assertEqual(content, "Created src/app.py")
            self.assertTrue((tmp_path / "src" / "app.py").exists())

    def test_git_status_reports_non_git_repository_cleanly(self) -> None:
        with TemporaryWorkspace() as tmp_path:
            workspace = Workspace(tmp_path)
            git_status = build_git_status_tool(workspace)

            content = git_status.invoke({})

            self.assertEqual(content, "NOT_GIT_REPOSITORY")

    def test_list_files_reports_empty_directory_clearly(self) -> None:
        from code_agent.tools.fs import build_list_files_tool

        with TemporaryWorkspace() as tmp_path:
            (tmp_path / "empty").mkdir()
            workspace = Workspace(tmp_path)
            list_files = build_list_files_tool(workspace)

            content = list_files.invoke({"path": "empty"})

            self.assertEqual(content, "EMPTY: empty has no visible entries.")

    def test_list_files_reports_truncation(self) -> None:
        from code_agent.tools.fs import build_list_files_tool

        with TemporaryWorkspace() as tmp_path:
            for index in range(3):
                (tmp_path / f"file_{index}.txt").write_text("x", encoding="utf-8")
            workspace = Workspace(tmp_path)
            list_files = build_list_files_tool(workspace)

            content = list_files.invoke({"path": ".", "max_entries": 2})

            self.assertIn("... truncated after 2 visible entries ...", content)


class TemporaryWorkspace:
    def __enter__(self) -> Path:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        return Path(self._tmp.name)

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._tmp.cleanup()
