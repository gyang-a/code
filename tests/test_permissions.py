from __future__ import annotations

from pathlib import Path
import unittest

from code_agent.models import RiskLevel
from code_agent.services.workspace import Workspace
from code_agent.tools.safety import classify_file_operation, classify_tool_call


class PermissionTests(unittest.TestCase):
    def test_package_json_patch_requires_approval(self) -> None:
        with TemporaryWorkspace() as tmp_path:
            (tmp_path / "package.json").write_text("{}", encoding="utf-8")
            workspace = Workspace(tmp_path)

            decision = classify_file_operation(workspace, "patch_file", "package.json")

            self.assertEqual(decision.risk, RiskLevel.level_2)
            self.assertTrue(decision.requires_approval)

    def test_env_access_is_forbidden(self) -> None:
        with TemporaryWorkspace() as tmp_path:
            (tmp_path / ".env").write_text("SECRET=1", encoding="utf-8")
            workspace = Workspace(tmp_path)

            decision = classify_file_operation(workspace, "patch_file", ".env")

            self.assertEqual(decision.risk, RiskLevel.level_3)
            self.assertFalse(decision.allowed)

    def test_create_file_in_src_is_low_risk(self) -> None:
        with TemporaryWorkspace() as tmp_path:
            (tmp_path / "src").mkdir()
            workspace = Workspace(tmp_path)

            decision = classify_file_operation(workspace, "create_file", "src/app.py")

            self.assertEqual(decision.risk, RiskLevel.level_1)
            self.assertTrue(decision.allowed)

    def test_delete_requires_approval(self) -> None:
        with TemporaryWorkspace() as tmp_path:
            (tmp_path / "src").mkdir()
            (tmp_path / "src" / "app.py").write_text("print('x')", encoding="utf-8")
            workspace = Workspace(tmp_path)

            decision = classify_file_operation(workspace, "delete_file", "src/app.py")

            self.assertEqual(decision.risk, RiskLevel.level_2)
            self.assertTrue(decision.requires_approval)

    def test_read_tool_is_level_0(self) -> None:
        with TemporaryWorkspace() as tmp_path:
            (tmp_path / "README.md").write_text("# demo", encoding="utf-8")
            workspace = Workspace(tmp_path)

            decision = classify_tool_call(workspace, "read_file", {"path": "README.md"})

            self.assertEqual(decision.risk, RiskLevel.level_0)
            self.assertTrue(decision.allowed)

    def test_unknown_tool_requires_approval(self) -> None:
        with TemporaryWorkspace() as tmp_path:
            workspace = Workspace(tmp_path)

            decision = classify_tool_call(workspace, "mystery_tool", {})

            self.assertEqual(decision.risk, RiskLevel.level_2)
            self.assertTrue(decision.requires_approval)


class TemporaryWorkspace:
    def __enter__(self) -> Path:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        return Path(self._tmp.name)

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
