from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import os

from code_agent.services.skill_review import _normalize_skill_markdown, _target_skill_name
from code_agent.services.skills import SkillStore, default_pending_skill_root, default_skill_root, format_pending_summary, format_skill_index
from code_agent.services.workspace import Workspace
from code_agent.tools import build_skill_view_tool, build_skills_list_tool


SKILL_MD = """---
name: python-testing
description: Use when writing or running Python tests with pytest.
---

# Python Testing

Run focused pytest commands before broad suites.
"""


class SkillTests(unittest.TestCase):
    def test_global_skill_store_lists_and_reads_skills(self) -> None:
        with tempfile.TemporaryDirectory() as skills_dir, tempfile.TemporaryDirectory() as pending_dir:
            skill_path = Path(skills_dir) / "python-testing" / "SKILL.md"
            skill_path.parent.mkdir(parents=True)
            skill_path.write_text(SKILL_MD, encoding="utf-8")
            store = SkillStore(skills_dir=skills_dir, pending_dir=pending_dir)

            rendered = format_skill_index(store.list_skills())

            self.assertIn("python-testing", rendered)
            self.assertIn("pytest", store.read_skill("python-testing"))

    def test_pending_skill_change_can_be_approved(self) -> None:
        with tempfile.TemporaryDirectory() as skills_dir, tempfile.TemporaryDirectory() as pending_dir:
            store = SkillStore(skills_dir=skills_dir, pending_dir=pending_dir)

            change = store.stage_skill_change(
                skill_name="python-testing",
                content=SKILL_MD,
                reason="Reusable test workflow.",
            )

            self.assertIn(change.id, format_pending_summary(store.list_pending()))
            self.assertIn("Python Testing", store.pending_diff(change.id))
            approved = store.approve_pending(change.id)

            self.assertEqual(approved.skill_name, "python-testing")
            self.assertTrue((Path(skills_dir) / "python-testing" / "SKILL.md").is_file())
            self.assertEqual(store.list_pending(), [])

    def test_skill_change_can_be_written_directly_after_review_approval(self) -> None:
        with tempfile.TemporaryDirectory() as skills_dir, tempfile.TemporaryDirectory() as pending_dir:
            store = SkillStore(skills_dir=skills_dir, pending_dir=pending_dir)

            diff = store.skill_diff("python-testing", SKILL_MD, tofile="proposed:python-testing")
            path = store.write_skill("python-testing", SKILL_MD)

            self.assertIn("proposed:python-testing", diff)
            self.assertEqual(path, Path(skills_dir).resolve() / "python-testing" / "SKILL.md")
            self.assertEqual(path.read_text(encoding="utf-8"), SKILL_MD)
            self.assertEqual(store.list_pending(), [])

    def test_default_skill_roots_use_code_agent_home(self) -> None:
        with tempfile.TemporaryDirectory() as state_dir:
            with patch.dict(os.environ, {"CODE_AGENT_HOME": state_dir}, clear=False):
                self.assertEqual(default_skill_root(), Path(state_dir).resolve() / "skills")
                self.assertEqual(default_pending_skill_root(), Path(state_dir).resolve() / "pending" / "skills")

    def test_legacy_pending_can_be_approved_into_default_skill_root(self) -> None:
        with tempfile.TemporaryDirectory() as state_dir, tempfile.TemporaryDirectory() as legacy_skills, tempfile.TemporaryDirectory() as legacy_pending:
            legacy_store = SkillStore(skills_dir=legacy_skills, pending_dir=legacy_pending)
            change = legacy_store.stage_skill_change(
                skill_name="python-testing",
                content=SKILL_MD,
                reason="Legacy pending proposal.",
            )

            with patch.dict(os.environ, {"CODE_AGENT_HOME": state_dir}, clear=False), patch(
                "code_agent.services.skills.legacy_pending_skill_root",
                return_value=Path(legacy_pending).resolve(),
            ):
                store = SkillStore()
                self.assertIn(change.id, format_pending_summary(store.list_pending()))
                approved = store.approve_pending(change.id)

                self.assertEqual(approved.id, change.id)
                self.assertTrue((Path(state_dir) / "skills" / "python-testing" / "SKILL.md").is_file())
                self.assertFalse((Path(legacy_pending) / f"{change.id}.json").exists())

    def test_skill_tools_use_global_store(self) -> None:
        with tempfile.TemporaryDirectory() as workspace_dir, tempfile.TemporaryDirectory() as skills_dir, tempfile.TemporaryDirectory() as pending_dir:
            skill_path = Path(skills_dir) / "python-testing" / "SKILL.md"
            skill_path.parent.mkdir(parents=True)
            skill_path.write_text(SKILL_MD, encoding="utf-8")
            workspace = Workspace(workspace_dir)

            with patch("code_agent.services.skills.default_skill_root", return_value=Path(skills_dir)), patch(
                "code_agent.services.skills.default_pending_skill_root", return_value=Path(pending_dir)
            ):
                list_tool = build_skills_list_tool(workspace)
                view_tool = build_skill_view_tool(workspace)

            self.assertIn("python-testing", list_tool.invoke({}))
            self.assertIn("Python Testing", view_tool.invoke({"name": "python-testing"}))

    def test_skill_review_normalizes_missing_skill_newline(self) -> None:
        content = "---\nname: demo\ndescription: demo skill\n---\n\n# Demo"

        self.assertTrue(_normalize_skill_markdown(content).endswith("\n"))

    def test_skill_review_normalizes_target_skill_filename(self) -> None:
        self.assertEqual(_target_skill_name("frontend-viteconfig.md"), "frontend-viteconfig")
        self.assertEqual(_target_skill_name("skills/Frontend ViteConfig.md"), "frontend-viteconfig")

    def test_skill_review_rewrites_invalid_frontmatter_name(self) -> None:
        content = (
            "---\n"
            "name: Frontend ViteConfig.md\n"
            "description: Vite config workflow.\n"
            "---\n\n"
            "# Frontend ViteConfig\n"
        )

        normalized = _normalize_skill_markdown(content, skill_name="frontend-viteconfig")

        self.assertIn("name: frontend-viteconfig\n", normalized)
        self.assertNotIn("Frontend ViteConfig.md", normalized)


if __name__ == "__main__":
    unittest.main()
