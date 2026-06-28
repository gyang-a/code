from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, ToolMessage

from code_agent.services.skill_review import count_tool_results, should_review_skills
from code_agent.services.skills import SkillStore, format_pending_summary, format_skill_index
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

    def test_skill_review_threshold_counts_tool_results(self) -> None:
        messages = [
            ToolMessage(content="ok", tool_call_id=f"call_{index}")
            for index in range(5)
        ]

        self.assertEqual(count_tool_results(messages), 5)
        self.assertTrue(should_review_skills(messages, reviewed_tool_count=0))
        self.assertFalse(should_review_skills(messages, reviewed_tool_count=4, threshold=5))

    def test_skill_review_triggers_on_write_tool_call(self) -> None:
        messages = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "patch_file",
                        "args": {"path": "src/app.py", "old": "a", "new": "b"},
                        "id": "call_1",
                    }
                ],
            )
        ]

        self.assertTrue(should_review_skills(messages, reviewed_tool_count=0, threshold=5))


if __name__ == "__main__":
    unittest.main()
