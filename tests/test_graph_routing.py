from __future__ import annotations

import unittest

from code_agent.graph import (
    _first_line,
    _normalize_intent_label,
    _parse_plan_lines,
    _should_validate,
    _tool_call_is_write,
)


class GraphRoutingTests(unittest.TestCase):
    def test_intent_label_normalization(self) -> None:
        self.assertEqual(_normalize_intent_label("casual_chat"), "casual_chat")
        self.assertEqual(_normalize_intent_label("general_question\n"), "general_question")
        self.assertEqual(_normalize_intent_label("`code_task`"), "code_task")
        self.assertEqual(_normalize_intent_label("something weird"), "code_task")

    def test_approval_marker_must_be_first_line(self) -> None:
        content = "# Code Agent\n\nAPPROVAL_REQUIRED[level_2]: docs mention this token"

        self.assertEqual(_first_line(content), "# Code Agent")
        self.assertFalse(_first_line(content).startswith("APPROVAL_REQUIRED[level_2]"))

    def test_parse_plan_lines(self) -> None:
        raw_plan = "1. 读取 README 了解项目\n- 搜索权限相关代码\n3）总结实现方式"

        self.assertEqual(
            _parse_plan_lines(raw_plan),
            ["读取 README 了解项目", "搜索权限相关代码", "总结实现方式"],
        )

    def test_validation_depends_on_current_turn_writes(self) -> None:
        self.assertFalse(_should_validate({"did_write": False, "changed_files": []}))
        self.assertTrue(_should_validate({"did_write": True, "changed_files": []}))
        self.assertTrue(_should_validate({"did_write": False, "changed_files": ["src/app.py"]}))

    def test_write_detection_depends_on_tool_name_not_file_content(self) -> None:
        self.assertTrue(_tool_call_is_write({"name": "patch_file", "args": {"path": "src/app.py"}}))
        self.assertTrue(_tool_call_is_write({"name": "create_file", "args": {"path": "tests/test_app.py"}}))
        self.assertTrue(_tool_call_is_write({"name": "delete_file", "args": {"path": "old.py"}}))
        self.assertFalse(_tool_call_is_write({"name": "read_file", "args": {"path": "src/tools/fs.py"}}))
        self.assertFalse(_tool_call_is_write({"name": "run_shell", "args": {"command": "python -m pytest"}}))
        self.assertTrue(_tool_call_is_write({"name": "run_shell", "args": {"command": "npm install"}}))
