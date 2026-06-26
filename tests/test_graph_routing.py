from __future__ import annotations

import unittest

from code_agent.graph import _first_line, _normalize_intent_label, _parse_plan_lines


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
