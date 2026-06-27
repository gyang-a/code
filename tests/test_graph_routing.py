from __future__ import annotations

import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage

from code_agent.graph import (
    build_tools,
    _execute_node,
    _first_line,
    _tool_message_for_pending,
    _tool_call_is_write,
)
from code_agent.main import _is_slash_command, _parse_undo_args, _print_raw
from code_agent.services.workspace import Workspace
from code_agent.ui.approval import format_approval_summary


class GraphRoutingTests(unittest.TestCase):
    def test_slash_commands_are_identified_before_graph_execution(self) -> None:
        self.assertTrue(_is_slash_command("/help"))
        self.assertTrue(_is_slash_command("  /diff"))
        self.assertFalse(_is_slash_command("explain /help"))

    def test_undo_args_default_to_tracked_restore_only(self) -> None:
        include_untracked, targets = _parse_undo_args("")

        self.assertFalse(include_untracked)
        self.assertEqual(targets, ["."])

    def test_undo_args_support_include_untracked_and_path(self) -> None:
        include_untracked, targets = _parse_undo_args('--include-untracked "src/app.py" src\\win.py')

        self.assertTrue(include_untracked)
        self.assertEqual(targets, ["src/app.py", "src\\win.py"])

    def test_raw_print_disables_rich_markup_for_diff_content(self) -> None:
        with patch("code_agent.main.console.print") as print_mock:
            _print_raw("diff contains [/not-open]")

        print_mock.assert_called_once_with("diff contains [/not-open]", markup=False, highlight=False)

    def test_approval_marker_must_be_first_line(self) -> None:
        content = "# Code Agent\n\nAPPROVAL_REQUIRED[level_2]: docs mention this token"

        self.assertEqual(_first_line(content), "# Code Agent")
        self.assertFalse(_first_line(content).startswith("APPROVAL_REQUIRED[level_2]"))

    def test_write_detection_depends_on_tool_name_not_file_content(self) -> None:
        self.assertTrue(_tool_call_is_write({"name": "patch_file", "args": {"path": "src/app.py"}}))
        self.assertTrue(_tool_call_is_write({"name": "create_file", "args": {"path": "tests/test_app.py"}}))
        self.assertTrue(_tool_call_is_write({"name": "delete_file", "args": {"path": "old.py"}}))
        self.assertFalse(_tool_call_is_write({"name": "read_file", "args": {"path": "src/tools/fs.py"}}))
        self.assertFalse(_tool_call_is_write({"name": "run_shell", "args": {"command": "python -m pytest"}}))
        self.assertTrue(_tool_call_is_write({"name": "run_shell", "args": {"command": "npm install"}}))

    def test_approval_followup_is_tool_message(self) -> None:
        message = _tool_message_for_pending(
            {"tool_call_id": "call_1", "tool_message_id": "tool-msg-1"},
            content="APPROVED[level_2]: ok",
        )

        self.assertEqual(message.type, "tool")
        self.assertEqual(message.tool_call_id, "call_1")
        self.assertEqual(message.id, "tool-msg-1")

    def test_approval_summary_keeps_multiline_shell_command_to_one_line(self) -> None:
        summary = format_approval_summary(
            {
                "tool": "run_shell",
                "args": {
                    "command": 'python -c "\nprint(1)\nprint(2)\n"',
                },
            },
            "APPROVAL_REQUIRED[level_2]: 未知或中风险 shell 命令需要确认",
            max_length=80,
        )

        self.assertNotIn("\n", summary)
        self.assertIn("run_shell: python -c", summary)
        self.assertLessEqual(len(summary), 80)

    def test_execute_node_is_independently_testable(self) -> None:
        tool = FakeTool("echo_tool", "ok")
        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "echo_tool",
                            "args": {"value": "hello"},
                            "id": "call_1",
                        }
                    ],
                )
            ]
        }

        update = _execute_node(state, tools_by_name={"echo_tool": tool}, max_tool_calls_per_turn=2)

        self.assertEqual(tool.calls, [{"value": "hello"}])
        self.assertEqual(update["messages"][0].type, "tool")
        self.assertEqual(update["messages"][0].content, "ok")
        self.assertEqual(update["messages"][0].tool_call_id, "call_1")

    def test_execute_node_enforces_tool_call_limit(self) -> None:
        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "one", "args": {}, "id": "call_1"},
                        {"name": "two", "args": {}, "id": "call_2"},
                    ],
                )
            ]
        }

        update = _execute_node(state, tools_by_name={}, max_tool_calls_per_turn=1)

        self.assertEqual(len(update["messages"]), 2)
        self.assertTrue(all(message.content.startswith("TOOL_LIMIT_EXCEEDED") for message in update["messages"]))

    def test_agent_toolset_does_not_expose_file_tree(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tool_names = {tool.name for tool in build_tools(Workspace(tmp))}

        self.assertIn("list_files", tool_names)
        self.assertIn("find_files", tool_names)
        self.assertNotIn("get_file_tree", tool_names)


class FakeTool:
    def __init__(self, name: str, response: str) -> None:
        self.name = name
        self.response = response
        self.calls = []

    def invoke(self, args):
        self.calls.append(args)
        return self.response
