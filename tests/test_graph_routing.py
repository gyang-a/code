from __future__ import annotations

import unittest

from langchain_core.messages import AIMessage

from code_agent.graph import (
    build_tools,
    _execute_node,
    _first_line,
    _tool_message_for_pending,
    _tool_call_is_write,
)
from code_agent.main import _is_slash_command
from code_agent.services.workspace import Workspace


class GraphRoutingTests(unittest.TestCase):
    def test_slash_commands_are_identified_before_graph_execution(self) -> None:
        self.assertTrue(_is_slash_command("/help"))
        self.assertTrue(_is_slash_command("  /diff"))
        self.assertFalse(_is_slash_command("explain /help"))

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
