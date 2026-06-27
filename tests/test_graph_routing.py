from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage

from code_agent.graph import (
    _approval_interrupt_payload,
    _execute_node,
    _first_line,
    _normalize_approval_decisions,
    _tool_call_is_write,
    build_tools,
)
from code_agent.main import _is_slash_command, _parse_undo_args, _print_raw
from code_agent.services.workspace import Workspace
from code_agent.ui.approval import format_approval_summary, format_tool_call_summary
from code_agent.ui.stream import _tool_result_summary


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

    def test_first_line_returns_first_content_line(self) -> None:
        content = "# Code Agent\n\nlater content"

        self.assertEqual(_first_line(content), "# Code Agent")
        self.assertFalse(_first_line(content).startswith("later"))

    def test_write_detection_depends_on_tool_name_not_file_content(self) -> None:
        self.assertTrue(_tool_call_is_write({"name": "patch_file", "args": {"path": "src/app.py"}}))
        self.assertTrue(_tool_call_is_write({"name": "create_file", "args": {"path": "tests/test_app.py"}}))
        self.assertTrue(_tool_call_is_write({"name": "delete_file", "args": {"path": "old.py"}}))
        self.assertFalse(_tool_call_is_write({"name": "read_file", "args": {"path": "src/tools/fs.py"}}))
        self.assertFalse(_tool_call_is_write({"name": "run_shell", "args": {"command": "python -m pytest"}}))
        self.assertTrue(_tool_call_is_write({"name": "run_shell", "args": {"command": "npm install"}}))

    def test_approval_interrupt_payload_groups_all_action_requests(self) -> None:
        payload = _approval_interrupt_payload(
            [
                {"name": "tool_a", "tool_call_id": "call_1", "args": {}, "reason": "one"},
                {"name": "tool_b", "tool_call_id": "call_2", "args": {}, "reason": "two"},
            ]
        )

        self.assertEqual([request["tool_call_id"] for request in payload["action_requests"]], ["call_1", "call_2"])
        self.assertEqual(len(payload["review_configs"]), 2)
        self.assertIn("approve", payload["review_configs"][0]["allowed_decisions"])

    def test_resume_decisions_are_normalized_per_action_request(self) -> None:
        decisions = _normalize_approval_decisions(
            {"decisions": [{"type": "approve"}, {"type": "reject", "message": "no"}]},
            expected_count=2,
        )

        self.assertEqual(decisions, [{"type": "approve"}, {"type": "reject", "message": "no"}])

    def test_approval_summary_keeps_multiline_shell_command_to_one_line(self) -> None:
        summary = format_approval_summary(
            {
                "tool": "run_shell",
                "args": {
                    "command": 'python -c "\nprint(1)\nprint(2)\n"',
                },
            },
            "Unknown or medium-risk shell command requires approval.",
            max_length=80,
        )

        self.assertNotIn("\n", summary)
        self.assertIn("run_shell: python -c", summary)
        self.assertLessEqual(len(summary), 80)

    def test_tool_call_summary_hides_multiline_shell_body(self) -> None:
        summary = format_tool_call_summary(
            "run_shell",
            {
                "command": "node -e \"\nconst fs = require('fs');\nconst html = fs.readFileSync('index.html', 'utf8');\n\"",
                "timeout_seconds": 15,
            },
            max_length=70,
        )

        self.assertNotIn("\n", summary)
        self.assertIn("run_shell: node -e", summary)
        self.assertLessEqual(len(summary), 70)
        self.assertNotIn("readFileSync", summary)

    def test_shell_tool_result_summary_hides_command_body(self) -> None:
        summary = _tool_result_summary(
            "ALLOWED[level_1]: allowed low-risk shell command: node -e \"const fs = require('fs');\""
        )

        self.assertEqual(summary, "ALLOWED[level_1]: allowed shell command")
        self.assertNotIn("node -e", summary)

    def test_execute_node_is_independently_testable(self) -> None:
        tool = FakeTool("list_files", "ok")
        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "list_files",
                            "args": {"value": "hello"},
                            "id": "call_1",
                        }
                    ],
                )
            ]
        }

        with tempfile.TemporaryDirectory() as tmp:
            update = _execute_node(
                state,
                workspace=Workspace(tmp),
                tools_by_name={"list_files": tool},
                max_tool_calls_per_turn=2,
            )

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

        with tempfile.TemporaryDirectory() as tmp:
            update = _execute_node(state, workspace=Workspace(tmp), tools_by_name={}, max_tool_calls_per_turn=1)

        self.assertEqual(len(update["messages"]), 2)
        self.assertTrue(all(message.content.startswith("TOOL_LIMIT_EXCEEDED") for message in update["messages"]))

    def test_agent_toolset_does_not_expose_file_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tool_names = {tool.name for tool in build_tools(Workspace(tmp))}

        self.assertIn("list_files", tool_names)
        self.assertIn("find_files", tool_names)
        self.assertNotIn("get_file_tree", tool_names)

    def test_tool_args_are_pydantic_schemas_with_descriptions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = {tool.name: tool for tool in build_tools(Workspace(tmp))}

        read_schema = tools["read_file"].args_schema.model_json_schema()
        self.assertIn("path", read_schema["properties"])
        self.assertIn("description", read_schema["properties"]["path"])
        self.assertEqual(read_schema["properties"]["start_line"]["minimum"], 1)

        shell_schema = tools["run_shell"].args_schema.model_json_schema()
        self.assertIn("command", shell_schema["properties"])
        self.assertEqual(shell_schema["properties"]["timeout_seconds"]["maximum"], 180)
        internal_token_field = "approval" + "_token"
        self.assertNotIn(internal_token_field, shell_schema["properties"])

    def test_execute_node_interrupts_once_for_multiple_approval_decisions(self) -> None:
        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "custom_a", "args": {"value": "a"}, "id": "call_a"},
                        {"name": "custom_b", "args": {"value": "b"}, "id": "call_b"},
                        {"name": "custom_c", "args": {"value": "c"}, "id": "call_c"},
                    ],
                )
            ]
        }
        tools = {
            "custom_a": FakeTool("custom_a", "result a"),
            "custom_b": FakeTool("custom_b", "result b"),
            "custom_c": FakeTool("custom_c", "result c"),
        }

        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "code_agent.graph.interrupt",
                return_value={
                    "decisions": [
                        {"type": "approve"},
                        {"type": "reject", "message": "do not run b"},
                        {"type": "approve"},
                    ]
                },
            ) as interrupt_mock:
                update = _execute_node(
                    state,
                    workspace=Workspace(tmp),
                    tools_by_name=tools,
                    max_tool_calls_per_turn=3,
                )

        interrupt_mock.assert_called_once()
        payload = interrupt_mock.call_args.args[0]
        self.assertEqual([request["tool_call_id"] for request in payload["action_requests"]], ["call_a", "call_b", "call_c"])
        self.assertEqual([message.tool_call_id for message in update["messages"]], ["call_a", "call_b", "call_c"])
        self.assertEqual([message.content for message in update["messages"]], ["result a", "do not run b", "result c"])
        self.assertEqual(tools["custom_a"].calls, [{"value": "a"}])
        self.assertEqual(tools["custom_b"].calls, [])
        self.assertEqual(tools["custom_c"].calls, [{"value": "c"}])


class FakeTool:
    def __init__(self, name: str, response: str) -> None:
        self.name = name
        self.response = response
        self.calls = []

    def invoke(self, args):
        self.calls.append(args)
        return self.response
