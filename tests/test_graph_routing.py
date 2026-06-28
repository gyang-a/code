from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain.agents.middleware import ModelRequest
from langchain_core.messages import SystemMessage

from code_agent.graph import (
    RuntimeMetadataMiddleware,
    _approval_interrupt_config,
    build_tools,
)
from code_agent.main import _is_slash_command, _parse_undo_args, _print_raw, _resolve_chat_workspace
from code_agent.services.workspace import Workspace
from code_agent.ui.stream import _should_render_update, _tool_result_summary


class GraphRoutingTests(unittest.TestCase):
    def test_slash_commands_are_identified_before_graph_execution(self) -> None:
        self.assertTrue(_is_slash_command("/help"))
        self.assertTrue(_is_slash_command("  /diff"))
        self.assertFalse(_is_slash_command("explain /help"))

    def test_default_workspace_resolves_from_current_process_directory(self) -> None:
        import os

        with tempfile.TemporaryDirectory() as tmp:
            original = Path.cwd()
            try:
                os.chdir(tmp)
                self.assertEqual(_resolve_chat_workspace("."), str(Path(tmp).resolve()))
            finally:
                os.chdir(original)

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

    def test_tool_result_summary_returns_plain_tool_output(self) -> None:
        summary = _tool_result_summary("hello from command")

        self.assertEqual(summary, "hello from command")

    def test_tool_result_summary_keeps_multiple_useful_lines(self) -> None:
        summary = _tool_result_summary("one\ntwo\nthree\nfour\nfive\n")

        self.assertIn("one", summary)
        self.assertIn("four", summary)
        self.assertIn("...", summary)

    def test_stream_hides_empty_middleware_lifecycle_updates(self) -> None:
        self.assertFalse(_should_render_update("SummarizationMiddleware.before_model", {}))
        self.assertFalse(_should_render_update("ModelCallLimitMiddleware.after_model", {}))
        self.assertTrue(_should_render_update("model", {"messages": ["x"]}))

    def test_agent_toolset_does_not_expose_command_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = build_tools(Workspace(tmp))
            tool_names = {tool.name for tool in tools}

        self.assertEqual(
            tool_names,
            {
                "list_files",
                "read_file",
                "search_text",
                "find_files",
                "patch_file",
                "create_file",
                "write_file",
                "delete_file",
                "skills_list",
                "skill_view",
                "git_status",
                "git_diff",
            },
        )

    def test_tool_args_are_pydantic_schemas_with_descriptions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = {tool.name: tool for tool in build_tools(Workspace(tmp))}

        read_schema = tools["read_file"].args_schema.model_json_schema()
        self.assertIn("path", read_schema["properties"])
        self.assertIn("description", read_schema["properties"]["path"])
        self.assertEqual(read_schema["properties"]["start_line"]["minimum"], 1)

    def test_human_in_the_loop_config_uses_permission_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Workspace(tmp)
            tools = build_tools(workspace)
            config = _approval_interrupt_config(workspace, tools)

            self.assertFalse(
                config["read_file"]["when"](
                    FakeToolCallRequest("read_file", {"path": "README.md"})
                )
            )
            self.assertTrue(
                config["patch_file"]["when"](
                    FakeToolCallRequest("patch_file", {"path": "package.json"})
                )
            )
            self.assertIn(
                "package.json",
                config["patch_file"]["description"](
                    {"name": "patch_file", "args": {"path": "package.json"}},
                    {},
                    None,
                ),
            )

    def test_runtime_metadata_middleware_injects_system_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            middleware = RuntimeMetadataMiddleware(
                workspace=Workspace(tmp),
                max_tool_calls_per_turn=3,
            )
            request = ModelRequest(
                model=object(),
                messages=[],
                system_message=SystemMessage(content="base"),
            )

            def handler(updated_request):
                return updated_request.system_message.content

            content = middleware.wrap_model_call(request, handler)

        self.assertIn("base", content)
        self.assertIn("Runtime metadata:", content)
        self.assertIn("Command execution: unavailable to the agent.", content)


class FakeToolCallRequest:
    def __init__(self, name: str, args: dict) -> None:
        self.tool_call = {"name": name, "args": args, "id": "call_1"}
