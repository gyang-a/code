from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch

from langchain.agents.middleware import ModelRequest
from langchain_core.messages import SystemMessage

from code_agent.graph import (
    RuntimeMetadataMiddleware,
    _approval_interrupt_config,
    build_tools,
)
from code_agent.main import _is_slash_command, _parse_undo_args, _print_raw
from code_agent.services.workspace import Workspace
from code_agent.tools.shell import build_run_command_tool
from code_agent.ui.approval import format_approval_summary, format_tool_call_summary
from code_agent.ui.stream import _should_render_update, _tool_result_summary


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

    def test_approval_summary_accepts_official_action_request_shape(self) -> None:
        summary = format_approval_summary(
            {
                "name": "run_shell",
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

    def test_tool_result_summary_returns_plain_tool_output(self) -> None:
        summary = _tool_result_summary("hello from command")

        self.assertEqual(summary, "hello from command")

    def test_stream_hides_empty_middleware_lifecycle_updates(self) -> None:
        self.assertFalse(_should_render_update("SummarizationMiddleware.before_model", {}))
        self.assertFalse(_should_render_update("ModelCallLimitMiddleware.after_model", {}))
        self.assertTrue(_should_render_update("model", {"messages": ["x"]}))

    def test_run_shell_tool_returns_command_output_without_permission_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_shell = build_run_command_tool(Workspace(tmp))

            result = run_shell.invoke({"command": "python -m pytest --version"})

        self.assertIn("pytest", result)
        self.assertNotIn("ALLOWED[", result)
        self.assertNotIn("requires approval", result.lower())

    def test_run_shell_tool_rejects_level_2_without_hitl_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_shell = build_run_command_tool(Workspace(tmp))

            result = run_shell.invoke({"command": "npm create vite@latest app"})

        self.assertIn("REJECTED[level_2]", result)

    def test_run_shell_tool_allows_level_2_after_hitl_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch(
            "code_agent.tools.shell.build_shell_sandbox"
        ) as sandbox_factory:
            sandbox = sandbox_factory.return_value
            sandbox.run_shell.return_value.returncode = 0
            sandbox.run_shell.return_value.stdout = "created\n"
            sandbox.run_shell.return_value.stderr = ""
            run_shell = build_run_command_tool(
                Workspace(tmp),
                allow_requires_approval=True,
            )

            result = run_shell.invoke({"command": "npm create vite@latest app"})

        self.assertEqual(result, "created")
        sandbox.run_shell.assert_called_once()

    def test_run_shell_tool_rejects_dev_server_even_after_hitl_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch(
            "code_agent.tools.shell.build_shell_sandbox"
        ) as sandbox_factory:
            sandbox = sandbox_factory.return_value
            run_shell = build_run_command_tool(
                Workspace(tmp),
                allow_requires_approval=True,
            )

            result = run_shell.invoke({"command": "npm run dev"})

        self.assertIn("REJECTED[level_3]", result)
        self.assertIn("dev server", result)
        sandbox.run_shell.assert_not_called()

    def test_agent_toolset_does_not_expose_file_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = build_tools(Workspace(tmp))
            tool_names = {tool.name for tool in tools}

        self.assertNotIn(None, tools)
        self.assertIn("list_files", tool_names)
        self.assertIn("find_files", tool_names)
        self.assertIn("git_diff", tool_names)
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
                shell_sandbox="local",
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
        self.assertIn("Shell sandbox: local.", content)


class FakeToolCallRequest:
    def __init__(self, name: str, args: dict) -> None:
        self.tool_call = {"name": name, "args": args, "id": "call_1"}
