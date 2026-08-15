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
from code_agent.config import AgentConfig
from code_agent.services.skills import SkillInfo
from code_agent.main import (
    _build_agent_undo_plan,
    _format_sessions,
    _format_trace_summary,
    _is_slash_command,
    _is_model_timeout_error,
    _parse_git_status_paths_z,
    _parse_undo_args,
    _print_raw,
    _resolve_chat_workspace,
)
from code_agent.services.persistence import SessionRecord
from code_agent.services.workspace import Workspace
from code_agent.ui.stream import (
    ModelWaitIndicator,
    _format_todos,
    _should_render_update,
    _tool_result_summary,
    chunk_begins_model_wait,
    render_stream_chunk,
)


class GraphRoutingTests(unittest.TestCase):
    def test_trace_summary_is_human_readable_and_bounded(self) -> None:
        trace = {
            "turn_count": 2,
            "turns": [
                {"turn_id": "turn_1", "user_request": "first", "tool_trace": [], "errors": []},
                {
                    "turn_id": "turn_2",
                    "user_request": "build the project",
                    "tool_trace": [{"tool": "shell_command"}],
                    "errors": [{"code": "nonzero_exit"}],
                },
            ],
        }

        rendered = _format_trace_summary(trace, limit=1)

        self.assertIn("Trace turns: 2", rendered)
        self.assertNotIn("turn_1", rendered)
        self.assertIn("turn_2: build the project [tools=1, errors=1]", rendered)

    def test_agent_defaults_include_model_timeout_and_five_tool_calls(self) -> None:
        config = AgentConfig()

        self.assertEqual(config.model_timeout_seconds, 120.0)
        self.assertEqual(config.max_tool_calls_per_turn, 5)
        self.assertEqual(config.max_total_tool_calls_per_run, 80)
        self.assertEqual(config.model_request_timeout_seconds, 45.0)
        self.assertEqual(config.model_max_retries, 2)

    def test_model_timeout_detection_handles_wrapped_client_errors(self) -> None:
        class APITimeoutError(RuntimeError):
            pass

        wrapper = RuntimeError("request failed")
        wrapper.__cause__ = APITimeoutError("upstream stalled")

        self.assertTrue(_is_model_timeout_error(wrapper))
        self.assertFalse(_is_model_timeout_error(RuntimeError("bad request")))

    def test_model_wait_indicator_tracks_before_model_lifecycle(self) -> None:
        self.assertTrue(
            chunk_begins_model_wait({"SummarizationMiddleware.before_model": {}})
        )
        self.assertFalse(chunk_begins_model_wait({"tools": {"messages": []}}))

        with patch("code_agent.ui.stream.Progress") as progress_factory:
            progress = progress_factory.return_value
            indicator = ModelWaitIndicator()
            indicator.start()
            indicator.start()
            indicator.stop()
            indicator.stop()

        progress_factory.assert_called_once()
        progress.add_task.assert_called_once_with("model-wait", total=None)
        progress.start.assert_called_once()
        progress.stop.assert_called_once()

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

    def test_git_status_z_paths_are_parsed_for_undo_baseline(self) -> None:
        output = " M src/app.py\0?? src/new file.tsx\0R  src/new-name.py\0src/old-name.py\0"

        self.assertEqual(
            _parse_git_status_paths_z(output),
            ["src/app.py", "src/new file.tsx", "src/new-name.py"],
        )

    def test_agent_undo_plan_skips_dirty_paths_without_snapshot(self) -> None:
        from code_agent.services.trace import append_turn_trace

        with tempfile.TemporaryDirectory() as tmp:
            append_turn_trace(
                tmp,
                {
                    "turn_id": "turn_001",
                    "thread_id": "thread_1",
                    "user_request": "edit files",
                    "git_baseline": {
                        "status": " M src/user.py",
                        "dirty_paths": ["src/user.py"],
                    },
                    "tool_trace": [],
                    "file_changes": [
                        {"path": "src/user.py", "operation": "write", "call_id": "call_1"},
                        {"path": "src/agent.py", "operation": "write", "call_id": "call_2"},
                        {"path": "src/new.py", "operation": "create", "call_id": "call_3"},
                    ],
                    "errors": [],
                },
            )

            plan = _build_agent_undo_plan(tmp, ["."])

            self.assertEqual(plan["restore_paths"], ["src/agent.py"])
            self.assertEqual(plan["clean_paths"], ["src/new.py"])
            self.assertEqual(plan["snapshot_restores"], [])
            self.assertEqual(plan["snapshot_deletes"], [])
            self.assertEqual(
                plan["skipped"],
                ["src/user.py was already modified before this agent turn and has no undo snapshot"],
            )

    def test_agent_undo_plan_restores_dirty_paths_with_snapshot(self) -> None:
        from code_agent.services.trace import append_turn_trace

        with tempfile.TemporaryDirectory() as tmp:
            snapshot = Path(tmp) / ".code-agent" / "undo" / "turn_001" / "snap.bin"
            snapshot.parent.mkdir(parents=True)
            snapshot.write_text("user version\n", encoding="utf-8")
            append_turn_trace(
                tmp,
                {
                    "turn_id": "turn_001",
                    "thread_id": "thread_1",
                    "user_request": "edit dirty file",
                    "git_baseline": {
                        "status": " M src/user.py",
                        "dirty_paths": ["src/user.py", "src/deleted.py"],
                    },
                    "undo_snapshots": [
                        {
                            "path": "src/user.py",
                            "snapshot_path": ".code-agent/undo/turn_001/snap.bin",
                            "existed": True,
                        },
                        {
                            "path": "src/deleted.py",
                            "snapshot_path": ".code-agent/undo/turn_001/deleted.bin",
                            "existed": False,
                        },
                    ],
                    "tool_trace": [],
                    "file_changes": [
                        {"path": "src/user.py", "operation": "write", "call_id": "call_1"},
                        {"path": "src/deleted.py", "operation": "create", "call_id": "call_2"},
                    ],
                    "errors": [],
                },
            )

            plan = _build_agent_undo_plan(tmp, ["."])

            self.assertEqual(plan["restore_paths"], [])
            self.assertEqual(plan["clean_paths"], [])
            self.assertEqual(
                plan["snapshot_restores"],
                [
                    {
                        "path": "src/user.py",
                        "snapshot_path": ".code-agent/undo/turn_001/snap.bin",
                        "existed": True,
                    }
                ],
            )
            self.assertEqual(plan["snapshot_deletes"], ["src/deleted.py"])
            self.assertEqual(plan["skipped"], [])

    def test_undo_snapshot_restore_writes_pre_agent_dirty_content(self) -> None:
        from code_agent.main import _restore_undo_snapshot

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            target = workspace / "src" / "user.py"
            target.parent.mkdir(parents=True)
            target.write_text("agent version\n", encoding="utf-8")
            snapshot = workspace / ".code-agent" / "undo" / "turn_001" / "snap.bin"
            snapshot.parent.mkdir(parents=True)
            snapshot.write_text("user version\n", encoding="utf-8")

            _restore_undo_snapshot(
                tmp,
                {
                    "path": "src/user.py",
                    "snapshot_path": ".code-agent/undo/turn_001/snap.bin",
                    "existed": True,
                },
            )

            self.assertEqual(target.read_text(encoding="utf-8"), "user version\n")

    def test_raw_print_disables_rich_markup_for_diff_content(self) -> None:
        with patch("code_agent.main.console.print") as print_mock:
            _print_raw("diff contains [/not-open]")

        print_mock.assert_called_once_with("diff contains [/not-open]", markup=False, highlight=False)

    def test_session_list_formats_full_title_on_own_line(self) -> None:
        long_title = "项目名称：" + "个人备忘录管理系统" * 20
        rendered = _format_sessions(
            [
                SessionRecord(
                    thread_id="thread_1",
                    title=long_title,
                    created_at="2026-06-29T00:00:00+00:00",
                    updated_at="2026-06-29T01:00:00+00:00",
                )
            ]
        )

        self.assertIn("1. thread_1", rendered)
        self.assertIn("updated_at: 2026-06-29T01:00:00+00:00", rendered)
        self.assertIn(f"title: {long_title}", rendered)

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

    def test_stream_hides_middleware_pass_through_messages(self) -> None:
        self.assertFalse(
            _should_render_update(
                "SummarizationMiddleware.before_model",
                {"messages": ["pass-through"]},
            )
        )

    def test_stream_renderer_never_prints_middleware_lifecycle_nodes(self) -> None:
        with patch("code_agent.ui.stream.console.print") as print_mock:
            render_stream_chunk(
                {
                    "HumanInTheLoopMiddleware.after_model": {
                        "messages": ["pass-through"]
                    },
                    "TodoListMiddleware.after_model": {
                        "todos": [{"content": "hidden lifecycle", "status": "pending"}]
                    },
                }
            )

        print_mock.assert_not_called()
        self.assertFalse(
            _should_render_update(
                "HumanInTheLoopMiddleware.after_model",
                {"messages": ["pass-through"]},
            )
        )

    def test_stream_renders_todo_updates(self) -> None:
        self.assertTrue(
            _should_render_update(
                "tools",
                {"todos": [{"content": "Inspect graph middleware", "status": "in_progress"}]},
            )
        )

    def test_format_todos_uses_stable_ascii_markers(self) -> None:
        lines = _format_todos(
            [
                {"content": "Inspect", "status": "completed"},
                {"content": "Patch", "status": "in_progress"},
                {"content": "Verify", "status": "pending"},
            ]
        )

        self.assertEqual(lines, ["[x] Inspect", "[>] Patch", "[ ] Verify"])

    def test_agent_toolset_exposes_windows_sandboxed_command_execution(self) -> None:
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
                "shell_command",
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
            self.assertFalse(
                config["shell_command"]["when"](
                    FakeToolCallRequest(
                        "shell_command",
                        {"command": "Get-ChildItem", "description": "List files"},
                    )
                )
            )
            # An escalation without a matching real denial is rejected by the
            # tool directly and must not show a misleading approval prompt.
            self.assertFalse(
                config["shell_command"]["when"](
                    FakeToolCallRequest(
                        "shell_command",
                        {
                            "command": "pytest",
                            "description": "Run tests",
                            "sandbox_permissions": "workspace-write",
                            "justification": "Tests create cache files.",
                        },
                    )
                )
            )
            shell_tool = next(tool for tool in tools if tool.name == "shell_command")
            object.__setattr__(shell_tool, "_approval_eligible", lambda args: True)
            self.assertTrue(
                config["shell_command"]["when"](
                    FakeToolCallRequest(
                        "shell_command",
                        {
                            "command": "pytest",
                            "description": "Run tests",
                            "sandbox_permissions": "workspace-write",
                            "justification": "Tests create cache files.",
                        },
                    )
                )
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
        self.assertIn("Command execution: use shell_command", content)
        self.assertIn("restricted-token sandbox", content)

    def test_runtime_metadata_injects_global_skill_index_and_loading_rule(self) -> None:
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

            class FakeSkillStore:
                def list_skills(self):
                    return [
                        SkillInfo(
                            name="vite-network-config",
                            description="Configure Vite network access.",
                            path="skills/vite-network-config/SKILL.md",
                        )
                    ]

            def handler(updated_request):
                return updated_request.system_message.content

            with patch("code_agent.graph.SkillStore", return_value=FakeSkillStore()):
                content = middleware.wrap_model_call(request, handler)

        self.assertIn("Available global skills:", content)
        self.assertIn("vite-network-config", content)
        self.assertIn("call skill_view for that skill before inspecting or editing", content)


class FakeToolCallRequest:
    def __init__(self, name: str, args: dict) -> None:
        self.tool_call = {"name": name, "args": args, "id": "call_1"}
    chunk_begins_model_wait,
