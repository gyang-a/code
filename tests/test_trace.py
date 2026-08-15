from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, ToolMessage

from code_agent.services.trace import (
    TurnTraceRecorder,
    append_turn_trace,
    load_project_trace,
    reviewable_project_trace,
    should_review_trace,
)


class TraceTests(unittest.TestCase):
    def test_turn_trace_records_tool_calls_results_and_file_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = TurnTraceRecorder(
                workspace=tmp,
                thread_id="thread_1",
                user_request="update app file",
            )

            recorder.record_message(
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "name": "write_file",
                            "args": {
                                "path": "src/app.py",
                                "content": "x" * 2000,
                                "api_key": "secret",
                            },
                        }
                    ],
                )
            )
            recorder.record_message(
                ToolMessage(
                    content="Wrote src/app.py\n\nDiff:\n+print('ok')",
                    name="write_file",
                    tool_call_id="call_1",
                )
            )

            trace = recorder.build_trace(final_answer="done", existing_skills=[])

            self.assertEqual(trace["tool_trace"][0]["tool"], "write_file")
            self.assertEqual(trace["tool_trace"][0]["status"], "success")
            self.assertEqual(trace["tool_trace"][0]["args"]["api_key"], "[REDACTED]")
            self.assertIn("truncated", trace["tool_trace"][0]["args"]["content"])
            self.assertEqual(
                trace["file_changes"],
                [{"path": "src/app.py", "operation": "write", "call_id": "call_1"}],
            )

    def test_turn_trace_snapshots_dirty_file_before_agent_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "src" / "app.py"
            path.parent.mkdir(parents=True)
            path.write_text("user version\n", encoding="utf-8")
            recorder = TurnTraceRecorder(
                workspace=tmp,
                thread_id="thread_1",
                user_request="update dirty app file",
                git_baseline_dirty_paths=["src/app.py"],
                turn_id="turn_001",
            )

            recorder.capture_write_snapshot("write_file", "src/app.py")
            path.write_text("agent version\n", encoding="utf-8")
            trace = recorder.build_trace(final_answer="done", existing_skills=[])

            self.assertEqual(len(trace["undo_snapshots"]), 1)
            snapshot = trace["undo_snapshots"][0]
            self.assertEqual(snapshot["path"], "src/app.py")
            snapshot_path = Path(tmp) / snapshot["snapshot_path"]
            self.assertEqual(snapshot_path.read_text(encoding="utf-8"), "user version\n")

    def test_shell_workspace_write_tracks_concrete_paths_for_safe_undo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tracked = root / "src" / "app.py"
            tracked.parent.mkdir(parents=True)
            tracked.write_text("baseline\n", encoding="utf-8")
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "add", "src/app.py"], cwd=root, check=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=Test",
                    "-c",
                    "user.email=test@example.com",
                    "commit",
                    "-qm",
                    "baseline",
                ],
                cwd=root,
                check=True,
            )
            tracked.write_text("user version\n", encoding="utf-8")
            recorder = TurnTraceRecorder(
                workspace=root,
                thread_id="thread_1",
                user_request="run formatter",
                git_baseline_dirty_paths=["src/app.py"],
                turn_id="turn_shell",
            )

            recorder.capture_workspace_write_snapshots("shell_command")
            tracked.write_text("shell version\n", encoding="utf-8")
            (root / "generated.txt").write_text("new\n", encoding="utf-8")
            recorder.record_shell_file_changes()
            trace = recorder.build_trace(final_answer="done", existing_skills=[])

            self.assertEqual(
                {(change["path"], change["operation"]) for change in trace["file_changes"]},
                {("src/app.py", "shell"), ("generated.txt", "create")},
            )
            self.assertNotIn(".", {change["path"] for change in trace["file_changes"]})
            snapshot = trace["undo_snapshots"][0]
            self.assertEqual(snapshot["path"], "src/app.py")
            self.assertEqual(
                (root / snapshot["snapshot_path"]).read_text(encoding="utf-8"),
                "user version\n",
            )

    def test_turn_trace_records_interrupt_action_requests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = TurnTraceRecorder(
                workspace=tmp,
                thread_id="thread_1",
                user_request="edit package",
            )

            recorder.record_interrupt(
                [
                    {
                        "value": {
                            "action_requests": [
                                {
                                    "id": "call_1",
                                    "name": "patch_file",
                                    "args": {"path": "package.json", "old": "a", "new": "b"},
                                }
                            ]
                        }
                    }
                ]
            )
            recorder.record_message(
                ToolMessage(
                    content="Patched package.json",
                    name="patch_file",
                    tool_call_id="call_1",
                )
            )

            trace = recorder.build_trace(final_answer="done", existing_skills=[])

            self.assertEqual(trace["tool_trace"][0]["tool"], "patch_file")
            self.assertEqual(trace["tool_trace"][0]["args"]["path"], "package.json")
            self.assertEqual(trace["tool_trace"][0]["status"], "success")

    def test_append_turn_trace_uses_one_workspace_project_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first_turn = {
                "turn_id": "turn_001",
                "thread_id": "thread_1",
                "user_request": "first",
                "tool_trace": [],
            }
            second_turn = {
                "turn_id": "turn_002",
                "thread_id": "thread_1",
                "user_request": "second",
                "tool_trace": [],
            }

            first_path, first_project_trace = append_turn_trace(tmp, first_turn)
            second_path, second_project_trace = append_turn_trace(tmp, second_turn)

            self.assertEqual(first_path, second_path)
            self.assertEqual(second_path, Path(tmp).resolve() / ".code-agent" / "traces.sqlite3")
            self.assertTrue(second_path.is_file())
            self.assertEqual(first_project_trace["turn_count"], 1)
            self.assertEqual(second_project_trace["turn_count"], 2)
            self.assertEqual([turn["turn_id"] for turn in second_project_trace["turns"]], ["turn_001", "turn_002"])
            self.assertEqual(load_project_trace(tmp)["turn_count"], 2)
            self.assertTrue((Path(tmp) / ".code-agent" / ".gitignore").is_file())

    def test_trace_store_rotates_full_turns_but_keeps_total_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch(
            "code_agent.services.trace.MAX_FULL_TRACE_TURNS",
            3,
        ):
            for index in range(5):
                append_turn_trace(
                    tmp,
                    {
                        "turn_id": f"turn_{index}",
                        "thread_id": "thread_1",
                        "user_request": f"request {index}",
                        "tool_trace": [],
                        "file_changes": [],
                        "errors": [],
                    },
                )

            trace = load_project_trace(tmp)

            self.assertEqual(trace["turn_count"], 5)
            self.assertEqual(trace["summary"]["turn_count"], 5)
            self.assertEqual(
                [turn["turn_id"] for turn in trace["turns"]],
                ["turn_2", "turn_3", "turn_4"],
            )
            review = reviewable_project_trace(trace, recent_turns_limit=2)
            self.assertEqual(review["turn_count"], 5)
            self.assertEqual(review["omitted_older_turns"], 3)

    def test_legacy_json_trace_is_imported_once_into_sqlite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            legacy = Path(tmp) / ".code-agent" / "traces" / "project_trace.json"
            legacy.parent.mkdir(parents=True)
            legacy.write_text(
                '{"schema_version":1,"turn_count":1,"turns":['
                '{"turn_id":"legacy_1","user_request":"old","tool_trace":[]}],'
                '"summary":{"turn_count":1}}',
                encoding="utf-8",
            )

            first = load_project_trace(tmp)
            second = load_project_trace(tmp)

            self.assertEqual(first["turn_count"], 1)
            self.assertEqual(second["turn_count"], 1)
            self.assertEqual(second["turns"][0]["turn_id"], "legacy_1")
            self.assertTrue((Path(tmp) / ".code-agent" / "traces.sqlite3").is_file())
            self.assertTrue(legacy.is_file())

    def test_project_trace_review_triggers_on_user_feedback(self) -> None:
        trace = {
            "turns": [
                {
                    "turn_id": "turn_001",
                    "user_request": "\u8fd9\u4e0d\u5bf9\uff0c\u4ee5\u540e\u4e0d\u8981\u8fd9\u4e48\u505a",
                    "tool_trace": [],
                    "file_changes": [],
                    "errors": [],
                }
            ]
        }

        self.assertTrue(should_review_trace(trace))

    def test_reviewable_project_trace_uses_summary_and_recent_turns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for index in range(12):
                append_turn_trace(
                    tmp,
                    {
                        "turn_id": f"turn_{index:03d}",
                        "thread_id": "thread_1",
                        "user_request": f"request {index}",
                        "tool_trace": [
                            {
                                "type": "tool_call",
                                "tool": "read_file",
                                "args": {"path": f"file_{index}.py"},
                                "status": "success",
                            }
                        ],
                        "file_changes": [],
                        "errors": [],
                    },
                )

            project_trace = load_project_trace(tmp)
            review_trace = reviewable_project_trace(project_trace, recent_turns_limit=3)

            self.assertEqual(review_trace["turn_count"], 12)
            self.assertEqual(review_trace["omitted_older_turns"], 9)
            self.assertEqual([turn["turn_id"] for turn in review_trace["recent_turns"]], ["turn_009", "turn_010", "turn_011"])
            self.assertNotIn("turns", review_trace)
            self.assertEqual(review_trace["historical_summary"]["turn_count"], 12)
            self.assertEqual(review_trace["historical_summary"]["tool_usage"]["read_file"], 12)

    def test_reviewable_project_trace_defaults_to_recent_10_turns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for index in range(12):
                append_turn_trace(
                    tmp,
                    {
                        "turn_id": f"turn_{index:03d}",
                        "thread_id": "thread_1",
                        "user_request": f"request {index}",
                        "tool_trace": [],
                        "file_changes": [],
                        "errors": [],
                    },
                )

            review_trace = reviewable_project_trace(load_project_trace(tmp))

            self.assertEqual(review_trace["omitted_older_turns"], 2)
            self.assertEqual(len(review_trace["recent_turns"]), 10)
            self.assertEqual(review_trace["recent_turns"][0]["turn_id"], "turn_002")


if __name__ == "__main__":
    unittest.main()
