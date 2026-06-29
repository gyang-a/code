from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from langchain_core.messages import AIMessage, ToolMessage

from code_agent.services.trace import TurnTraceRecorder, write_turn_trace


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

    def test_write_turn_trace_uses_workspace_local_state_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace = {
                "turn_id": "turn_001",
                "thread_id": "thread_1",
                "tool_trace": [],
            }

            path = write_turn_trace(tmp, trace)

            self.assertTrue(path.is_file())
            self.assertEqual(path.parent, Path(tmp).resolve() / ".code-agent" / "traces")
            self.assertEqual(trace["trace_path"], str(path))
            self.assertTrue((Path(tmp) / ".code-agent" / ".gitignore").is_file())


if __name__ == "__main__":
    unittest.main()
