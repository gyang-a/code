from __future__ import annotations

import unittest

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from code_agent.graph import (
    _generate_context_summary,
    _has_unanswered_tool_calls,
    _messages_for_agent,
    _messages_with_context_summary,
)
from code_agent.services.context import compact_messages, should_compact_messages
from code_agent.services.workspace import Workspace
from code_agent.ui.stream import _context_update_compacted


class ContextCompactionTests(unittest.TestCase):
    def test_should_compact_by_count_or_chars(self) -> None:
        messages = [HumanMessage(content=str(index)) for index in range(4)]

        self.assertTrue(should_compact_messages(messages, max_messages=3, max_chars=1000))
        self.assertTrue(should_compact_messages([HumanMessage(content="x" * 20)], max_messages=3, max_chars=10))
        self.assertFalse(should_compact_messages(messages[:2], max_messages=3, max_chars=1000))

    def test_compaction_removes_old_non_system_messages_and_keeps_recent(self) -> None:
        messages = [
            SystemMessage(content="system", id="system"),
            HumanMessage(content="old user", id="old-user"),
            AIMessage(content="", tool_calls=[{"name": "read_file", "args": {"path": "README.md"}, "id": "call_1"}], id="old-ai"),
            ToolMessage(content="old tool result", tool_call_id="call_1", id="old-tool"),
            HumanMessage(content="recent user", id="recent-user"),
        ]

        result = compact_messages(
            messages,
            existing_summary=None,
            changed_files=["src/app.py"],
            test_result="pytest passed",
            keep_recent=1,
        )

        self.assertTrue(result.compacted)
        self.assertEqual(result.removed_count, 3)
        self.assertIn("src/app.py", result.context_summary)
        self.assertIn("pytest passed", result.context_summary)
        self.assertIn("README.md", result.context_summary)

    def test_llm_summary_uses_isolated_prompt_material(self) -> None:
        llm = FakeSummaryLLM()

        summary = _generate_context_summary(llm, "old messages")

        self.assertEqual(summary, "compressed summary")
        self.assertEqual(len(llm.calls), 1)
        sent_messages = llm.calls[0]
        self.assertEqual([message.type for message in sent_messages], ["system", "human"])
        self.assertIn("old messages", sent_messages[1].content)

    def test_context_summary_is_injected_without_mutating_state_messages(self) -> None:
        original_messages = [
            SystemMessage(content="system", id="system"),
            AIMessage(content="", tool_calls=[{"name": "read_file", "args": {"path": "README.md"}, "id": "call_1"}], id="ai"),
        ]
        state = {
            "messages": original_messages,
            "context_summary": "compressed history",
        }

        llm_messages = _messages_with_context_summary(state)

        self.assertEqual(len(original_messages), 2)
        self.assertEqual(original_messages[-1].type, "ai")
        self.assertEqual([message.type for message in llm_messages], ["system", "system", "ai"])
        self.assertIn("compressed history", llm_messages[1].content)

    def test_agent_messages_include_runtime_metadata_without_mutating_history(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "README.md").write_text("# demo", encoding="utf-8")
            original_messages = [HumanMessage(content="hello")]
            state = {
                "messages": original_messages,
                "context_summary": None,
            }

            llm_messages = _messages_for_agent(state, Workspace(tmp))

        self.assertEqual([message.type for message in original_messages], ["human"])
        self.assertEqual(llm_messages[0].type, "system")
        self.assertEqual(llm_messages[1].type, "system")
        self.assertIn("Runtime metadata:", llm_messages[1].content)
        self.assertIn("README.md", llm_messages[1].content)

    def test_unanswered_tool_calls_block_compaction(self) -> None:
        messages = [
            HumanMessage(content="read"),
            AIMessage(content="", tool_calls=[{"name": "read_file", "args": {"path": "README.md"}, "id": "call_1"}]),
        ]

        self.assertTrue(_has_unanswered_tool_calls(messages))

    def test_context_stream_only_prints_real_compaction(self) -> None:
        self.assertFalse(_context_update_compacted({}))
        self.assertFalse(_context_update_compacted({"messages": []}))
        self.assertTrue(_context_update_compacted({"compaction_count": 1}))


class FakeSummaryLLM:
    def __init__(self) -> None:
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return AIMessage(content="compressed summary")
