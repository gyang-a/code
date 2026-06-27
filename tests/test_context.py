from __future__ import annotations

import unittest

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage

from code_agent.graph import (
    _generate_context_summary,
    _has_unanswered_tool_calls,
    _messages_for_agent,
    _messages_for_llm_with_context_summary,
)
from code_agent.prompts import SYSTEM_PROMPT
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

    def test_compaction_returns_removals_and_string_summary_not_ai_summary_message(self) -> None:
        messages = [
            HumanMessage(content="old user", id="old-user"),
            AIMessage(content="old answer", id="old-ai"),
            HumanMessage(content="recent user", id="recent-user"),
        ]

        result = compact_messages(
            messages,
            existing_summary=None,
            changed_files=[],
            test_result=None,
            keep_recent=1,
        )

        self.assertTrue(result.compacted)
        self.assertIsInstance(result.context_summary, str)
        self.assertTrue(all(isinstance(message, RemoveMessage) for message in result.messages))

    def test_compaction_does_not_orphan_recent_tool_messages(self) -> None:
        messages = [
            HumanMessage(content="old user", id="old-user"),
            AIMessage(content="", tool_calls=[{"name": "read_file", "args": {"path": "README.md"}, "id": "call_1"}], id="ai"),
            ToolMessage(content="tool result", tool_call_id="call_1", id="tool"),
        ]

        result = compact_messages(
            messages,
            existing_summary=None,
            changed_files=[],
            test_result=None,
            keep_recent=1,
        )

        removed_ids = {message.id for message in result.messages}
        self.assertIn("old-user", removed_ids)
        self.assertNotIn("ai", removed_ids)
        self.assertNotIn("tool", removed_ids)

    def test_compaction_removes_all_old_blocks_but_keeps_recent_blocks(self) -> None:
        messages = [
            HumanMessage(content="old 1", id="old-1"),
            HumanMessage(content="old 2", id="old-2"),
            HumanMessage(content="old 3", id="old-3"),
            HumanMessage(content="old 4", id="old-4"),
            HumanMessage(content="recent 1", id="recent-1"),
            AIMessage(content="", tool_calls=[{"name": "read_file", "args": {"path": "app.py"}, "id": "call_1"}], id="recent-ai"),
            ToolMessage(content="fresh file content", tool_call_id="call_1", id="recent-tool"),
        ]

        result = compact_messages(
            messages,
            existing_summary=None,
            changed_files=[],
            test_result=None,
            keep_recent=3,
        )

        removed_ids = {message.id for message in result.messages}
        self.assertEqual(removed_ids, {"old-1", "old-2", "old-3", "old-4"})
        self.assertNotIn("recent-1", removed_ids)
        self.assertNotIn("recent-ai", removed_ids)
        self.assertNotIn("recent-tool", removed_ids)

    def test_llm_summary_uses_isolated_prompt_material(self) -> None:
        llm = FakeSummaryLLM()

        summary = _generate_context_summary(llm, "old messages")

        self.assertEqual(summary, "compressed summary")
        self.assertEqual(len(llm.calls), 1)
        sent_messages = llm.calls[0]
        self.assertEqual([message.type for message in sent_messages], ["system", "human"])
        self.assertIn("old messages", sent_messages[1].content)

    def test_context_summary_is_temporary_llm_context_not_state_history(self) -> None:
        original_messages = [
            SystemMessage(content="system", id="system"),
            AIMessage(content="", tool_calls=[{"name": "read_file", "args": {"path": "README.md"}, "id": "call_1"}], id="ai"),
        ]
        state = {
            "messages": original_messages,
            "context_summary": "compressed history",
        }

        llm_messages = _messages_for_llm_with_context_summary(state)

        self.assertEqual(len(original_messages), 2)
        self.assertEqual(original_messages[-1].type, "ai")
        self.assertEqual([message.type for message in llm_messages], ["system", "system", "ai"])
        self.assertIn("compressed history", llm_messages[1].content)
        self.assertIsInstance(state["context_summary"], str)
        self.assertFalse(any(message.content == "compressed history" and message.type == "ai" for message in original_messages))

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

    def test_agent_messages_keep_base_system_prompt_when_context_summary_exists(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            state = {
                "messages": [HumanMessage(content="continue")],
                "context_summary": "compressed history",
            }

            llm_messages = _messages_for_agent(state, Workspace(tmp))

        self.assertEqual(llm_messages[0].type, "system")
        self.assertEqual(llm_messages[0].content, SYSTEM_PROMPT)
        self.assertEqual(llm_messages[1].type, "system")
        self.assertIn("Runtime metadata:", llm_messages[1].content)
        self.assertEqual(llm_messages[2].type, "system")
        self.assertIn("compressed history", llm_messages[2].content)

    def test_agent_messages_do_not_duplicate_base_system_prompt(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            state = {
                "messages": [
                    SystemMessage(content=SYSTEM_PROMPT),
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content="hello"),
                ],
                "context_summary": None,
            }

            llm_messages = _messages_for_agent(state, Workspace(tmp))

        base_prompt_count = sum(
            1
            for message in llm_messages
            if message.type == "system" and message.content == SYSTEM_PROMPT
        )
        self.assertEqual(base_prompt_count, 1)

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
