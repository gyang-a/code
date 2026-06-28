from __future__ import annotations

import unittest

from langchain_core.messages import AIMessage, HumanMessage

from code_agent.main import Session, _update_usage_from_chunk
from code_agent.services.usage import estimate_message_tokens, format_usage_line, usage_from_messages


class UsageTests(unittest.TestCase):
    def test_extracts_actual_model_usage_from_ai_messages(self) -> None:
        usage = usage_from_messages(
            [
                HumanMessage(content="hello"),
                AIMessage(
                    content="world",
                    usage_metadata={
                        "input_tokens": 10,
                        "output_tokens": 3,
                        "total_tokens": 13,
                    },
                ),
            ]
        )

        self.assertTrue(usage.actual)
        self.assertEqual(usage.input_tokens, 10)
        self.assertEqual(usage.output_tokens, 3)
        self.assertEqual(usage.total_tokens, 13)

    def test_estimates_current_context_tokens(self) -> None:
        tokens = estimate_message_tokens(
            [
                HumanMessage(content="a" * 40),
                AIMessage(content="b" * 40),
            ]
        )

        self.assertGreaterEqual(tokens, 20)

    def test_stream_usage_tracks_compression_updates(self) -> None:
        session = Session(workspace=".")

        _update_usage_from_chunk(
            session,
            {
                "model": {
                    "messages": [
                        AIMessage(
                            content="ok",
                            usage_metadata={
                                "input_tokens": 4,
                                "output_tokens": 2,
                                "total_tokens": 6,
                            },
                        )
                    ]
                },
                "SummarizationMiddleware.before_model": {"messages": [AIMessage(content="summary")]},
            },
        )

        self.assertEqual(session.model_usage.total_tokens, 6)
        self.assertEqual(session.context_compressions, 1)

    def test_format_usage_line_marks_estimates(self) -> None:
        line = format_usage_line(
            context_tokens=100,
            model_usage=usage_from_messages([AIMessage(content="estimated")]),
            compression_count=2,
        )

        self.assertIn("context~100", line)
        self.assertIn("model ~", line)
        self.assertIn("context compressions: 2", line)


if __name__ == "__main__":
    unittest.main()
