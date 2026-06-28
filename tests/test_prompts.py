from __future__ import annotations

import unittest

from code_agent.prompts import SYSTEM_PROMPT


class PromptTests(unittest.TestCase):
    def test_broad_review_has_explicit_exploration_budget(self) -> None:
        self.assertIn("broad review", SYSTEM_PROMPT)
        self.assertIn("Do not read the whole repository", SYSTEM_PROMPT)
        self.assertIn("at most 6 file reads or 12 total tool calls", SYSTEM_PROMPT)
        self.assertIn("prioritized findings", SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
