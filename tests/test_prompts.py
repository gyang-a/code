from __future__ import annotations

import unittest

from code_agent.prompts import SYSTEM_PROMPT


class PromptTests(unittest.TestCase):
    def test_prompt_reuses_available_file_context_without_forced_rereads(self) -> None:
        self.assertIn("Reuse relevant file content already available", SYSTEM_PROMPT)
        self.assertNotIn("Read the relevant files before editing", SYSTEM_PROMPT)

    def test_broad_review_has_explicit_exploration_budget(self) -> None:
        self.assertIn("broad review", SYSTEM_PROMPT)
        self.assertIn("Do not read the whole repository", SYSTEM_PROMPT)
        self.assertIn("at most 6 file reads or 12 total tool calls", SYSTEM_PROMPT)
        self.assertIn("prioritized findings", SYSTEM_PROMPT)

    def test_prompt_does_not_suggest_absolute_workspace_path(self) -> None:
        self.assertNotIn("/workspace", SYSTEM_PROMPT)
        self.assertIn("current project folder", SYSTEM_PROMPT)
        self.assertIn("Use shell_command", SYSTEM_PROMPT)
        self.assertIn("independently of approvals", SYSTEM_PROMPT)
        self.assertIn("Never request full-access", SYSTEM_PROMPT)
        self.assertNotIn("sandbox_permissions=", SYSTEM_PROMPT)



if __name__ == "__main__":
    unittest.main()
