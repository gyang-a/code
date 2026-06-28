from __future__ import annotations

import os
from pathlib import Path
import unittest
from unittest.mock import patch

from code_agent.services.env import load_dotenv


class EnvTests(unittest.TestCase):
    def test_load_dotenv_sets_missing_values(self) -> None:
        with TemporaryWorkspace() as tmp_path:
            env_path = tmp_path / ".env"
            env_path.write_text(
                "DEEPSEEK_API_KEY=test-key\n"
                "CODE_AGENT_MODEL=demo-model\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=True):
                loaded = load_dotenv(env_path)

                self.assertEqual(loaded["DEEPSEEK_API_KEY"], "test-key")
                self.assertEqual(os.environ["DEEPSEEK_API_KEY"], "test-key")
                self.assertEqual(os.environ["CODE_AGENT_MODEL"], "demo-model")

    def test_deepseek_key_is_primary_model_key(self) -> None:
        with TemporaryWorkspace() as tmp_path:
            env_path = tmp_path / ".env"
            env_path.write_text("DEEPSEEK_API_KEY=deepseek-key\n", encoding="utf-8")

            with patch.dict(os.environ, {}, clear=True):
                load_dotenv(env_path)

                self.assertEqual(os.environ["DEEPSEEK_API_KEY"], "deepseek-key")

    def test_existing_environment_wins_by_default(self) -> None:
        with TemporaryWorkspace() as tmp_path:
            env_path = tmp_path / ".env"
            env_path.write_text("DEEPSEEK_API_KEY=file-key\n", encoding="utf-8")

            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "env-key"}, clear=True):
                load_dotenv(env_path)

                self.assertEqual(os.environ["DEEPSEEK_API_KEY"], "env-key")


class TemporaryWorkspace:
    def __enter__(self) -> Path:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        return Path(self._tmp.name)

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._tmp.cleanup()
