from __future__ import annotations

import os
from dataclasses import dataclass, field


DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_MAX_ITERATIONS = 30
DEFAULT_FILE_READ_LIMIT = 200_000
DEFAULT_FILE_READ_MAX_LINES = 100
DEFAULT_TOOL_OUTPUT_LIMIT = 12_000
DEFAULT_MAX_TOOL_CALLS_PER_TURN = 3
DEFAULT_CONTEXT_MESSAGE_LIMIT = 30
DEFAULT_CONTEXT_TOKEN_LIMIT = 84_000
DEFAULT_CONTEXT_KEEP_RECENT = 18

DEFAULT_EXCLUDE_GLOBS = (
    ".git/**",
    ".venv/**",
    "venv/**",
    "node_modules/**",
    "dist/**",
    "build/**",
    ".next/**",
    ".pytest_cache/**",
    "__pycache__/**",
    ".code-agent/checkpoints.sqlite3",
    ".code-agent/checkpoints.sqlite3-*",
)

SENSITIVE_FILE_NAMES = {
    ".env",
    ".env.local",
    ".env.production",
    ".env.development",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
}

SENSITIVE_SUFFIXES = (
    ".pem",
    ".key",
    ".p12",
    ".pfx",
    ".crt",
    ".cer",
)


@dataclass(frozen=True)
class AgentConfig:
    model: str = field(default_factory=lambda: os.getenv("CODE_AGENT_MODEL", DEFAULT_MODEL))
    api_key: str | None = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY"))
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    file_read_limit: int = DEFAULT_FILE_READ_LIMIT
    file_read_max_lines: int = DEFAULT_FILE_READ_MAX_LINES
    tool_output_limit: int = DEFAULT_TOOL_OUTPUT_LIMIT
    max_tool_calls_per_turn: int = DEFAULT_MAX_TOOL_CALLS_PER_TURN
    context_message_limit: int = DEFAULT_CONTEXT_MESSAGE_LIMIT
    context_token_limit: int = DEFAULT_CONTEXT_TOKEN_LIMIT
    context_keep_recent: int = DEFAULT_CONTEXT_KEEP_RECENT
    exclude_globs: tuple[str, ...] = field(default_factory=lambda: DEFAULT_EXCLUDE_GLOBS)
