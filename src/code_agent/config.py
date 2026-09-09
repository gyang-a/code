from __future__ import annotations

import os
from dataclasses import dataclass, field


DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_MAX_ITERATIONS = 30
DEFAULT_FILE_READ_LIMIT = 200_000
DEFAULT_FILE_READ_MAX_LINES = 100
DEFAULT_TOOL_OUTPUT_LIMIT = 12_000
DEFAULT_MAX_TOOL_CALLS_PER_TURN = 5
DEFAULT_MAX_TOTAL_TOOL_CALLS_PER_RUN = 80
DEFAULT_MODEL_REQUEST_TIMEOUT_SECONDS = 120.0
DEFAULT_MODEL_TOTAL_TIMEOUT_SECONDS = 390.0
DEFAULT_MODEL_MAX_RETRIES = 2
DEFAULT_MODEL_RETRY_BASE_DELAY_SECONDS = 1.0
DEFAULT_MODEL_RETRY_MAX_DELAY_SECONDS = 6.0
DEFAULT_SHELL_TIMEOUT_MS = 10_000
DEFAULT_SHELL_MAX_TIMEOUT_MS = 120_000
DEFAULT_SHELL_OUTPUT_LIMIT = 12_000
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
    ".code-agent/memory.sqlite3",
    ".code-agent/memory.sqlite3-*",
    ".code-agent/traces.sqlite3",
    ".code-agent/traces.sqlite3-*",
    ".code-agent/traces/**",
    ".code-agent/.gitignore",
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


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class AgentConfig:
    model: str = field(default_factory=lambda: os.getenv("CODE_AGENT_MODEL", DEFAULT_MODEL))
    api_key: str | None = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY"))
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    file_read_limit: int = DEFAULT_FILE_READ_LIMIT
    file_read_max_lines: int = DEFAULT_FILE_READ_MAX_LINES
    tool_output_limit: int = DEFAULT_TOOL_OUTPUT_LIMIT
    max_tool_calls_per_turn: int = DEFAULT_MAX_TOOL_CALLS_PER_TURN
    max_total_tool_calls_per_run: int = DEFAULT_MAX_TOTAL_TOOL_CALLS_PER_RUN
    model_request_timeout_seconds: float = field(
        default_factory=lambda: _env_float(
            "CODE_AGENT_MODEL_REQUEST_TIMEOUT_SECONDS",
            DEFAULT_MODEL_REQUEST_TIMEOUT_SECONDS,
        )
    )
    model_total_timeout_seconds: float = field(
        default_factory=lambda: _env_float(
            "CODE_AGENT_MODEL_TOTAL_TIMEOUT_SECONDS",
            DEFAULT_MODEL_TOTAL_TIMEOUT_SECONDS,
        )
    )
    model_max_retries: int = field(
        default_factory=lambda: _env_int(
            "CODE_AGENT_MODEL_MAX_RETRIES",
            DEFAULT_MODEL_MAX_RETRIES,
        )
    )
    model_retry_base_delay_seconds: float = DEFAULT_MODEL_RETRY_BASE_DELAY_SECONDS
    model_retry_max_delay_seconds: float = DEFAULT_MODEL_RETRY_MAX_DELAY_SECONDS
    fallback_model: str | None = field(default_factory=lambda: os.getenv("CODE_AGENT_FALLBACK_MODEL"))
    shell_timeout_ms: int = DEFAULT_SHELL_TIMEOUT_MS
    shell_max_timeout_ms: int = DEFAULT_SHELL_MAX_TIMEOUT_MS
    shell_output_limit: int = DEFAULT_SHELL_OUTPUT_LIMIT
    shell_mode: str = 'workspace-write'
    shell_approval_policy: str = 'on-risk'
    # Host-owned exact commands; never load allow rules from an agent-writable file.
    shell_allowed_commands: tuple[str, ...] = ()
    context_message_limit: int = DEFAULT_CONTEXT_MESSAGE_LIMIT
    context_token_limit: int = DEFAULT_CONTEXT_TOKEN_LIMIT
    context_keep_recent: int = DEFAULT_CONTEXT_KEEP_RECENT
    exclude_globs: tuple[str, ...] = field(default_factory=lambda: DEFAULT_EXCLUDE_GLOBS)
