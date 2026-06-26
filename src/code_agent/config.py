from __future__ import annotations

import os
from dataclasses import dataclass, field


DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_MAX_ITERATIONS = 16
DEFAULT_FILE_READ_LIMIT = 200_000
DEFAULT_TOOL_OUTPUT_LIMIT = 12_000

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
    tool_output_limit: int = DEFAULT_TOOL_OUTPUT_LIMIT
    exclude_globs: tuple[str, ...] = field(default_factory=lambda: DEFAULT_EXCLUDE_GLOBS)
