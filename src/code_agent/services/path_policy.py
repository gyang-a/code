"""Shared path policy; search exclusions are deliberately not write permissions."""
from pathlib import Path

PROTECTED_NAMES = frozenset({'.git', '.agents', '.codex', '.code-agent'})


def is_protected(path: Path, root: Path) -> bool:
    return bool(PROTECTED_NAMES.intersection(p.lower() for p in path.relative_to(root).parts))


def is_secret(path: Path) -> bool:
    from code_agent.config import SENSITIVE_FILE_NAMES, SENSITIVE_SUFFIXES
    return (path.name.lower() in SENSITIVE_FILE_NAMES
            or path.name.lower().startswith('.env.') and path.name.lower() != '.env.example'
            or path.name.lower().endswith(SENSITIVE_SUFFIXES)
            or '.ssh' in {p.lower() for p in path.parts})
