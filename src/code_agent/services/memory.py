from __future__ import annotations

from dataclasses import dataclass

from code_agent.services.summarizer import truncate
from code_agent.services.workspace import Workspace, WorkspaceError


PROJECT_MEMORY_FILES = (
    "CLAUDE.md",
    "AGENTS.md",
    ".code-agent/memory.md",
)


@dataclass(frozen=True)
class MemoryEntry:
    path: str
    content: str


def load_project_memory(workspace: Workspace, *, limit: int = 8000) -> list[MemoryEntry]:
    entries: list[MemoryEntry] = []
    for path in PROJECT_MEMORY_FILES:
        try:
            resolved = workspace.resolve(path)
            if not resolved.exists() or not resolved.is_file():
                continue
            content = truncate(workspace.read_text(path), limit)
        except WorkspaceError:
            continue
        entries.append(MemoryEntry(path=path, content=content))
    return entries


def format_project_memory(entries: list[MemoryEntry]) -> str:
    if not entries:
        return "未检测到项目记忆文件。"
    blocks = []
    for entry in entries:
        blocks.append(f"## {entry.path}\n{entry.content}")
    return "\n\n".join(blocks)
