from __future__ import annotations

import hashlib
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from code_agent.services.summarizer import truncate
from code_agent.services.workspace import Workspace, WorkspaceError


PROJECT_RULES_FILE = "AGENTS.md"
PROJECT_MEMORY_FILE = ".code-agent/memory.md"
MEMORY_DB_FILE = ".code-agent/memory.sqlite3"
PROJECT_MEMORY_FILES = (PROJECT_RULES_FILE, PROJECT_MEMORY_FILE)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_LATIN_TOKEN_RE = re.compile(r"[a-zA-Z0-9_./-]{2,}")
_CJK_RE = re.compile(r"[\u3400-\u9fff]+")


@dataclass(frozen=True)
class MemoryEntry:
    path: str
    content: str


def load_project_memory(workspace: Workspace, *, limit: int = 8000) -> list[MemoryEntry]:
    """Load complete project memory files for compatibility and explicit inspection."""

    entries: list[MemoryEntry] = []
    for path in PROJECT_MEMORY_FILES:
        entry = _read_memory_file(workspace, path, limit=limit)
        if entry is not None:
            entries.append(entry)
    return entries


def load_relevant_project_memory(
    workspace: Workspace,
    *,
    query: str,
    limit: int = 8000,
    max_chunks: int = 5,
) -> list[MemoryEntry]:
    """Always load project rules and retrieve only relevant long-term memory chunks."""

    entries: list[MemoryEntry] = []
    rules = _read_memory_file(workspace, PROJECT_RULES_FILE, limit=min(limit, 4000))
    if rules is not None:
        entries.append(rules)

    try:
        memory_path = workspace.resolve(PROJECT_MEMORY_FILE)
        if not memory_path.is_file():
            _remove_stale_memory_index(workspace)
            return entries
        content = workspace.read_text(PROJECT_MEMORY_FILE)
        chunks = _sync_memory_index(workspace, content)
    except (OSError, sqlite3.Error, WorkspaceError):
        fallback = _read_memory_file(workspace, PROJECT_MEMORY_FILE, limit=max(0, limit - 4000))
        if fallback is not None:
            entries.append(fallback)
        return entries

    selected = _rank_chunks(chunks, query=query, max_chunks=max_chunks)
    remaining = max(0, limit - sum(len(entry.content) for entry in entries))
    for section, content in selected:
        if remaining <= 0:
            break
        rendered = truncate(content, remaining)
        entries.append(MemoryEntry(path=f"{PROJECT_MEMORY_FILE}#{section}", content=rendered))
        remaining -= len(rendered)
    return entries


def format_project_memory(entries: list[MemoryEntry]) -> str:
    if not entries:
        return "未检测到项目记忆文件。"
    return "\n\n".join(f"## {entry.path}\n{entry.content}" for entry in entries)


def _read_memory_file(workspace: Workspace, path: str, *, limit: int) -> MemoryEntry | None:
    try:
        resolved = workspace.resolve(path)
        if not resolved.exists() or not resolved.is_file():
            return None
        return MemoryEntry(path=path, content=truncate(workspace.read_text(path), limit))
    except WorkspaceError:
        return None


def _sync_memory_index(workspace: Workspace, content: str) -> list[tuple[str, str]]:
    db_path = workspace.resolve(MEMORY_DB_FILE)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    _ensure_local_state_gitignore(db_path.parent)
    content_hash = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()
    chunks = _split_markdown_chunks(content)

    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS memory_sources "
            "(source TEXT PRIMARY KEY, content_hash TEXT NOT NULL)"
        )
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS memory_chunks USING fts5("
            "source UNINDEXED, section UNINDEXED, content)"
        )
        row = conn.execute(
            "SELECT content_hash FROM memory_sources WHERE source = ?",
            (PROJECT_MEMORY_FILE,),
        ).fetchone()
        if row is None or row[0] != content_hash:
            with conn:
                conn.execute("DELETE FROM memory_chunks WHERE source = ?", (PROJECT_MEMORY_FILE,))
                conn.executemany(
                    "INSERT INTO memory_chunks(source, section, content) VALUES (?, ?, ?)",
                    [(PROJECT_MEMORY_FILE, section, chunk) for section, chunk in chunks],
                )
                conn.execute(
                    "INSERT INTO memory_sources(source, content_hash) VALUES (?, ?) "
                    "ON CONFLICT(source) DO UPDATE SET content_hash = excluded.content_hash",
                    (PROJECT_MEMORY_FILE, content_hash),
                )
        rows = conn.execute(
            "SELECT section, content FROM memory_chunks WHERE source = ? ORDER BY rowid",
            (PROJECT_MEMORY_FILE,),
        ).fetchall()
    return [(str(section), str(chunk)) for section, chunk in rows]


def _remove_stale_memory_index(workspace: Workspace) -> None:
    try:
        db_path = workspace.resolve(MEMORY_DB_FILE)
        if not db_path.is_file():
            return
        with closing(sqlite3.connect(db_path)) as conn, conn:
            conn.execute("DELETE FROM memory_chunks WHERE source = ?", (PROJECT_MEMORY_FILE,))
            conn.execute("DELETE FROM memory_sources WHERE source = ?", (PROJECT_MEMORY_FILE,))
    except (OSError, sqlite3.Error, WorkspaceError):
        return


def _split_markdown_chunks(content: str) -> list[tuple[str, str]]:
    chunks: list[tuple[str, str]] = []
    section = "general"
    lines: list[str] = []
    for line in content.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            if any(item.strip() for item in lines):
                chunks.append((section, "\n".join(lines).strip()))
            section = _section_slug(match.group(2))
            lines = [line]
        else:
            lines.append(line)
    if any(item.strip() for item in lines):
        chunks.append((section, "\n".join(lines).strip()))
    return chunks or [("general", content.strip())] if content.strip() else []


def _rank_chunks(
    chunks: list[tuple[str, str]],
    *,
    query: str,
    max_chunks: int,
) -> list[tuple[str, str]]:
    if not query.strip():
        return chunks[:max_chunks]

    query_lower = query.lower()
    query_tokens = _search_tokens(query_lower)
    scored: list[tuple[int, int, tuple[str, str]]] = []
    for index, chunk in enumerate(chunks):
        section, content = chunk
        haystack = f"{section}\n{content}".lower()
        haystack_tokens = _search_tokens(haystack)
        overlap = len(query_tokens & haystack_tokens)
        exact_bonus = 8 if query_lower in haystack else 0
        section_bonus = 3 * len(query_tokens & _search_tokens(section.lower()))
        score = exact_bonus + section_bonus + overlap
        if score > 0:
            scored.append((score, -index, chunk))
    scored.sort(reverse=True)
    return [chunk for _score, _index, chunk in scored[:max_chunks]]


def _search_tokens(text: str) -> set[str]:
    tokens = {token.lower() for token in _LATIN_TOKEN_RE.findall(text)}
    for group in _CJK_RE.findall(text):
        tokens.update(group)
        tokens.update(group[index : index + 2] for index in range(max(0, len(group) - 1)))
    return {token for token in tokens if token}


def _section_slug(value: str) -> str:
    slug = re.sub(r"\s+", "-", value.strip().lower())
    return re.sub(r"[^\w\-\u3400-\u9fff]", "", slug) or "general"


def _ensure_local_state_gitignore(state_dir: Path) -> None:
    gitignore = state_dir / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text("*\n", encoding="utf-8", newline="")
