from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


STATE_DIR = ".code-agent"
CHECKPOINT_DB = "checkpoints.sqlite3"


@dataclass(frozen=True)
class SessionRecord:
    thread_id: str
    title: str
    created_at: str
    updated_at: str


def project_state_dir(workspace: str | Path) -> Path:
    return Path(workspace).expanduser().resolve() / STATE_DIR


def checkpoint_db_path(workspace: str | Path) -> Path:
    return project_state_dir(workspace) / CHECKPOINT_DB


@contextmanager
def open_project_checkpointer(workspace: str | Path) -> Iterator[Any]:
    path = checkpoint_db_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    ensure_session_store(workspace)
    from langgraph.checkpoint.sqlite import SqliteSaver

    with SqliteSaver.from_conn_string(str(path)) as checkpointer:
        yield checkpointer


def ensure_session_store(workspace: str | Path) -> None:
    path = checkpoint_db_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS code_agent_sessions (
                thread_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )


def record_session(workspace: str | Path, thread_id: str, *, title: str | None = None) -> None:
    ensure_session_store(workspace)
    now = _utc_now()
    clean_title = _clean_title(title) if title else "Untitled conversation"
    with sqlite3.connect(checkpoint_db_path(workspace)) as conn:
        conn.execute(
            """
            INSERT INTO code_agent_sessions (thread_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(thread_id) DO UPDATE SET
                title = CASE
                    WHEN code_agent_sessions.title = 'Untitled conversation' THEN excluded.title
                    ELSE code_agent_sessions.title
                END,
                updated_at = excluded.updated_at
            """,
            (thread_id, clean_title, now, now),
        )


def list_sessions(workspace: str | Path, *, limit: int | None = None) -> list[SessionRecord]:
    ensure_session_store(workspace)
    query = """
        SELECT thread_id, title, created_at, updated_at
        FROM code_agent_sessions
        ORDER BY updated_at DESC
    """
    params: tuple[int, ...] = ()
    if limit is not None:
        query += " LIMIT ?"
        params = (limit,)

    with sqlite3.connect(checkpoint_db_path(workspace)) as conn:
        rows = conn.execute(query, params).fetchall()
    return [record for row in rows if (record := _record_from_row(row)) is not None]


def _record_from_row(row) -> SessionRecord | None:
    if row is None:
        return None
    return SessionRecord(
        thread_id=str(row[0]),
        title=str(row[1]),
        created_at=str(row[2]),
        updated_at=str(row[3]),
    )


def _clean_title(title: str) -> str:
    text = " ".join(title.split())
    if not text:
        return "Untitled conversation"
    return text[:80]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
