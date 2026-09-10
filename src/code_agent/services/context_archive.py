from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from langchain_core.messages import message_to_dict


class ContextArchive:
    """Immutable, thread-scoped originals, independent of model-visible projections."""

    def __init__(self, workspace: str | Path):
        self.path = Path(workspace).resolve() / '.code-agent' / 'context.sqlite3'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        ignore = self.path.parent / '.gitignore'
        if not ignore.exists():
            ignore.write_text('*\n', encoding='utf-8')
        with self.connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS originals (
                    seq INTEGER PRIMARY KEY, thread TEXT NOT NULL,
                    ref TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL,
                    content TEXT NOT NULL, UNIQUE(thread, ref));
                CREATE TABLE IF NOT EXISTS projections (
                    thread TEXT PRIMARY KEY, payload TEXT NOT NULL);
            ''')

    def connect(self):
        return closing(sqlite3.connect(self.path, timeout=30))

    def put(self, thread: str, kind: str, payload: Any, content: str) -> str:
        encoded = json.dumps(payload, ensure_ascii=False, default=str, sort_keys=True)
        ref = hashlib.sha256((kind + encoded).encode()).hexdigest()[:32]
        with self.connect() as db, db:
            db.execute('INSERT OR IGNORE INTO originals(thread,ref,kind,payload,content) VALUES(?,?,?,?,?)',
                       (thread, ref, kind, encoded, content))
        return ref

    def message(self, thread: str, message) -> str:
        return self.put(thread, 'message', message_to_dict(message), str(message.content))

    def read(self, thread: str, ref: str) -> str:
        with self.connect() as db:
            row = db.execute('SELECT content FROM originals WHERE thread=? AND ref=?', (thread, ref)).fetchone()
        if row is None:
            raise ValueError('Archive reference not found in this conversation.')
        return row[0]

    def search(self, thread: str, query: str, ref: str | None = None) -> list[dict]:
        if not query:
            raise ValueError('Search query must not be empty.')
        with self.connect() as db:
            rows = db.execute('SELECT ref,content FROM originals WHERE thread=? AND (? IS NULL OR ref=?) '
                              'AND instr(content,?)>0 ORDER BY seq DESC LIMIT 20',
                              (thread, ref, ref, query)).fetchall()
        hits = []
        for key, content in rows:
            offset = content.index(query)
            hits.append(dict(ref=key, offset_chars=offset,
                             line=content[:offset].count('\n') + 1,
                             preview=content[max(0, offset-80):offset+240]))
        return hits

    def projection(self, thread: str) -> dict:
        with self.connect() as db:
            row = db.execute('SELECT payload FROM projections WHERE thread=?', (thread,)).fetchone()
        return json.loads(row[0]) if row else {}

    def save_projection(self, thread: str, value: dict):
        with self.connect() as db, db:
            db.execute('INSERT INTO projections VALUES(?,?) ON CONFLICT(thread) DO UPDATE SET payload=excluded.payload',
                       (thread, json.dumps(value, ensure_ascii=False)))
