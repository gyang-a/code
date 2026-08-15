from __future__ import annotations

import json
import re
import shutil
import sqlite3
import subprocess
import time
import uuid
from collections.abc import Mapping
from contextlib import closing
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from code_agent.services.skills import SkillInfo
from code_agent.services.summarizer import truncate
from code_agent.services.workspace import Workspace, WorkspaceError


STATE_DIR = ".code-agent"
TRACE_DIR = "traces"
UNDO_DIR = "undo"
PROJECT_TRACE_FILE = "project_trace.json"
PROJECT_TRACE_DB = "traces.sqlite3"
DEFAULT_REVIEW_RECENT_TURNS = 10
MAX_FULL_TRACE_TURNS = 100
UNDO_RECENT_TURNS = 20
UNDO_MAX_AGE_DAYS = 30
SUMMARY_LIST_LIMIT = 40
WRITE_TOOL_NAMES = {"patch_file", "create_file", "write_file", "delete_file"}
SNAPSHOT_TOOL_NAMES = WRITE_TOOL_NAMES | {"shell_command"}
VALIDATION_TOOL_NAMES = {"git_status", "git_diff"}
SENSITIVE_KEY_RE = re.compile(r"(api[_-]?key|token|secret|password|passwd|credential)", re.I)
FEEDBACK_RE = re.compile(
    r"(\u4e0d\u5bf9|\u4e0d\u662f|\u4e0d\u5e94\u8be5|\u5e94\u8be5|\u95ee\u9898|\u5931\u8d25|\u9519\u4e86|\u9519\u8bef|\u522b|\u4e0d\u8981|\u4ee5\u540e|\u4e0b\u6b21|\u8bb0\u4f4f|\u6ee1\u610f|bug|wrong|should|shouldn't|do not)",
    re.I,
)
_CURRENT_TRACE_RECORDER: ContextVar[Any] = ContextVar("current_trace_recorder", default=None)


@dataclass
class TurnTraceRecorder:
    workspace: str | Path
    thread_id: str
    user_request: str
    git_baseline_status: str = ""
    git_baseline_dirty_paths: list[str] = field(default_factory=list)
    turn_id: str = field(default_factory=lambda: f"turn_{uuid.uuid4().hex[:8]}")

    def __post_init__(self) -> None:
        self.workspace = Path(self.workspace).expanduser().resolve()
        self.started_at = _utc_now()
        self.tool_trace: list[dict[str, Any]] = []
        self._call_index: dict[str, int] = {}
        self._seen_tool_calls: set[str] = set()
        self._seen_tool_results: set[str] = set()
        self.undo_snapshots: list[dict[str, Any]] = []
        self._snapshot_paths: set[str] = set()
        self.shell_file_changes: list[dict[str, Any]] = []

    def capture_write_snapshot(self, tool_name: str, path: str) -> None:
        if tool_name not in SNAPSHOT_TOOL_NAMES:
            return

        rel_path = _normalize_workspace_path(self.workspace, path)
        if not rel_path:
            return

        baseline_dirty = {
            _normalize_path(str(item))
            for item in self.git_baseline_dirty_paths
            if str(item)
        }
        if rel_path not in baseline_dirty or rel_path in self._snapshot_paths:
            return

        workspace = Workspace(self.workspace)
        target = workspace.resolve(rel_path)
        if target.exists() and not target.is_file():
            raise WorkspaceError(f"Cannot snapshot non-file path for undo: {rel_path}")

        ensure_agent_state_ignored(self.workspace)
        snapshot_id = uuid.uuid4().hex
        snapshot_path = undo_root(self.workspace) / self.turn_id / f"{snapshot_id}.bin"
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)

        existed = target.exists()
        if existed:
            snapshot_path.write_bytes(target.read_bytes())
        else:
            snapshot_path.write_bytes(b"")

        self._snapshot_paths.add(rel_path)
        self.undo_snapshots.append(
            {
                "path": rel_path,
                "snapshot_path": snapshot_path.relative_to(self.workspace).as_posix(),
                "existed": existed,
                "tool": tool_name,
                "created_at": _utc_now(),
            }
        )

    def capture_workspace_write_snapshots(self, tool_name: str) -> None:
        """Protect user-dirty files before a command gains workspace write access."""
        if tool_name != "shell_command":
            return
        for path in self.git_baseline_dirty_paths:
            normalized = _normalize_path(str(path))
            if not normalized:
                continue
            target = self.workspace / normalized
            if target.exists() and not target.is_file():
                continue
            self.capture_write_snapshot(tool_name, normalized)

    def record_shell_file_changes(self) -> None:
        current = _git_status_changes(self.workspace)
        current_paths = {change["path"] for change in current}
        baseline_paths = {
            _normalize_path(str(path))
            for path in self.git_baseline_dirty_paths
            if str(path)
        }
        for missing_path in sorted(baseline_paths - current_paths):
            current.append({"path": missing_path, "operation": "shell"})

        existing = {str(change.get("path") or "") for change in self.shell_file_changes}
        for change in current:
            if change["path"] in existing:
                continue
            self.shell_file_changes.append(
                {
                    **change,
                    "call_id": None,
                }
            )
            existing.add(change["path"])

    def record_chunk(self, chunk: Mapping[str, Any]) -> None:
        if "__interrupt__" in chunk:
            self.record_interrupt(chunk["__interrupt__"])

        for update in chunk.values():
            if not isinstance(update, Mapping):
                continue
            messages = update.get("messages")
            if not isinstance(messages, list):
                continue
            for message in messages:
                self.record_message(message)

    def record_message(self, message: Any) -> None:
        if isinstance(message, AIMessage):
            for tool_call in getattr(message, "tool_calls", None) or []:
                self.record_tool_call(tool_call)
        elif isinstance(message, ToolMessage):
            self.record_tool_result(message)

    def record_interrupt(self, interrupts: Any) -> None:
        for interrupt in _as_list(interrupts):
            value = interrupt.get("value") if isinstance(interrupt, Mapping) and "value" in interrupt else getattr(interrupt, "value", interrupt)
            payload = value if isinstance(value, Mapping) else _object_fields(value, ("action_requests",))
            action_requests = payload.get("action_requests")
            if not isinstance(action_requests, list):
                continue
            for action in action_requests:
                action_payload = action if isinstance(action, Mapping) else _object_fields(
                    action,
                    ("id", "name", "args", "action_name"),
                )
                if action_payload:
                    self.record_tool_call(action_payload)

    def record_tool_call(self, tool_call: Mapping[str, Any]) -> None:
        call_id = _tool_call_id(tool_call)
        if call_id and call_id in self._seen_tool_calls:
            return
        if call_id:
            self._seen_tool_calls.add(call_id)

        step = {
            "type": "tool_call",
            "call_id": call_id or None,
            "tool": _tool_call_name(tool_call),
            "args": sanitize_json(_tool_call_args(tool_call)),
            "status": "pending",
            "started_at": _utc_now(),
        }
        if call_id:
            self._call_index[call_id] = len(self.tool_trace)
        self.tool_trace.append(step)

    def record_tool_result(self, message: ToolMessage) -> None:
        call_id = str(getattr(message, "tool_call_id", "") or "")
        result_key = call_id or f"result_{len(self._seen_tool_results)}"
        if result_key in self._seen_tool_results:
            return
        self._seen_tool_results.add(result_key)

        content = str(getattr(message, "content", "") or "")
        message_status = str(getattr(message, "status", "") or "")
        status = "error" if message_status == "error" else _status_from_tool_result(content)
        result_payload = {
            "completed_at": _utc_now(),
            "status": status,
            "result_summary": truncate(content, 2000),
        }
        artifact = getattr(message, "artifact", None)
        if isinstance(artifact, Mapping) and isinstance(artifact.get("agent_error"), Mapping):
            result_payload["structured_error"] = sanitize_json(artifact["agent_error"])

        if call_id in self._call_index:
            self.tool_trace[self._call_index[call_id]].update(result_payload)
            return

        self.tool_trace.append(
            {
                "type": "tool_result",
                "call_id": call_id or None,
                "tool": str(getattr(message, "name", "") or "unknown"),
                **result_payload,
            }
        )

    def build_trace(
        self,
        *,
        final_answer: str,
        existing_skills: list[SkillInfo],
        user_feedback: str = "",
    ) -> dict[str, Any]:
        file_changes = _file_changes_from_steps(self.tool_trace) + list(self.shell_file_changes)
        errors = _errors_from_steps(self.tool_trace)
        return {
            "turn_id": self.turn_id,
            "thread_id": self.thread_id,
            "workspace": str(self.workspace),
            "created_at": _utc_now(),
            "started_at": self.started_at,
            "user_request": self.user_request,
            "git_baseline": {
                "status": truncate(self.git_baseline_status, 4000),
                "dirty_paths": list(self.git_baseline_dirty_paths),
            },
            "undo_snapshots": list(self.undo_snapshots),
            "final_answer": truncate(final_answer, 4000),
            "tool_trace": self.tool_trace,
            "file_changes": file_changes,
            "errors": errors,
            "validation": _validation_from_steps(self.tool_trace),
            "user_feedback": user_feedback,
            "existing_skills": [
                {
                    "name": skill.name,
                    "description": skill.description,
                    "path": skill.path,
                }
                for skill in existing_skills
            ],
        }

    def save(
        self,
        *,
        final_answer: str,
        existing_skills: list[SkillInfo],
        user_feedback: str = "",
    ) -> Path:
        trace = self.build_trace(
            final_answer=final_answer,
            existing_skills=existing_skills,
            user_feedback=user_feedback,
        )
        path, _project_trace = append_turn_trace(self.workspace, trace)
        return path


def trace_root(workspace: str | Path) -> Path:
    return Path(workspace).expanduser().resolve() / STATE_DIR / TRACE_DIR


def undo_root(workspace: str | Path) -> Path:
    return Path(workspace).expanduser().resolve() / STATE_DIR / UNDO_DIR


def project_trace_path(workspace: str | Path) -> Path:
    return Path(workspace).expanduser().resolve() / STATE_DIR / PROJECT_TRACE_DB


def legacy_project_trace_path(workspace: str | Path) -> Path:
    return trace_root(workspace) / PROJECT_TRACE_FILE


def set_current_trace_recorder(recorder: Any) -> Token:
    return _CURRENT_TRACE_RECORDER.set(recorder)


def reset_current_trace_recorder(token: Token) -> None:
    _CURRENT_TRACE_RECORDER.reset(token)


def capture_current_write_snapshot(tool_name: str, path: str) -> None:
    recorder = _CURRENT_TRACE_RECORDER.get()
    if recorder is not None:
        recorder.capture_write_snapshot(tool_name, path)


def capture_current_workspace_write_snapshots(tool_name: str) -> None:
    recorder = _CURRENT_TRACE_RECORDER.get()
    if recorder is not None:
        recorder.capture_workspace_write_snapshots(tool_name)


def record_current_shell_file_changes() -> None:
    recorder = _CURRENT_TRACE_RECORDER.get()
    if recorder is not None:
        recorder.record_shell_file_changes()


def append_turn_trace(workspace: str | Path, turn_trace: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = project_trace_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    ensure_agent_state_ignored(workspace)
    _ensure_trace_store(workspace)

    turn_trace = dict(turn_trace)
    turn_trace["trace_path"] = str(path)
    turn_id = str(turn_trace.get("turn_id") or f"turn_{uuid.uuid4().hex[:8]}")
    turn_trace["turn_id"] = turn_id
    now = _utc_now()

    with closing(sqlite3.connect(path)) as conn, conn:
        existing = conn.execute(
            "SELECT 1 FROM trace_turns WHERE turn_id = ?",
            (turn_id,),
        ).fetchone()
        current_count = _meta_int(conn, "turn_count")
        summary = _meta_json(conn, "summary", _new_summary())
        if existing is None:
            current_count += 1
            summary = _update_summary(
                _normalize_summary(summary),
                turn_trace,
                turn_count=current_count,
            )
        conn.execute(
            "INSERT INTO trace_turns(turn_id, created_at, payload_json) VALUES (?, ?, ?) "
            "ON CONFLICT(turn_id) DO UPDATE SET created_at = excluded.created_at, "
            "payload_json = excluded.payload_json",
            (turn_id, str(turn_trace.get("created_at") or now), _json_dump(turn_trace)),
        )
        _set_meta(conn, "workspace", str(Path(workspace).expanduser().resolve()))
        _set_meta(conn, "updated_at", now)
        _set_meta(conn, "latest_turn_id", turn_id)
        _set_meta(conn, "turn_count", current_count)
        _set_meta(conn, "existing_skills", list(turn_trace.get("existing_skills") or []))
        _set_meta(conn, "summary", summary)
        _prune_trace_rows(conn, keep=MAX_FULL_TRACE_TURNS)

    project_trace = load_project_trace(workspace)
    _prune_undo_snapshots(workspace, project_trace)
    return path, project_trace


def load_project_trace(workspace: str | Path) -> dict[str, Any]:
    path = project_trace_path(workspace)
    try:
        _ensure_trace_store(workspace)
        with closing(sqlite3.connect(path)) as conn:
            rows = conn.execute(
                "SELECT payload_json FROM trace_turns ORDER BY sequence"
            ).fetchall()
            turns = [_json_object(row[0]) for row in rows]
            turns = [turn for turn in turns if turn is not None]
            project_trace = _new_project_trace(workspace, path)
            project_trace.update(
                {
                    "created_at": _meta_text(conn, "created_at") or project_trace["created_at"],
                    "updated_at": _meta_text(conn, "updated_at") or project_trace["updated_at"],
                    "turn_count": _meta_int(conn, "turn_count") or len(turns),
                    "latest_turn_id": _meta_text(conn, "latest_turn_id"),
                    "existing_skills": _meta_json(conn, "existing_skills", []),
                    "summary": _normalize_summary(_meta_json(conn, "summary", _new_summary())),
                    "turns": turns,
                }
            )
            return project_trace
    except (OSError, sqlite3.Error):
        return _new_project_trace(workspace, path)


def reviewable_project_trace(
    project_trace: Mapping[str, Any],
    *,
    recent_turns_limit: int = DEFAULT_REVIEW_RECENT_TURNS,
) -> dict[str, Any]:
    turns = project_trace.get("turns")
    turn_list = turns if isinstance(turns, list) else []
    if not turn_list and project_trace.get("user_request"):
        return dict(project_trace)

    recent_turns = turn_list[-recent_turns_limit:]
    total_turns = int(project_trace.get("turn_count") or len(turn_list))
    return {
        "schema_version": project_trace.get("schema_version", 1),
        "workspace": project_trace.get("workspace", ""),
        "trace_path": project_trace.get("trace_path", ""),
        "created_at": project_trace.get("created_at", ""),
        "updated_at": project_trace.get("updated_at", ""),
        "turn_count": total_turns,
        "omitted_older_turns": max(0, total_turns - len(recent_turns)),
        "latest_turn_id": project_trace.get("latest_turn_id", ""),
        "existing_skills": project_trace.get("existing_skills", []),
        "historical_summary": _normalize_summary(project_trace.get("summary")),
        "recent_turns": recent_turns,
    }


def ensure_agent_state_ignored(workspace: str | Path) -> None:
    workspace_path = Path(workspace).expanduser().resolve()
    state_dir = workspace_path / STATE_DIR
    state_dir.mkdir(parents=True, exist_ok=True)

    local_gitignore = state_dir / ".gitignore"
    if not local_gitignore.exists():
        local_gitignore.write_text("*\n", encoding="utf-8", newline="")

    git_exclude = workspace_path / ".git" / "info" / "exclude"
    if git_exclude.is_file():
        current = git_exclude.read_text(encoding="utf-8", errors="replace")
        pattern = f"{STATE_DIR}/"
        if pattern not in {line.strip() for line in current.splitlines()}:
            suffix = "" if not current or current.endswith("\n") else "\n"
            git_exclude.write_text(f"{current}{suffix}{pattern}\n", encoding="utf-8", newline="")


def should_review_trace(trace: Mapping[str, Any], *, tool_threshold: int = 4) -> bool:
    turns = trace.get("turns")
    if isinstance(turns, list) and turns:
        latest_turn = turns[-1]
        if isinstance(latest_turn, Mapping):
            return _should_review_turn(latest_turn, tool_threshold=tool_threshold)
        return False
    return _should_review_turn(trace, tool_threshold=tool_threshold)


def sanitize_json(value: Any, *, max_string: int = 1200) -> Any:
    if isinstance(value, Mapping):
        clean: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if SENSITIVE_KEY_RE.search(key_text):
                clean[key_text] = "[REDACTED]"
            else:
                clean[key_text] = sanitize_json(item, max_string=max_string)
        return clean
    if isinstance(value, list):
        return [sanitize_json(item, max_string=max_string) for item in value[:100]]
    if isinstance(value, str):
        return truncate(value, max_string)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return truncate(str(value), max_string)


def _ensure_trace_store(workspace: str | Path) -> Path:
    path = project_trace_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    ensure_agent_state_ignored(workspace)
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS trace_meta "
            "(key TEXT PRIMARY KEY, value_json TEXT NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS trace_turns ("
            "sequence INTEGER PRIMARY KEY AUTOINCREMENT, "
            "turn_id TEXT NOT NULL UNIQUE, "
            "created_at TEXT NOT NULL, "
            "payload_json TEXT NOT NULL)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_trace_turns_created_at "
            "ON trace_turns(created_at)"
        )
        if _meta_text(conn, "created_at") == "":
            _set_meta(conn, "created_at", _utc_now())
            _set_meta(conn, "turn_count", 0)
            _set_meta(conn, "summary", _new_summary())
        _migrate_legacy_trace(conn, workspace)
    return path


def _migrate_legacy_trace(conn: sqlite3.Connection, workspace: str | Path) -> None:
    if _meta_text(conn, "legacy_migration_complete") == "true":
        return
    legacy_path = legacy_project_trace_path(workspace)
    if not legacy_path.is_file():
        _set_meta(conn, "legacy_migration_complete", "true")
        return

    try:
        raw = json.loads(legacy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        _set_meta(conn, "legacy_migration_complete", "true")
        return
    if not isinstance(raw, dict):
        _set_meta(conn, "legacy_migration_complete", "true")
        return

    legacy = _normalize_project_trace(raw, workspace, legacy_path)
    for turn in legacy.get("turns", []):
        if not isinstance(turn, Mapping):
            continue
        turn_payload = dict(turn)
        turn_id = str(turn_payload.get("turn_id") or f"turn_{uuid.uuid4().hex[:8]}")
        turn_payload["turn_id"] = turn_id
        turn_payload["trace_path"] = str(project_trace_path(workspace))
        conn.execute(
            "INSERT OR IGNORE INTO trace_turns(turn_id, created_at, payload_json) VALUES (?, ?, ?)",
            (
                turn_id,
                str(turn_payload.get("created_at") or _utc_now()),
                _json_dump(turn_payload),
            ),
        )
    _set_meta(conn, "workspace", str(Path(workspace).expanduser().resolve()))
    _set_meta(conn, "created_at", str(legacy.get("created_at") or _utc_now()))
    _set_meta(conn, "updated_at", str(legacy.get("updated_at") or _utc_now()))
    _set_meta(conn, "turn_count", int(legacy.get("turn_count") or len(legacy.get("turns", []))))
    _set_meta(conn, "latest_turn_id", str(legacy.get("latest_turn_id") or ""))
    _set_meta(conn, "existing_skills", list(legacy.get("existing_skills") or []))
    _set_meta(conn, "summary", _normalize_summary(legacy.get("summary")))
    _set_meta(conn, "legacy_migration_complete", "true")
    _prune_trace_rows(conn, keep=MAX_FULL_TRACE_TURNS)


def _prune_trace_rows(conn: sqlite3.Connection, *, keep: int) -> None:
    conn.execute(
        "DELETE FROM trace_turns WHERE sequence NOT IN "
        "(SELECT sequence FROM trace_turns ORDER BY sequence DESC LIMIT ?)",
        (max(1, keep),),
    )


def _prune_undo_snapshots(workspace: str | Path, project_trace: Mapping[str, Any]) -> None:
    root = undo_root(workspace).resolve()
    if not root.is_dir():
        return
    turns = project_trace.get("turns")
    turn_list = turns if isinstance(turns, list) else []
    protected = {
        str(turn.get("turn_id") or "")
        for turn in turn_list[-UNDO_RECENT_TURNS:]
        if isinstance(turn, Mapping)
    }
    cutoff = time.time() - UNDO_MAX_AGE_DAYS * 24 * 60 * 60
    for candidate in root.iterdir():
        try:
            resolved = candidate.resolve()
            resolved.relative_to(root)
            if not resolved.is_dir() or resolved.name in protected:
                continue
            if resolved.stat().st_mtime >= cutoff:
                continue
            shutil.rmtree(resolved)
        except (OSError, ValueError):
            continue


def _set_meta(conn: sqlite3.Connection, key: str, value: Any) -> None:
    conn.execute(
        "INSERT INTO trace_meta(key, value_json) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json",
        (key, _json_dump(value)),
    )


def _meta_json(conn: sqlite3.Connection, key: str, default: Any) -> Any:
    row = conn.execute("SELECT value_json FROM trace_meta WHERE key = ?", (key,)).fetchone()
    if row is None:
        return default
    try:
        return json.loads(str(row[0]))
    except json.JSONDecodeError:
        return default


def _meta_text(conn: sqlite3.Connection, key: str) -> str:
    value = _meta_json(conn, key, "")
    return str(value) if value is not None else ""


def _meta_int(conn: sqlite3.Connection, key: str) -> int:
    value = _meta_json(conn, key, 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _json_object(raw: Any) -> dict[str, Any] | None:
    try:
        value = json.loads(str(raw))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _new_project_trace(workspace: str | Path, path: Path) -> dict[str, Any]:
    now = _utc_now()
    return {
        "schema_version": 1,
        "workspace": str(Path(workspace).expanduser().resolve()),
        "trace_path": str(path),
        "created_at": now,
        "updated_at": now,
        "turn_count": 0,
        "latest_turn_id": "",
        "existing_skills": [],
        "summary": _new_summary(),
        "turns": [],
    }


def _normalize_project_trace(data: dict[str, Any], workspace: str | Path, path: Path) -> dict[str, Any]:
    normalized = _new_project_trace(workspace, path)
    normalized.update(data)
    turns = normalized.get("turns")
    normalized["turns"] = turns if isinstance(turns, list) else []
    normalized["turn_count"] = len(normalized["turns"])
    normalized["trace_path"] = str(path)
    normalized["summary"] = _normalize_summary(normalized.get("summary"))
    return normalized


def _new_summary() -> dict[str, Any]:
    return {
        "turn_count": 0,
        "threads": [],
        "tool_usage": {},
        "changed_files": [],
        "errors": [],
        "user_feedback": [],
        "recent_user_requests": [],
        "validation": [],
    }


def _normalize_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return _new_summary()

    summary = _new_summary()
    summary.update(dict(value))
    for key in ("threads", "changed_files", "errors", "user_feedback", "recent_user_requests", "validation"):
        if not isinstance(summary.get(key), list):
            summary[key] = []
        summary[key] = list(summary[key])[-SUMMARY_LIST_LIMIT:]
    if not isinstance(summary.get("tool_usage"), Mapping):
        summary["tool_usage"] = {}
    else:
        summary["tool_usage"] = {
            str(key): int(count)
            for key, count in summary["tool_usage"].items()
            if _is_int_like(count)
        }
    summary["turn_count"] = int(summary.get("turn_count") or 0)
    return summary


def _update_summary(summary: dict[str, Any], turn_trace: Mapping[str, Any], *, turn_count: int) -> dict[str, Any]:
    turn_id = str(turn_trace.get("turn_id") or "")
    thread_id = str(turn_trace.get("thread_id") or "")
    summary["turn_count"] = turn_count
    if thread_id and thread_id not in summary["threads"]:
        summary["threads"] = _append_capped(summary["threads"], thread_id)

    request = str(turn_trace.get("user_request") or "")
    if request:
        summary["recent_user_requests"] = _append_capped(
            summary["recent_user_requests"],
            {"turn_id": turn_id, "request": truncate(request, 500)},
        )
    if turn_trace.get("user_feedback") or FEEDBACK_RE.search(request):
        feedback = str(turn_trace.get("user_feedback") or request)
        summary["user_feedback"] = _append_capped(
            summary["user_feedback"],
            {"turn_id": turn_id, "feedback": truncate(feedback, 800)},
        )

    for step in _list_of_mappings(turn_trace.get("tool_trace")):
        tool = str(step.get("tool") or "unknown")
        summary["tool_usage"][tool] = int(summary["tool_usage"].get(tool, 0)) + 1

    for change in _list_of_mappings(turn_trace.get("file_changes")):
        summary["changed_files"] = _append_capped(
            summary["changed_files"],
            {
                "turn_id": turn_id,
                "path": change.get("path"),
                "operation": change.get("operation"),
            },
        )

    for error in _list_of_mappings(turn_trace.get("errors")):
        summary["errors"] = _append_capped(
            summary["errors"],
            {
                "turn_id": turn_id,
                "tool": error.get("tool") or error.get("source"),
                "status": error.get("status") or error.get("category"),
                "message": truncate(str(error.get("message") or ""), 800),
            },
        )

    validation = str(turn_trace.get("validation") or "")
    if validation:
        summary["validation"] = _append_capped(
            summary["validation"],
            {"turn_id": turn_id, "result": truncate(validation, 800)},
        )
    return summary


def _should_review_turn(trace: Mapping[str, Any], *, tool_threshold: int) -> bool:
    tool_trace = trace.get("tool_trace")
    steps = tool_trace if isinstance(tool_trace, list) else []
    if len(steps) >= tool_threshold:
        return True
    if trace.get("file_changes"):
        return True
    if trace.get("errors"):
        return True
    if trace.get("user_feedback"):
        return True
    if FEEDBACK_RE.search(str(trace.get("user_request") or "")):
        return True
    return any(str(step.get("tool") or "") in WRITE_TOOL_NAMES for step in steps if isinstance(step, Mapping))


def _append_capped(items: list[Any], item: Any, *, limit: int = SUMMARY_LIST_LIMIT) -> list[Any]:
    items.append(item)
    return items[-limit:]


def _list_of_mappings(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value] if value else []


def _tool_call_id(tool_call: Mapping[str, Any]) -> str:
    return str(tool_call.get("id") or tool_call.get("call_id") or "")


def _tool_call_name(tool_call: Mapping[str, Any]) -> str:
    return str(tool_call.get("name") or tool_call.get("action_name") or "tool")


def _tool_call_args(tool_call: Mapping[str, Any]) -> Any:
    return tool_call.get("args") or {}


def _normalize_workspace_path(workspace: str | Path, path: str) -> str:
    try:
        workspace_obj = Workspace(workspace)
        return _normalize_path(workspace_obj.relative(workspace_obj.resolve(path)))
    except WorkspaceError:
        return ""


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/").strip().strip("/")


def _object_fields(value: Any, fields: tuple[str, ...]) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for field_name in fields:
        if hasattr(value, field_name):
            data[field_name] = getattr(value, field_name)
    return data


def _is_int_like(value: Any) -> bool:
    try:
        int(value)
    except (TypeError, ValueError):
        return False
    return True


def _status_from_tool_result(content: str) -> str:
    if content.startswith("REJECTED["):
        return "rejected"
    if content.startswith("ERROR:"):
        return "error"
    return "success"


def _file_changes_from_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    for step in steps:
        if step.get("tool") not in WRITE_TOOL_NAMES or step.get("status") != "success":
            continue
        args = step.get("args") if isinstance(step.get("args"), Mapping) else {}
        path = str(args.get("path") or "")
        if not path:
            continue
        changes.append(
            {
                "path": path,
                "operation": str(step.get("tool")).replace("_file", ""),
                "call_id": step.get("call_id"),
            }
        )
    return changes


def _git_status_changes(workspace: Path) -> list[dict[str, str]]:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain=v1", "-z", "--", "."],
            cwd=workspace,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            shell=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []

    records = [record for record in result.stdout.split("\0") if record]
    changes: list[dict[str, str]] = []
    index = 0
    while index < len(records):
        record = records[index]
        status = record[:2]
        path = _normalize_path(record[3:] if len(record) > 3 else "")
        if path:
            changes.append(
                {
                    "path": path,
                    "operation": "create" if status == "??" else "shell",
                }
            )
        index += 2 if "R" in status or "C" in status else 1
    return changes


def _errors_from_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for step in steps:
        if step.get("status") not in {"error", "rejected"}:
            continue
        structured = step.get("structured_error")
        if isinstance(structured, Mapping):
            errors.append(dict(structured))
        else:
            errors.append(
                {
                    "tool": step.get("tool"),
                    "call_id": step.get("call_id"),
                    "status": step.get("status"),
                    "message": step.get("result_summary"),
                }
            )
    return errors


def _validation_from_steps(steps: list[dict[str, Any]]) -> str:
    for step in reversed(steps):
        if step.get("tool") in VALIDATION_TOOL_NAMES:
            return str(step.get("result_summary") or "")
    return ""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
