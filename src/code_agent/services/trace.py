from __future__ import annotations

import json
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from code_agent.services.skills import SkillInfo
from code_agent.services.summarizer import truncate


STATE_DIR = ".code-agent"
TRACE_DIR = "traces"
PROJECT_TRACE_FILE = "project_trace.json"
DEFAULT_REVIEW_RECENT_TURNS = 8
SUMMARY_LIST_LIMIT = 40
WRITE_TOOL_NAMES = {"patch_file", "create_file", "write_file", "delete_file"}
VALIDATION_TOOL_NAMES = {"git_status", "git_diff"}
SENSITIVE_KEY_RE = re.compile(r"(api[_-]?key|token|secret|password|passwd|credential)", re.I)
FEEDBACK_RE = re.compile(
    r"(\u4e0d\u5bf9|\u4e0d\u662f|\u4e0d\u5e94\u8be5|\u5e94\u8be5|\u95ee\u9898|\u5931\u8d25|\u9519\u4e86|\u9519\u8bef|\u522b|\u4e0d\u8981|\u4ee5\u540e|\u4e0b\u6b21|\u8bb0\u4f4f|\u6ee1\u610f|bug|wrong|should|shouldn't|do not)",
    re.I,
)


@dataclass
class TurnTraceRecorder:
    workspace: str | Path
    thread_id: str
    user_request: str
    turn_id: str = field(default_factory=lambda: f"turn_{uuid.uuid4().hex[:8]}")

    def __post_init__(self) -> None:
        self.workspace = Path(self.workspace).expanduser().resolve()
        self.started_at = _utc_now()
        self.tool_trace: list[dict[str, Any]] = []
        self._call_index: dict[str, int] = {}
        self._seen_tool_calls: set[str] = set()
        self._seen_tool_results: set[str] = set()

    def record_chunk(self, chunk: Mapping[str, Any]) -> None:
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

    def record_tool_call(self, tool_call: Mapping[str, Any]) -> None:
        call_id = str(tool_call.get("id") or "")
        if call_id and call_id in self._seen_tool_calls:
            return
        if call_id:
            self._seen_tool_calls.add(call_id)

        step = {
            "type": "tool_call",
            "call_id": call_id or None,
            "tool": str(tool_call.get("name") or "tool"),
            "args": sanitize_json(tool_call.get("args") or {}),
            "status": "pending",
            "started_at": _utc_now(),
        }
        self._call_index[call_id] = len(self.tool_trace)
        self.tool_trace.append(step)

    def record_tool_result(self, message: ToolMessage) -> None:
        call_id = str(getattr(message, "tool_call_id", "") or "")
        result_key = call_id or f"result_{len(self._seen_tool_results)}"
        if result_key in self._seen_tool_results:
            return
        self._seen_tool_results.add(result_key)

        content = str(getattr(message, "content", "") or "")
        status = _status_from_tool_result(content)
        result_payload = {
            "completed_at": _utc_now(),
            "status": status,
            "result_summary": truncate(content, 2000),
        }

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
        file_changes = _file_changes_from_steps(self.tool_trace)
        errors = _errors_from_steps(self.tool_trace)
        return {
            "turn_id": self.turn_id,
            "thread_id": self.thread_id,
            "workspace": str(self.workspace),
            "created_at": _utc_now(),
            "started_at": self.started_at,
            "user_request": self.user_request,
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


def project_trace_path(workspace: str | Path) -> Path:
    return trace_root(workspace) / PROJECT_TRACE_FILE


def append_turn_trace(workspace: str | Path, turn_trace: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = project_trace_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    ensure_agent_state_ignored(workspace)

    project_trace = load_project_trace(workspace)
    turn_trace["trace_path"] = str(path)
    project_trace["workspace"] = str(Path(workspace).expanduser().resolve())
    project_trace["updated_at"] = _utc_now()
    project_trace["trace_path"] = str(path)
    project_trace["latest_turn_id"] = str(turn_trace.get("turn_id") or "")
    project_trace["existing_skills"] = list(turn_trace.get("existing_skills") or [])
    project_trace.setdefault("turns", []).append(turn_trace)
    project_trace["turn_count"] = len(project_trace["turns"])
    project_trace["summary"] = _update_summary(
        _normalize_summary(project_trace.get("summary")),
        turn_trace,
        turn_count=project_trace["turn_count"],
    )

    path.write_text(json.dumps(project_trace, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="")
    return path, project_trace


def write_turn_trace(workspace: str | Path, trace: dict[str, Any]) -> Path:
    path, _project_trace = append_turn_trace(workspace, trace)
    return path


def load_project_trace(workspace: str | Path) -> dict[str, Any]:
    path = project_trace_path(workspace)
    if not path.is_file():
        return _new_project_trace(workspace, path)

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _new_project_trace(workspace, path)
    if not isinstance(data, dict):
        return _new_project_trace(workspace, path)
    return _normalize_project_trace(data, workspace, path)


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
    return {
        "schema_version": project_trace.get("schema_version", 1),
        "workspace": project_trace.get("workspace", ""),
        "trace_path": project_trace.get("trace_path", ""),
        "created_at": project_trace.get("created_at", ""),
        "updated_at": project_trace.get("updated_at", ""),
        "turn_count": len(turn_list),
        "omitted_older_turns": max(0, len(turn_list) - len(recent_turns)),
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
                "tool": error.get("tool"),
                "status": error.get("status"),
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


def _errors_from_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for step in steps:
        if step.get("status") not in {"error", "rejected"}:
            continue
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
