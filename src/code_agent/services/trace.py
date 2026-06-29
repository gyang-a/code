from __future__ import annotations

import json
import os
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
WRITE_TOOL_NAMES = {"patch_file", "create_file", "write_file", "delete_file"}
VALIDATION_TOOL_NAMES = {"git_status", "git_diff"}
SENSITIVE_KEY_RE = re.compile(r"(api[_-]?key|token|secret|password|passwd|credential)", re.I)


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
        return write_turn_trace(self.workspace, trace)


def trace_root(workspace: str | Path) -> Path:
    return Path(workspace).expanduser().resolve() / STATE_DIR / TRACE_DIR


def write_turn_trace(workspace: str | Path, trace: dict[str, Any]) -> Path:
    root = trace_root(workspace)
    root.mkdir(parents=True, exist_ok=True)
    ensure_agent_state_ignored(workspace)

    safe_turn_id = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(trace.get("turn_id") or "turn"))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = root / f"{timestamp}_{safe_turn_id}.json"
    trace["trace_path"] = str(path)
    path.write_text(json.dumps(trace, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="")
    return path


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
    tool_trace = trace.get("tool_trace")
    steps = tool_trace if isinstance(tool_trace, list) else []
    if len(steps) >= tool_threshold:
        return True
    if trace.get("file_changes"):
        return True
    if trace.get("errors"):
        return True
    return any(str(step.get("tool") or "") in WRITE_TOOL_NAMES for step in steps if isinstance(step, Mapping))


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
