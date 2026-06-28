from __future__ import annotations

import difflib
import json
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from code_agent.services.summarizer import truncate
from code_agent.services.workspace import WorkspaceError


DEFAULT_SKILL_ROOT = "~/.code-agent/skills"
DEFAULT_PENDING_SKILL_ROOT = "~/.code-agent/pending/skills"
SKILL_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")


@dataclass(frozen=True)
class SkillInfo:
    name: str
    description: str
    path: str


@dataclass(frozen=True)
class PendingSkillChange:
    id: str
    skill_name: str
    action: str
    reason: str
    created_at: str
    content: str
    source: dict[str, Any]


def default_skill_root() -> Path:
    return Path(os.getenv("CODE_AGENT_SKILLS_DIR", DEFAULT_SKILL_ROOT)).expanduser().resolve()


def default_pending_skill_root() -> Path:
    return Path(
        os.getenv("CODE_AGENT_PENDING_SKILLS_DIR", DEFAULT_PENDING_SKILL_ROOT)
    ).expanduser().resolve()


def normalize_skill_name(value: str) -> str:
    name = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    name = re.sub(r"-+", "-", name)
    return name[:63].strip("-") or "learned-skill"


def validate_skill_name(name: str) -> None:
    if not SKILL_NAME_RE.fullmatch(name):
        raise WorkspaceError(
            "Invalid skill name. Use lowercase letters, digits, and hyphens only."
        )


def parse_skill_frontmatter(content: str) -> tuple[str, str]:
    if not content.startswith("---\n"):
        raise WorkspaceError("SKILL.md must start with YAML frontmatter.")

    end = content.find("\n---", 4)
    if end == -1:
        raise WorkspaceError("SKILL.md frontmatter is not closed.")

    frontmatter = content[4:end].strip().splitlines()
    data: dict[str, str] = {}
    for line in frontmatter:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip().strip("\"'")

    name = data.get("name", "")
    description = data.get("description", "")
    if not name or not description:
        raise WorkspaceError("SKILL.md frontmatter requires name and description.")
    validate_skill_name(name)
    return name, description


def validate_skill_markdown(content: str, *, expected_name: str | None = None) -> None:
    name, _description = parse_skill_frontmatter(content)
    if expected_name is not None and name != expected_name:
        raise WorkspaceError(
            f"SKILL.md name '{name}' must match target skill '{expected_name}'."
        )
    if not content.endswith("\n"):
        raise WorkspaceError("SKILL.md content must end with a newline.")


class SkillStore:
    def __init__(
        self,
        *,
        skills_dir: str | Path | None = None,
        pending_dir: str | Path | None = None,
    ) -> None:
        self._skills_dir = Path(skills_dir).expanduser().resolve() if skills_dir else default_skill_root()
        self._pending_dir = (
            Path(pending_dir).expanduser().resolve()
            if pending_dir
            else default_pending_skill_root()
        )

    @property
    def skills_dir(self) -> Path:
        return self._skills_dir

    @property
    def pending_dir(self) -> Path:
        return self._pending_dir

    def list_skills(self) -> list[SkillInfo]:
        root = self.skills_dir
        if not root.exists():
            return []

        skills: list[SkillInfo] = []
        for skill_file in sorted(root.glob("*/SKILL.md")):
            try:
                content = skill_file.read_text(encoding="utf-8", errors="replace")
                name, description = parse_skill_frontmatter(content)
                if skill_file.parent.name != name:
                    continue
                skills.append(
                    SkillInfo(
                        name=name,
                        description=description,
                        path=str(skill_file),
                    )
                )
            except (OSError, WorkspaceError):
                continue
        return skills

    def read_skill(self, name: str, *, limit: int = 12_000) -> str:
        validate_skill_name(name)
        path = self._skill_file(name)
        if not path.is_file():
            raise WorkspaceError(f"Skill not found: {name}")
        return truncate(path.read_text(encoding="utf-8", errors="replace"), limit)

    def skill_exists(self, name: str) -> bool:
        validate_skill_name(name)
        return self._skill_file(name).is_file()

    def stage_skill_change(
        self,
        *,
        skill_name: str,
        content: str,
        reason: str,
        source: dict[str, Any] | None = None,
        action: str | None = None,
    ) -> PendingSkillChange:
        validate_skill_name(skill_name)
        validate_skill_markdown(content, expected_name=skill_name)

        pending_id = uuid.uuid4().hex[:12]
        change = PendingSkillChange(
            id=pending_id,
            skill_name=skill_name,
            action=action or ("update" if self.skill_exists(skill_name) else "create"),
            reason=reason.strip() or "Skill review proposed this change.",
            created_at=datetime.now(timezone.utc).isoformat(),
            content=content,
            source=source or {},
        )
        payload = {
            "id": change.id,
            "skill_name": change.skill_name,
            "action": change.action,
            "reason": change.reason,
            "created_at": change.created_at,
            "content": change.content,
            "source": change.source,
        }
        path = self._pending_path(pending_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise WorkspaceError(f"Pending skill change already exists: {pending_id}")
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="",
        )
        return change

    def list_pending(self) -> list[PendingSkillChange]:
        root = self.pending_dir
        if not root.exists():
            return []

        changes: list[PendingSkillChange] = []
        for path in sorted(root.glob("*.json")):
            try:
                changes.append(_pending_from_json(path.read_text(encoding="utf-8")))
            except (OSError, KeyError, TypeError, json.JSONDecodeError):
                continue
        return changes

    def read_pending(self, pending_id: str) -> PendingSkillChange:
        path = self._pending_path(pending_id)
        if not path.exists():
            raise WorkspaceError(f"Pending skill change not found: {pending_id}")
        return _pending_from_json(path.read_text(encoding="utf-8"))

    def pending_diff(self, pending_id: str) -> str:
        change = self.read_pending(pending_id)
        current = ""
        current_path = self._skill_file(change.skill_name)
        if current_path.exists():
            current = current_path.read_text(encoding="utf-8", errors="replace")

        diff = difflib.unified_diff(
            current.splitlines(keepends=True),
            change.content.splitlines(keepends=True),
            fromfile=str(current_path),
            tofile=f"pending:{pending_id}",
        )
        rendered = "".join(diff)
        return rendered or "NO_DIFF"

    def approve_pending(self, pending_id: str) -> PendingSkillChange:
        change = self.read_pending(pending_id)
        validate_skill_markdown(change.content, expected_name=change.skill_name)
        path = self._skill_file(change.skill_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(change.content, encoding="utf-8", newline="")
        self._pending_path(pending_id).unlink()
        return change

    def reject_pending(self, pending_id: str) -> PendingSkillChange:
        change = self.read_pending(pending_id)
        self._pending_path(pending_id).unlink()
        return change

    def _pending_path(self, pending_id: str) -> Path:
        if not re.fullmatch(r"[a-f0-9]{12}", pending_id):
            raise WorkspaceError("Invalid pending skill change id.")
        return self._resolve_under(self.pending_dir, f"{pending_id}.json")

    def _skill_file(self, name: str) -> Path:
        validate_skill_name(name)
        return self._resolve_under(self.skills_dir, f"{name}/SKILL.md")

    @staticmethod
    def _resolve_under(root: Path, relative_path: str | Path) -> Path:
        path = (root / relative_path).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise WorkspaceError(f"Path escapes skill root: {relative_path}") from exc
        return path


def format_skill_index(skills: list[SkillInfo]) -> str:
    if not skills:
        return "No project skills found."
    return "\n".join(
        f"- {skill.name}: {skill.description} ({skill.path})"
        for skill in skills
    )


def format_pending_summary(changes: list[PendingSkillChange]) -> str:
    if not changes:
        return "No pending skill changes."
    return "\n".join(
        f"- {change.id} [{change.action}] {change.skill_name}: {change.reason}"
        for change in changes
    )


def _pending_from_json(raw: str) -> PendingSkillChange:
    data = json.loads(raw)
    return PendingSkillChange(
        id=str(data["id"]),
        skill_name=str(data["skill_name"]),
        action=str(data["action"]),
        reason=str(data["reason"]),
        created_at=str(data["created_at"]),
        content=str(data["content"]),
        source=dict(data.get("source") or {}),
    )
