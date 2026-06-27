from __future__ import annotations

import subprocess
from pathlib import Path

from code_agent.services.memory import format_project_memory, load_project_memory
from code_agent.services.summarizer import truncate
from code_agent.services.workspace import Workspace, WorkspaceError


PROJECT_MARKERS = (
    "README.md",
    "AGENTS.md",
    "CLAUDE.md",
    "pyproject.toml",
    "package.json",
    "uv.lock",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
)


def build_turn_metadata(workspace: Workspace, *, max_entries: int = 40) -> str:
    lines = [
        "Runtime metadata:",
        f"- workspace: {workspace.root}",
        f"- process_cwd: {Path.cwd().resolve()}",
        f"- git_branch: {_git_one_line(workspace, ['branch', '--show-current']) or 'unknown'}",
        f"- git_status: {_git_status(workspace)}",
        f"- project_markers: {_project_markers(workspace)}",
        "- top_level:",
        *_top_level_entries(workspace, max_entries=max_entries),
    ]

    memory = format_project_memory(load_project_memory(workspace))
    if memory.strip():
        lines.extend(["", "Project memory:", truncate(memory, 4000)])

    return "\n".join(lines)


def _project_markers(workspace: Workspace) -> str:
    present = [name for name in PROJECT_MARKERS if (workspace.root / name).exists()]
    return ", ".join(present) if present else "none"


def _top_level_entries(workspace: Workspace, *, max_entries: int) -> list[str]:
    entries: list[str] = []
    try:
        for child in sorted(workspace.root.iterdir(), key=lambda path: (not path.is_dir(), path.name.lower())):
            if workspace.is_excluded(child) or workspace.is_sensitive(child):
                continue
            suffix = "/" if child.is_dir() else ""
            entries.append(f"  - {child.name}{suffix}")
            if len(entries) >= max_entries:
                entries.append("  - ... truncated ...")
                break
    except (OSError, WorkspaceError) as exc:
        return [f"  - ERROR: {exc}"]
    return entries or ["  - none"]


def _git_status(workspace: Workspace) -> str:
    status = _git(workspace, ["status", "--short"])
    if status is None:
        return "unknown"
    if not status.strip():
        return "clean"
    return truncate(" | ".join(line.strip() for line in status.splitlines() if line.strip()), 1000)


def _git_one_line(workspace: Workspace, args: list[str]) -> str:
    output = _git(workspace, args)
    return output.strip().splitlines()[0] if output and output.strip() else ""


def _git(workspace: Workspace, args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=workspace.root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            shell=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout
