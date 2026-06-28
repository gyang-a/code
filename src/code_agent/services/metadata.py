from __future__ import annotations

import subprocess
from pathlib import Path

from code_agent.services.memory import format_project_memory, load_project_memory
from code_agent.services.summarizer import truncate
from code_agent.services.workspace import Workspace, WorkspaceError


PROJECT_MARKERS = (
    "README.md",
    "AGENTS.md",
    "pyproject.toml",
    "requirements.txt",
    "uv.lock",
    "package.json",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "settings.gradle",
    "settings.gradle.kts",
    "go.mod",
    "Cargo.toml",
    "Makefile",
)


def build_turn_metadata(workspace: Workspace, *, max_entries: int = 25) -> str:
    """
    Build compact per-turn context for the agent.

    This should be small, stable, and action-guiding.
    Do not include file contents or large diffs here.
    """
    lines = [
        "Runtime metadata:",
        f"- current_folder_name: {workspace.root.name}",
        f"- project_type: {_detect_project_type(workspace)}",
        f"- git_branch: {_git_one_line(workspace, ['branch', '--show-current']) or 'unknown'}",
        f"- git_status: {_git_status_summary(workspace)}",
        f"- project_markers: {_project_markers(workspace)}",
        "",
        "Tool usage rules:",
        "- Use find_files/search_text before read_file when locating code.",
        "- Use read_file for small bounded windows only; do not read large files blindly.",
        "- Broad review/improvement requests require bounded triage, not full-repository reading.",
        "- Broad triage budget: at most 6 file reads or 12 total tool calls before giving prioritized findings.",
        "- For broad triage, inspect project markers, entry points, config, and representative core files first.",
        "- Use patch_file for normal edits; avoid full-file rewrites.",
        "- Use create_file only for genuinely new files.",
        "- Use git_status/git_diff to inspect changes.",
        "- Command execution is not available to the agent.",
        "- Do not claim to run tests, builds, package installs, scaffolding, or dev servers.",
        "- If validation would be useful, include the exact command the user can run locally in the final answer.",
        "",
        "Command policy:",
        *_format_command_policy(workspace),
        "",
        "top_level_visible_entries:",
        *_top_level_entries(workspace, max_entries=max_entries),
    ]

    memory = format_project_memory(load_project_memory(workspace))
    if memory.strip():
        lines.extend(
            [
                "",
                "Project memory:",
                truncate(memory, 4000),
            ]
        )

    return "\n".join(lines)


def _project_markers(workspace: Workspace) -> str:
    present = [name for name in PROJECT_MARKERS if (workspace.root / name).exists()]
    return ", ".join(present) if present else "none"


def _detect_project_type(workspace: Workspace) -> str:
    root = workspace.root
    types: list[str] = []

    if (root / "package.json").exists():
        types.append("node")
    if (root / "pyproject.toml").exists() or (root / "requirements.txt").exists():
        types.append("python")
    if (root / "pom.xml").exists():
        types.append("java-maven")
    if (
        (root / "build.gradle").exists()
        or (root / "build.gradle.kts").exists()
        or (root / "settings.gradle").exists()
        or (root / "settings.gradle.kts").exists()
    ):
        types.append("java-gradle")
    if (root / "go.mod").exists():
        types.append("go")
    if (root / "Cargo.toml").exists():
        types.append("rust")
    if (root / "Makefile").exists():
        types.append("make")
    return ", ".join(types) if types else "unknown"


def _format_command_policy(workspace: Workspace) -> list[str]:
    """
    Summarize the project command policy if Workspace has one.

    This assumes you add workspace.command_policy later.
    If not present, return a conservative generic message.
    """
    policy = getattr(workspace, "command_policy", None)

    if policy is None:
        return [
            "- command execution: unavailable to the agent",
            "- validation: suggest exact commands for the user to run locally",
        ]

    lines: list[str] = []

    allowed = getattr(policy, "allowed", ())
    requires_approval = getattr(policy, "requires_approval", ())

    if allowed:
        lines.append("- allowed:")
        for pattern in allowed[:20]:
            argv_prefix = getattr(pattern, "argv_prefix", ())
            command = " ".join(argv_prefix)
            description = getattr(pattern, "description", "")
            if description:
                lines.append(f"  - {command}  # {description}")
            else:
                lines.append(f"  - {command}")
        if len(allowed) > 20:
            lines.append("  - ... truncated ...")
    else:
        lines.append("- allowed: none configured")

    if requires_approval:
        lines.append("- requires_approval:")
        for pattern in requires_approval[:20]:
            argv_prefix = getattr(pattern, "argv_prefix", ())
            command = " ".join(argv_prefix)
            description = getattr(pattern, "description", "")
            if description:
                lines.append(f"  - {command}  # {description}")
            else:
                lines.append(f"  - {command}")
        if len(requires_approval) > 20:
            lines.append("  - ... truncated ...")
    else:
        lines.append("- requires_approval: installs, scaffolding, dev servers, dependency changes, unknown commands")

    return lines


def _top_level_entries(workspace: Workspace, *, max_entries: int) -> list[str]:
    entries: list[str] = []

    try:
        for child in sorted(
            workspace.root.iterdir(),
            key=lambda path: (not path.is_dir(), path.name.lower()),
        ):
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


def _git_status_summary(workspace: Workspace) -> str:
    status = _git(workspace, ["status", "--short"])

    if status is None:
        return "unknown"

    if not status.strip():
        return "clean"

    lines = [line.strip() for line in status.splitlines() if line.strip()]
    return f"{len(lines)} changed files; use git_status or git_diff for details"


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
