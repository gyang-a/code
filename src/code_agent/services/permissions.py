from __future__ import annotations

import re
import shlex
import sys
from pathlib import Path
from typing import Any

from code_agent.models import PermissionDecision, RiskLevel
from code_agent.services.workspace import Workspace


LEVEL_2_FILES = {"package.json", "pyproject.toml"}

LEVEL_3_COMMAND_MARKERS = (
    "rm -rf",
    "sudo ",
    "chmod 777",
    "git reset --hard",
    "~/.ssh",
    ".ssh/",
    ".env",
)

LEVEL_2_COMMAND_PREFIXES = (
    "npm install",
    "npm i",
    "npm create",
    "npm exec",
    "npx",
    "pnpm add",
    "pnpm install",
    "pnpm create",
    "pnpm dlx",
    "yarn add",
    "yarn install",
    "yarn create",
    "yarn dlx",
    "pip install",
    "python -m pip install",
    "uv add",
    "uv pip install",
    "docker compose up",
    "docker-compose up",
)

DEV_SERVER_COMMAND_PREFIXES = (
    "npm run dev",
    "npm run start",
    "npm start",
    "pnpm dev",
    "pnpm run dev",
    "pnpm start",
    "yarn dev",
    "yarn start",
    "vite",
    "next dev",
    "python -m http.server",
    "flask run",
    "uvicorn",
)

LEVEL_1_COMMAND_ROOTS = {
    "pytest",
    "ruff",
    "mypy",
    "npm",
    "pnpm",
    "yarn",
    "uv",
    "python",
    "node",
    "git",
}

LEVEL_1_COMMAND_PATTERNS = (
    ("npm", "test"),
    ("npm", "run", "build"),
    ("npm", "run", "test"),
    ("npm", "run", "lint"),
    ("npm", "run", "typecheck"),
    ("npm", "run", "check"),
    ("pnpm", "test"),
    ("pnpm", "run", "test"),
    ("pnpm", "lint"),
    ("pnpm", "run", "lint"),
    ("pnpm", "build"),
    ("pnpm", "run", "build"),
    ("yarn", "test"),
    ("yarn", "run", "test"),
    ("yarn", "lint"),
    ("yarn", "run", "lint"),
    ("yarn", "build"),
    ("yarn", "run", "build"),
    ("uv", "run"),
    ("python", "-m"),
    ("git", "status"),
    ("git", "diff"),
)

SHELL_FILESYSTEM_COMMAND_ROOTS = {
    "cat",
    "copy",
    "cp",
    "del",
    "dir",
    "erase",
    "find",
    "findstr",
    "grep",
    "ls",
    "md",
    "mkdir",
    "more",
    "move",
    "mv",
    "rd",
    "ren",
    "rename",
    "rg",
    "rm",
    "rmdir",
    "touch",
    "type",
    "xcopy",
}

READ_ONLY_TOOLS = {
    "list_files",
    "get_file_tree",
    "read_file",
    "search_text",
    "find_files",
    "git_status",
    "git_diff",
}

WRITE_TOOLS = {"patch_file", "create_file", "write_file", "delete_file"}


def rejected(risk: RiskLevel, reason: str) -> str:
    return f"REJECTED[{risk.value}]: {reason}"


def describe_permission_policy() -> str:
    return "\n".join(
        [
            "Permission rules:",
            "- Level 0: read-only workspace tools are allowed.",
            "- Level 1: low-risk validation/build/test commands are allowed in the sandbox.",
            "- Level 2: package installs, scaffolding, Docker Compose, deletes, and unknown shell commands require approval.",
            "- Level 3: destructive commands, sensitive paths, and shell-based file inspection/editing are rejected.",
            "- Use dedicated tools for listing, reading, searching, diffing, creating, editing, or deleting files.",
        ]
    )


def classify_tool_call(workspace: Workspace, tool_name: str, args: dict[str, Any]) -> PermissionDecision:
    if tool_name in READ_ONLY_TOOLS:
        path = args.get("path")
        if isinstance(path, str):
            resolved = workspace.resolve(path)
            if workspace.is_sensitive(resolved):
                return PermissionDecision(
                    risk=RiskLevel.level_3,
                    allowed=False,
                    reason=f"{tool_name} rejected sensitive path: {workspace.relative(resolved)}",
                )
        return PermissionDecision(
            risk=RiskLevel.level_0,
            allowed=True,
            reason=f"{tool_name} is a read-only workspace tool.",
        )

    if tool_name in WRITE_TOOLS:
        path = args.get("path")
        if not isinstance(path, str):
            return PermissionDecision(
                risk=RiskLevel.level_3,
                allowed=False,
                reason=f"{tool_name} is missing required path argument.",
            )
        return classify_file_operation(workspace, tool_name, path)

    if tool_name in {"run_shell", "run_command"}:
        command = str(args.get("command") or "")
        decision, _ = classify_command(command, workspace)
        return decision

    return PermissionDecision(
        risk=RiskLevel.level_2,
        allowed=False,
        requires_approval=True,
        reason=f"Unknown tool requires approval: {tool_name}",
    )


def classify_file_operation(workspace: Workspace, operation: str, path: str) -> PermissionDecision:
    resolved = workspace.resolve(path)
    rel = workspace.relative(resolved)
    normalized = rel.replace("\\", "/")
    name = resolved.name.lower()

    if workspace.is_sensitive(resolved):
        return PermissionDecision(
            risk=RiskLevel.level_3,
            allowed=False,
            reason=f"{operation} rejected sensitive path: {normalized}",
        )

    if operation == "delete_file":
        return PermissionDecision(
            risk=RiskLevel.level_2,
            allowed=False,
            requires_approval=True,
            reason=f"Deleting files requires approval: {normalized}",
        )

    if name in LEVEL_2_FILES:
        return PermissionDecision(
            risk=RiskLevel.level_2,
            allowed=False,
            requires_approval=True,
            reason=f"Modifying {normalized} requires approval.",
        )

    if operation == "create_file" and not (
        normalized.startswith("src/") or normalized.startswith("tests/")
    ):
        return PermissionDecision(
            risk=RiskLevel.level_2,
            allowed=False,
            requires_approval=True,
            reason=f"Creating files outside src/ or tests/ requires approval: {normalized}",
        )

    if operation == "write_file" and not (
        normalized.startswith("src/") or normalized.startswith("tests/")
    ):
        return PermissionDecision(
            risk=RiskLevel.level_2,
            allowed=False,
            requires_approval=True,
            reason=f"Overwriting files outside src/ or tests/ requires approval: {normalized}",
        )

    return PermissionDecision(
        risk=RiskLevel.level_1,
        allowed=True,
        reason=f"{operation} is a low-risk workspace edit: {normalized}",
    )


def classify_command(command: str, workspace: Workspace) -> tuple[PermissionDecision, list[str] | None]:
    stripped = command.strip()
    lower = " ".join(stripped.lower().split())

    try:
        argv = shlex.split(stripped, posix=sys.platform != "win32")
    except ValueError:
        return (
            PermissionDecision(
                risk=RiskLevel.level_3,
                allowed=False,
                reason=f"Command could not be parsed safely: {command}",
            ),
            None,
        )

    if not argv:
        return (
            PermissionDecision(
                risk=RiskLevel.level_3,
                allowed=False,
                reason="Empty command rejected.",
            ),
            None,
        )

    if any(marker in lower for marker in LEVEL_3_COMMAND_MARKERS):
        return (
            PermissionDecision(
                risk=RiskLevel.level_3,
                allowed=False,
                reason=f"Command rejected by safety policy: {command}",
            ),
            None,
        )

    filesystem_decision = _reject_shell_filesystem_work(command)
    if filesystem_decision is not None:
        return filesystem_decision, argv

    if "|" in lower and "bash" in lower and ("curl" in lower or "wget" in lower):
        return (
            PermissionDecision(
                risk=RiskLevel.level_3,
                allowed=False,
                reason=f"Command rejected by safety policy: {command}",
            ),
            None,
        )

    if _command_contains_prefix(lower, DEV_SERVER_COMMAND_PREFIXES):
        return (
            PermissionDecision(
                risk=RiskLevel.level_3,
                allowed=False,
                reason=(
                    "Do not start long-running dev servers from the agent. "
                    "Use a focused build, lint, or test command for validation."
                ),
            ),
            argv,
        )

    if _command_contains_prefix(lower, LEVEL_2_COMMAND_PREFIXES):
        return (
            PermissionDecision(
                risk=RiskLevel.level_2,
                allowed=False,
                requires_approval=True,
                reason=f"Shell command may download packages, scaffold files, or modify dependencies: {command}",
            ),
            argv,
        )

    if _is_level_1_command(argv):
        return (
            PermissionDecision(
                risk=RiskLevel.level_1,
                allowed=True,
                reason=f"Allowed low-risk shell command: {command}",
            ),
            argv,
        )

    return (
        PermissionDecision(
            risk=RiskLevel.level_2,
            allowed=False,
            requires_approval=True,
            reason=f"Unknown or medium-risk shell command requires approval: {command}",
        ),
        argv,
    )


def _is_level_1_command(argv: list[str]) -> bool:
    root = _normalize_command_root(argv[0])
    if root not in LEVEL_1_COMMAND_ROOTS:
        return False

    if root in {"pytest", "ruff", "mypy", "node"}:
        return True

    normalized = tuple(
        _normalize_command_root(part) if index == 0 else part.lower()
        for index, part in enumerate(argv)
    )
    return any(_argv_startswith(normalized, pattern) for pattern in LEVEL_1_COMMAND_PATTERNS)


def _reject_shell_filesystem_work(command: str) -> PermissionDecision | None:
    for segment in _shell_command_segments(command):
        root = _shell_segment_root(segment)
        if root in {"cd", "chdir", "pushd", "popd"}:
            continue
        if root in SHELL_FILESYSTEM_COMMAND_ROOTS or _segment_has_redirection(segment):
            return PermissionDecision(
                risk=RiskLevel.level_3,
                allowed=False,
                reason=(
                    "Use dedicated workspace tools instead of shell for file inspection or edits "
                    "(list_files/read_file/search_text/find_files/git_diff/patch_file/create_file/write_file/delete_file)."
                ),
            )
    return None


def _shell_command_segments(command: str) -> list[str]:
    normalized = " ".join(command.split())
    segments = [normalized]
    for separator in ("&&", "||", ";", "&"):
        segments = [
            part.strip()
            for segment in segments
            for part in segment.split(separator)
            if part.strip()
        ]
    return segments


def _shell_segment_root(segment: str) -> str:
    try:
        parts = shlex.split(segment, posix=sys.platform != "win32")
    except ValueError:
        parts = segment.split()
    if not parts:
        return ""
    return _normalize_command_root(parts[0])


def _normalize_command_root(value: str) -> str:
    root = Path(value.strip("\"'")).name.lower()
    if root.endswith((".cmd", ".exe", ".bat")):
        root = root.rsplit(".", 1)[0]
    return root


def _segment_has_redirection(segment: str) -> bool:
    without_quotes = re.sub(r"""(['"]).*?\1""", "", segment)
    without_stderr_merge = without_quotes.replace("2>&1", "").replace("1>&2", "")
    return bool(re.search(r"(^|\s)\d?>{1,2}(?!=|&)", without_stderr_merge))


def _command_contains_prefix(lower_command: str, prefixes: tuple[str, ...]) -> bool:
    return any(segment.startswith(prefix) for segment in _shell_command_segments(lower_command) for prefix in prefixes)


def _argv_startswith(argv: tuple[str, ...], prefix: tuple[str, ...]) -> bool:
    return len(argv) >= len(prefix) and argv[: len(prefix)] == prefix


def normalize_path_arg(path: str | Path) -> str:
    return str(path).replace("\\", "/")
