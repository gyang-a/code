from __future__ import annotations

import re
import shlex
import sys
from pathlib import Path
from typing import Any

from code_agent.models import PermissionDecision, RiskLevel
from code_agent.services.workspace import Workspace, WorkspaceError


LEVEL_2_FILES = {
    "package.json",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "pyproject.toml",
    "requirements.txt",
    "uv.lock",
}

LEVEL_3_COMMAND_MARKERS = (
    "rm -rf",
    "sudo ",
    "chmod 777",
    "git reset --hard",
    "git clean -fd",
    "git clean -xdf",
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
    "next dev",
    "vite",
    "python -m http.server",
    "flask run",
    "uvicorn",
    "npm run serve",
    "pnpm serve",
    "yarn serve",
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
    "sed",
    "awk",
    "perl",
    "tee",
    "echo",
    "printf",
}

READ_ONLY_TOOLS = {
    "list_files",
    "read_file",
    "search_text",
    "find_files",
    "git_status",
    "git_diff",
}

WRITE_TOOLS = {
    "patch_file",
    "create_file",
    "write_file",
    "delete_file",
}

ALLOWED_SIMPLE_ROOTS = {
    "pytest",
    "mypy",
}

ALLOWED_ARGV_PREFIXES = (
    ("pytest",),
    ("python", "-m", "pytest"),
    ("ruff", "check"),
    ("ruff", "format", "--check"),
    ("mypy",),
    ("npm", "test"),
    ("npm", "run", "test"),
    ("npm", "run", "build"),
    ("npm", "run", "lint"),
    ("npm", "run", "typecheck"),
    ("npm", "run", "check"),
    ("pnpm", "test"),
    ("pnpm", "run", "test"),
    ("pnpm", "run", "build"),
    ("pnpm", "run", "lint"),
    ("pnpm", "run", "typecheck"),
    ("pnpm", "run", "check"),
    ("yarn", "test"),
    ("yarn", "run", "test"),
    ("yarn", "run", "build"),
    ("yarn", "run", "lint"),
    ("yarn", "run", "typecheck"),
    ("yarn", "run", "check"),
    ("tsc",),
    ("npx", "--no-install", "tsc"),
    ("npx", "--no-install", "vite", "build"),
    ("npx", "--no-install", "vite", "--version"),
    ("npx", "--no-install", "oxlint"),
    ("uv", "run", "pytest"),
    ("uv", "run", "python", "-m", "pytest"),
    ("uv", "run", "ruff", "check"),
    ("uv", "run", "ruff", "format", "--check"),
    ("uv", "run", "mypy"),
)


def rejected(risk: RiskLevel, reason: str) -> str:
    return f"REJECTED[{risk.value}]: {reason}"


def describe_permission_policy() -> str:
    return "\n".join(
        [
            "Permission rules:",
            "- Level 0: read-only workspace tools are allowed.",
            "- Level 1: low-risk validation/build/test commands are allowed in the sandbox.",
            "- Level 2: package installs, scaffolding, deletes, full-file overwrites, and unknown shell commands require approval.",
            "- Level 3: destructive commands, sensitive paths, shell-based file inspection/editing, redirection, pipes, and command substitution are rejected.",
            "- Use dedicated tools for listing, reading, searching, diffing, creating, editing, or deleting files.",
        ]
    )


def classify_tool_call(
    workspace: Workspace,
    tool_name: str,
    args: dict[str, Any],
) -> PermissionDecision:
    if tool_name in READ_ONLY_TOOLS:
        path = args.get("path", ".")

        if isinstance(path, str):
            try:
                resolved = workspace.resolve(path)

                if workspace.is_sensitive(resolved):
                    return PermissionDecision(
                        risk=RiskLevel.level_3,
                        allowed=False,
                        reason=f"{tool_name} rejected sensitive path: {workspace.relative(resolved)}",
                    )

                if workspace.is_excluded(resolved):
                    return PermissionDecision(
                        risk=RiskLevel.level_3,
                        allowed=False,
                        reason=f"{tool_name} rejected excluded path: {workspace.relative(resolved)}",
                    )

            except WorkspaceError as exc:
                return PermissionDecision(
                    risk=RiskLevel.level_3,
                    allowed=False,
                    reason=str(exc),
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


def classify_file_operation(
    workspace: Workspace,
    operation: str,
    path: str,
) -> PermissionDecision:
    try:
        resolved = workspace.resolve(path)
        rel = workspace.relative(resolved)
    except WorkspaceError as exc:
        return PermissionDecision(
            risk=RiskLevel.level_3,
            allowed=False,
            reason=str(exc),
        )

    normalized = rel.replace("\\", "/")
    name = resolved.name.lower()

    if workspace.is_sensitive(resolved):
        return PermissionDecision(
            risk=RiskLevel.level_3,
            allowed=False,
            reason=f"{operation} rejected sensitive path: {normalized}",
        )

    if workspace.is_excluded(resolved):
        return PermissionDecision(
            risk=RiskLevel.level_3,
            allowed=False,
            reason=f"{operation} rejected excluded path: {normalized}",
        )

    if operation == "delete_file":
        return PermissionDecision(
            risk=RiskLevel.level_2,
            allowed=False,
            requires_approval=True,
            reason=f"Deleting files requires approval: {normalized}",
        )

    if operation == "write_file":
        return PermissionDecision(
            risk=RiskLevel.level_2,
            allowed=False,
            requires_approval=True,
            reason=f"Full-file overwrite requires approval: {normalized}",
        )

    if name in LEVEL_2_FILES:
        return PermissionDecision(
            risk=RiskLevel.level_2,
            allowed=False,
            requires_approval=True,
            reason=f"Modifying dependency/config file requires approval: {normalized}",
        )

    if operation == "create_file" and not _is_normal_project_path(normalized):
        return PermissionDecision(
            risk=RiskLevel.level_2,
            allowed=False,
            requires_approval=True,
            reason=f"Creating files outside normal project paths requires approval: {normalized}",
        )

    return PermissionDecision(
        risk=RiskLevel.level_1,
        allowed=True,
        reason=f"{operation} is a low-risk workspace edit: {normalized}",
    )


def classify_command(
    command: str,
    workspace: Workspace,
) -> tuple[PermissionDecision, list[str] | None]:
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

    shell_syntax_decision = _reject_shell_syntax(command)
    if shell_syntax_decision is not None:
        if _has_command_substitution(command) or _has_pipe(command):
            return shell_syntax_decision, None
        return shell_syntax_decision, argv

    filesystem_decision = _reject_shell_filesystem_work(command)
    if filesystem_decision is not None:
        return filesystem_decision, argv

    git_decision = _reject_git_shell_work(command)
    if git_decision is not None:
        return git_decision, argv

    if _is_dev_server_command(lower):
        return (
            PermissionDecision(
                risk=RiskLevel.level_3,
                allowed=False,
                reason=(
                    "Long-running dev server commands are rejected because run_shell waits "
                    "for the process to exit. Use build/test/lint for validation, or start "
                    "the dev server in a separate local terminal."
                ),
            ),
            argv,
        )

    if _is_level_1_shell_command(stripped):
        return (
            PermissionDecision(
                risk=RiskLevel.level_1,
                allowed=True,
                reason=f"Allowed low-risk shell command: {command}",
            ),
            argv,
        )

    if _command_contains_prefix(lower, LEVEL_2_COMMAND_PREFIXES):
        return (
            PermissionDecision(
                risk=RiskLevel.level_2,
                allowed=False,
                requires_approval=True,
                reason=(
                    "Shell command may download packages, scaffold files, "
                    f"or modify dependencies: {command}"
                ),
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


def _is_normal_project_path(normalized: str) -> bool:
    return normalized.startswith(
        (
            "src/",
            "tests/",
            "test/",
            "app/",
            "server/",
            "client/",
            "frontend/",
            "backend/",
            "docs/",
        )
    )


def _reject_shell_syntax(command: str) -> PermissionDecision | None:
    if _has_command_substitution(command):
        return PermissionDecision(
            risk=RiskLevel.level_3,
            allowed=False,
            reason="Command substitution is rejected. Use a simple validation command.",
        )

    if _has_pipe(command):
        return PermissionDecision(
            risk=RiskLevel.level_3,
            allowed=False,
            reason="Pipes are rejected. Use dedicated tools instead of shell pipelines.",
        )

    if _has_redirection(command):
        return PermissionDecision(
            risk=RiskLevel.level_3,
            allowed=False,
            reason="Shell redirection is rejected. Use write_file or another dedicated file tool instead.",
        )

    return None


def _reject_shell_filesystem_work(command: str) -> PermissionDecision | None:
    for segment in _shell_command_segments(command):
        root = _shell_segment_root(segment)

        if root in {"cd", "chdir", "pushd", "popd"}:
            continue

        if root in SHELL_FILESYSTEM_COMMAND_ROOTS:
            return PermissionDecision(
                risk=RiskLevel.level_3,
                allowed=False,
                reason=(
                    "Use dedicated workspace tools instead of shell for file inspection or edits "
                    "(list_files/read_file/search_text/find_files/git_diff/patch_file/create_file/write_file/delete_file)."
                ),
            )

    return None


def _reject_git_shell_work(command: str) -> PermissionDecision | None:
    for segment in _shell_command_segments(command):
        try:
            argv = shlex.split(segment, posix=sys.platform != "win32")
        except ValueError:
            return PermissionDecision(
                risk=RiskLevel.level_3,
                allowed=False,
                reason="Command could not be parsed safely.",
            )

        if not argv:
            continue

        root = _normalize_command_root(argv[0])

        if root != "git":
            continue

        subcommand = argv[1].lower() if len(argv) >= 2 else ""

        if subcommand in {"status", "diff"}:
            return PermissionDecision(
                risk=RiskLevel.level_3,
                allowed=False,
                reason="Use dedicated git_status/git_diff tools instead of shell git status/diff.",
            )

        return PermissionDecision(
            risk=RiskLevel.level_2,
            allowed=False,
            requires_approval=True,
            reason=f"Git command requires approval: {segment}",
        )

    return None


def _is_level_1_shell_command(command: str) -> bool:
    saw_command = False

    for segment in _shell_command_segments(command):
        root = _shell_segment_root(segment)

        if root in {"cd", "chdir", "pushd", "popd"}:
            continue

        try:
            segment_argv = shlex.split(segment, posix=sys.platform != "win32")
        except ValueError:
            return False

        if not segment_argv:
            continue

        if not _is_level_1_command(segment_argv):
            return False

        saw_command = True

    return saw_command


def _is_level_1_command(argv: list[str]) -> bool:
    normalized = tuple(
        _normalize_command_root(part) if index == 0 else part.lower()
        for index, part in enumerate(argv)
    )

    root = normalized[0]

    if root in ALLOWED_SIMPLE_ROOTS:
        return True

    return any(
        _argv_startswith(normalized, pattern)
        for pattern in ALLOWED_ARGV_PREFIXES
    )


def _shell_command_segments(command: str) -> list[str]:
    normalized = " ".join(command.split())

    normalized = normalized.replace("2>&1", "2>__CODE_AGENT_AMP__1")
    normalized = normalized.replace("1>&2", "1>__CODE_AGENT_AMP__2")

    segments = [normalized]

    for separator in ("&&", "||", ";", "&"):
        segments = [
            part.strip()
            for segment in segments
            for part in segment.split(separator)
            if part.strip()
        ]

    return [
        segment.replace("__CODE_AGENT_AMP__", "&")
        for segment in segments
    ]


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


def _has_command_substitution(command: str) -> bool:
    without_quotes = _strip_quoted_text(command)
    return "$(" in without_quotes or "`" in without_quotes


def _has_pipe(command: str) -> bool:
    without_quotes = _strip_quoted_text(command)
    return "|" in without_quotes


def _has_redirection(command: str) -> bool:
    without_quotes = _strip_quoted_text(command)

    cleaned = without_quotes.replace("2>&1", "").replace("1>&2", "")
    cleaned = re.sub(r"(?i)(^|\s)2>\s*(nul|/dev/null)(?=\s|$)", " ", cleaned)

    return bool(re.search(r"(^|\s)\d?>{1,2}(?!=|&)", cleaned))


def _strip_quoted_text(value: str) -> str:
    return re.sub(r"""(['"]).*?\1""", "", value)


def _command_contains_prefix(
    lower_command: str,
    prefixes: tuple[str, ...],
) -> bool:
    return any(
        segment.startswith(prefix)
        for segment in _shell_command_segments(lower_command)
        for prefix in prefixes
    )


def _is_dev_server_command(lower_command: str) -> bool:
    for segment in _shell_command_segments(lower_command):
        if segment.startswith(DEV_SERVER_COMMAND_PREFIXES):
            return True

        try:
            argv = shlex.split(segment, posix=sys.platform != "win32")
        except ValueError:
            argv = segment.split()

        normalized = [
            _normalize_command_root(part) if index == 0 else part.lower()
            for index, part in enumerate(argv)
        ]

        if normalized[:2] in (["npm", "run"], ["pnpm", "run"], ["yarn", "run"]):
            return len(normalized) >= 3 and normalized[2] in {"dev", "start", "serve"}

        if normalized[:1] in (["pnpm"], ["yarn"]) and len(normalized) >= 2:
            return normalized[1] in {"dev", "start", "serve"}

        if normalized[:2] in (["npx", "vite"], ["npx", "next"], ["npx", "nuxt"], ["npx", "astro"]):
            return not any(
                arg in {"build", "--version", "test", "lint", "check"}
                for arg in normalized[2:]
            )

        if normalized and normalized[0] in {"vite", "next", "nuxt", "astro"}:
            return not any(
                arg in {"build", "--version", "test", "lint", "check"}
                for arg in normalized[1:]
            )

    return False


def _argv_startswith(
    argv: tuple[str, ...],
    prefix: tuple[str, ...],
) -> bool:
    return len(argv) >= len(prefix) and argv[: len(prefix)] == prefix


def normalize_path_arg(path: str | Path) -> str:
    return str(path).replace("\\", "/")
