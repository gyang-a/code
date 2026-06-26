from __future__ import annotations

import shlex
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
    "pnpm add",
    "pnpm install",
    "yarn add",
    "yarn install",
    "pip install",
    "python -m pip install",
    "uv add",
    "uv pip install",
    "docker compose up",
    "docker-compose up",
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
    "rg",
    "grep",
    "ls",
    "dir",
    "pwd",
    "cat",
    "type",
}

LEVEL_1_COMMAND_PATTERNS = (
    ("npm", "test"),
    ("npm", "run"),
    ("pnpm", "test"),
    ("pnpm", "lint"),
    ("pnpm", "build"),
    ("yarn", "test"),
    ("yarn", "lint"),
    ("yarn", "build"),
    ("uv", "run"),
    ("python", "-m"),
    ("git", "status"),
    ("git", "diff"),
)

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


def approval_required(risk: RiskLevel, reason: str) -> str:
    return f"APPROVAL_REQUIRED[{risk.value}]: {reason}"


def rejected(risk: RiskLevel, reason: str) -> str:
    return f"REJECTED[{risk.value}]: {reason}"


def allowed(risk: RiskLevel, reason: str) -> str:
    return f"ALLOWED[{risk.value}]: {reason}"


def describe_permission_policy() -> str:
    return "\n".join(
        [
            "Permission rules:",
            "- Level 0: 只读工具直接允许，例如 read_file/search_text/git_diff。",
            "- Level 1: 低风险编辑或常见测试/构建/查看命令，直接在 sandbox 中执行。",
            "- Level 2: 安装依赖、Docker、删除文件、未知 shell 命令等，需要用户确认。",
            "- Level 3: rm -rf、sudo、chmod 777、curl | bash、git reset --hard、访问 .env/.ssh，直接拒绝。",
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
                    reason=f"{tool_name} 拒绝访问敏感路径: {workspace.relative(resolved)}",
                )
        return PermissionDecision(
            risk=RiskLevel.level_0,
            allowed=True,
            reason=f"{tool_name} 是只读工具，直接允许。",
        )

    if tool_name in WRITE_TOOLS:
        path = args.get("path")
        if not isinstance(path, str):
            return PermissionDecision(
                risk=RiskLevel.level_3,
                allowed=False,
                reason=f"{tool_name} 缺少 path 参数，拒绝执行。",
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
        reason=f"未知工具需要确认: {tool_name}",
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
            reason=f"{operation} 拒绝访问敏感路径: {normalized}",
        )

    if operation == "delete_file":
        return PermissionDecision(
            risk=RiskLevel.level_2,
            allowed=False,
            requires_approval=True,
            reason=f"删除文件需要确认: {normalized}",
        )

    if name in LEVEL_2_FILES:
        return PermissionDecision(
            risk=RiskLevel.level_2,
            allowed=False,
            requires_approval=True,
            reason=f"修改 {normalized} 需要确认。",
        )

    if operation == "create_file" and not (
        normalized.startswith("src/") or normalized.startswith("tests/")
    ):
        return PermissionDecision(
            risk=RiskLevel.level_2,
            allowed=False,
            requires_approval=True,
            reason=f"在 src/ 或 tests/ 之外创建文件需要确认: {normalized}",
        )

    if operation == "write_file" and not (
        normalized.startswith("src/") or normalized.startswith("tests/")
    ):
        return PermissionDecision(
            risk=RiskLevel.level_2,
            allowed=False,
            requires_approval=True,
            reason=f"覆盖 src/ 或 tests/ 之外的文件需要确认: {normalized}",
        )

    return PermissionDecision(
        risk=RiskLevel.level_1,
        allowed=True,
        reason=f"{operation} 已允许，属于低风险工作区编辑: {normalized}",
    )


def classify_command(command: str, workspace: Workspace) -> tuple[PermissionDecision, list[str] | None]:
    stripped = command.strip()
    lower = " ".join(stripped.lower().split())

    try:
        argv = shlex.split(stripped)
    except ValueError:
        return (
            PermissionDecision(
                risk=RiskLevel.level_3,
                allowed=False,
                reason=f"命令无法安全解析: {command}",
            ),
            None,
        )

    if not argv:
        return (
            PermissionDecision(
                risk=RiskLevel.level_3,
                allowed=False,
                reason="空命令被拒绝。",
            ),
            None,
        )

    if any(marker in lower for marker in LEVEL_3_COMMAND_MARKERS):
        return (
            PermissionDecision(
                risk=RiskLevel.level_3,
                allowed=False,
                reason=f"命令被安全策略禁止: {command}",
            ),
            None,
        )

    if "|" in lower and "bash" in lower and ("curl" in lower or "wget" in lower):
        return (
            PermissionDecision(
                risk=RiskLevel.level_3,
                allowed=False,
                reason=f"命令被安全策略禁止: {command}",
            ),
            None,
        )

    if any(lower.startswith(prefix) for prefix in LEVEL_2_COMMAND_PREFIXES):
        return (
            PermissionDecision(
                risk=RiskLevel.level_2,
                allowed=False,
                requires_approval=True,
                reason=f"命令需要确认: {command}",
            ),
            argv,
        )

    if _is_level_1_command(argv):
        return (
            PermissionDecision(
                risk=RiskLevel.level_1,
                allowed=True,
                reason=f"已允许低风险 shell 命令: {command}",
            ),
            argv,
        )

    return (
        PermissionDecision(
            risk=RiskLevel.level_2,
            allowed=False,
            requires_approval=True,
            reason=f"未知或中风险 shell 命令需要确认: {command}",
        ),
        argv,
    )


def _is_level_1_command(argv: list[str]) -> bool:
    root = Path(argv[0]).name.lower()
    if root.endswith(".cmd") or root.endswith(".exe"):
        root = root.rsplit(".", 1)[0]

    if root not in LEVEL_1_COMMAND_ROOTS:
        return False

    if root in {"rg", "grep", "ls", "dir", "pwd", "cat", "type", "pytest", "ruff", "mypy", "node"}:
        return True

    command_prefix = tuple(part.lower() for part in argv[:2])
    return any(command_prefix == pattern for pattern in LEVEL_1_COMMAND_PATTERNS)


def normalize_path_arg(path: str | Path) -> str:
    return str(path).replace("\\", "/")
