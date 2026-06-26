from __future__ import annotations

import shutil
import shlex
import subprocess
from pathlib import Path

from code_agent.models import CommandSpec, PermissionDecision, RiskLevel
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


DEFAULT_COMMANDS = {
    "pytest": CommandSpec(
        name="pytest",
        argv=["python", "-m", "pytest"],
        description="使用 pytest 运行 Python 测试。",
    ),
    "npm_test": CommandSpec(
        name="npm_test",
        argv=["npm", "test"],
        description="运行 npm test 脚本。",
    ),
    "npm_lint": CommandSpec(
        name="npm_lint",
        argv=["npm", "run", "lint"],
        description="运行 npm lint 脚本。",
    ),
    "npm_build": CommandSpec(
        name="npm_build",
        argv=["npm", "run", "build"],
        description="运行 npm build 脚本。",
    ),
    "pnpm_test": CommandSpec(
        name="pnpm_test",
        argv=["pnpm", "test"],
        description="运行 pnpm test 脚本。",
    ),
    "pnpm_lint": CommandSpec(
        name="pnpm_lint",
        argv=["pnpm", "lint"],
        description="运行 pnpm lint 脚本。",
    ),
    "pnpm_build": CommandSpec(
        name="pnpm_build",
        argv=["pnpm", "build"],
        description="运行 pnpm build 脚本。",
    ),
    "uv_pytest": CommandSpec(
        name="uv_pytest",
        argv=["uv", "run", "pytest"],
        description="通过 uv 运行 pytest。",
    ),
}


def available_commands(workspace: Workspace) -> dict[str, CommandSpec]:
    commands: dict[str, CommandSpec] = {}
    root = workspace.root

    if (root / "pyproject.toml").exists() or (root / "pytest.ini").exists():
        commands["pytest"] = DEFAULT_COMMANDS["pytest"]

    if (root / "package.json").exists():
        commands["npm_test"] = DEFAULT_COMMANDS["npm_test"]
        commands["npm_lint"] = DEFAULT_COMMANDS["npm_lint"]
        commands["npm_build"] = DEFAULT_COMMANDS["npm_build"]

    if (root / "pnpm-lock.yaml").exists():
        commands["pnpm_test"] = DEFAULT_COMMANDS["pnpm_test"]
        commands["pnpm_lint"] = DEFAULT_COMMANDS["pnpm_lint"]
        commands["pnpm_build"] = DEFAULT_COMMANDS["pnpm_build"]

    if (root / "uv.lock").exists():
        commands["uv_pytest"] = DEFAULT_COMMANDS["uv_pytest"]

    return commands


def approval_required(risk: RiskLevel, reason: str) -> str:
    return f"APPROVAL_REQUIRED[{risk.value}]: {reason}"


def rejected(risk: RiskLevel, reason: str) -> str:
    return f"REJECTED[{risk.value}]: {reason}"


def allowed(risk: RiskLevel, reason: str) -> str:
    return f"ALLOWED[{risk.value}]: {reason}"


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
        try:
            argv = shlex.split(stripped)
        except ValueError:
            argv = None
        return (
            PermissionDecision(
                risk=RiskLevel.level_2,
                allowed=False,
                requires_approval=True,
                reason=f"命令需要确认: {command}",
            ),
            argv,
        )

    commands = available_commands(workspace)
    if stripped in commands:
        spec = commands[stripped]
        return (
            PermissionDecision(
                risk=spec.risk,
                allowed=True,
                reason=f"已允许命名验证命令: {stripped}",
            ),
            spec.argv,
        )

    for spec in commands.values():
        rendered = " ".join(spec.argv)
        if lower == rendered.lower():
            return (
                PermissionDecision(
                    risk=spec.risk,
                    allowed=True,
                    reason=f"已允许验证命令: {rendered}",
                ),
                spec.argv,
            )

    try:
        argv = shlex.split(stripped)
    except ValueError:
        argv = []

    if argv and argv[0] in {"pytest"}:
        return (
            PermissionDecision(
                risk=RiskLevel.level_1,
                allowed=True,
                reason=f"已允许验证命令: {command}",
            ),
            argv,
        )

    return (
        PermissionDecision(
            risk=RiskLevel.level_2,
            allowed=False,
            requires_approval=True,
            reason=f"命令不在安全验证命令白名单中: {command}",
        ),
        None,
    )


def run_argv(argv: list[str], cwd: Path, *, timeout: int) -> subprocess.CompletedProcess[str]:
    executable = shutil.which(argv[0])
    if executable is None:
        raise FileNotFoundError(f"找不到可执行文件: {argv[0]}")

    return subprocess.run(
        argv,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        shell=False,
    )


def command_risk(command: CommandSpec) -> RiskLevel:
    return command.risk
