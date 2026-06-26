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
        description="Run Python tests with pytest.",
    ),
    "python_compile": CommandSpec(
        name="python_compile",
        argv=["python", "-m", "compileall", "-q", "."],
        description="Compile Python files for syntax validation.",
    ),
    "npm_test": CommandSpec(
        name="npm_test",
        argv=["npm", "test"],
        description="Run npm test script.",
    ),
    "npm_lint": CommandSpec(
        name="npm_lint",
        argv=["npm", "run", "lint"],
        description="Run npm lint script.",
    ),
    "npm_build": CommandSpec(
        name="npm_build",
        argv=["npm", "run", "build"],
        description="Run npm build script.",
    ),
    "pnpm_test": CommandSpec(
        name="pnpm_test",
        argv=["pnpm", "test"],
        description="Run pnpm test script.",
    ),
    "pnpm_lint": CommandSpec(
        name="pnpm_lint",
        argv=["pnpm", "lint"],
        description="Run pnpm lint script.",
    ),
    "pnpm_build": CommandSpec(
        name="pnpm_build",
        argv=["pnpm", "build"],
        description="Run pnpm build script.",
    ),
    "uv_pytest": CommandSpec(
        name="uv_pytest",
        argv=["uv", "run", "pytest"],
        description="Run pytest through uv.",
    ),
}


def available_commands(workspace: Workspace) -> dict[str, CommandSpec]:
    commands: dict[str, CommandSpec] = {}
    root = workspace.root

    if (root / "pyproject.toml").exists() or (root / "pytest.ini").exists():
        commands["python_compile"] = DEFAULT_COMMANDS["python_compile"]
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
            reason=f"{operation} refused for sensitive path: {normalized}",
        )

    if operation == "delete_file":
        return PermissionDecision(
            risk=RiskLevel.level_2,
            allowed=False,
            requires_approval=True,
            reason=f"Deleting files requires confirmation: {normalized}",
        )

    if name in LEVEL_2_FILES:
        return PermissionDecision(
            risk=RiskLevel.level_2,
            allowed=False,
            requires_approval=True,
            reason=f"Modifying {normalized} requires confirmation.",
        )

    if operation == "create_file" and not (
        normalized.startswith("src/") or normalized.startswith("tests/")
    ):
        return PermissionDecision(
            risk=RiskLevel.level_2,
            allowed=False,
            requires_approval=True,
            reason=f"Creating files outside src/ or tests/ requires confirmation: {normalized}",
        )

    if operation == "write_file" and not (
        normalized.startswith("src/") or normalized.startswith("tests/")
    ):
        return PermissionDecision(
            risk=RiskLevel.level_2,
            allowed=False,
            requires_approval=True,
            reason=f"Overwriting files outside src/ or tests/ requires confirmation: {normalized}",
        )

    return PermissionDecision(
        risk=RiskLevel.level_1,
        allowed=True,
        reason=f"{operation} allowed as a low-risk workspace edit: {normalized}",
    )


def classify_command(command: str, workspace: Workspace) -> tuple[PermissionDecision, list[str] | None]:
    stripped = command.strip()
    lower = " ".join(stripped.lower().split())

    if any(marker in lower for marker in LEVEL_3_COMMAND_MARKERS):
        return (
            PermissionDecision(
                risk=RiskLevel.level_3,
                allowed=False,
                reason=f"Command is forbidden by safety policy: {command}",
            ),
            None,
        )

    if "|" in lower and "bash" in lower and ("curl" in lower or "wget" in lower):
        return (
            PermissionDecision(
                risk=RiskLevel.level_3,
                allowed=False,
                reason=f"Command is forbidden by safety policy: {command}",
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
                reason=f"Command requires confirmation: {command}",
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
                reason=f"Named validation command allowed: {stripped}",
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
                    reason=f"Validation command allowed: {rendered}",
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
                reason=f"Validation command allowed: {command}",
            ),
            argv,
        )

    return (
        PermissionDecision(
            risk=RiskLevel.level_2,
            allowed=False,
            requires_approval=True,
            reason=f"Command is not in the safe validation allowlist: {command}",
        ),
        None,
    )


def run_argv(argv: list[str], cwd: Path, *, timeout: int) -> subprocess.CompletedProcess[str]:
    executable = shutil.which(argv[0])
    if executable is None:
        raise FileNotFoundError(f"Executable not found: {argv[0]}")

    return subprocess.run(
        argv,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        shell=False,
    )


def command_risk(command: CommandSpec) -> RiskLevel:
    return command.risk
