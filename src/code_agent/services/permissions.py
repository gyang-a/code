from __future__ import annotations

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

READ_ONLY_TOOLS = {
    "list_files",
    "read_file",
    "search_text",
    "find_files",
    "git_status",
    "git_diff",
    "skills_list",
    "skill_view",
}

WRITE_TOOLS = {
    "patch_file",
    "create_file",
    "write_file",
    "delete_file",
}


def rejected(risk: RiskLevel, reason: str) -> str:
    return f"REJECTED[{risk.value}]: {reason}"


def describe_permission_policy() -> str:
    return "\n".join(
        [
            "Permission rules:",
            "- Level 0: read-only project tools are allowed.",
            "- Level 1: low-risk project file edits are allowed.",
            "- Level 2: deletes, full-file overwrites, dependency/config edits, and unknown tools require approval.",
            "- Level 3: sensitive paths and excluded paths are rejected.",
            "- shell_command runs in the Windows read-only sandbox by default.",
            "- A denied command may be retried exactly once with workspace-write after approval.",
            "- A process-pipe denial may be retried exactly once with danger-full-access after separate approval.",
        ]
    )


def classify_tool_call(
    workspace: Workspace,
    tool_name: str,
    args: dict[str, Any],
) -> PermissionDecision:
    if tool_name == "shell_command":
        workdir = args.get("workdir", ".")
        if not isinstance(workdir, str):
            return PermissionDecision(
                risk=RiskLevel.level_3,
                allowed=False,
                reason="shell_command has an invalid workdir.",
            )
        try:
            resolved = workspace.resolve(workdir)
        except WorkspaceError as exc:
            return PermissionDecision(
                risk=RiskLevel.level_3,
                allowed=False,
                reason=str(exc),
            )

        requested = args.get("sandbox_permissions")
        if requested == "workspace-write":
            justification = str(args.get("justification") or "").strip()
            return PermissionDecision(
                risk=RiskLevel.level_2,
                allowed=False,
                requires_approval=True,
                reason=(
                    f"PowerShell requests one-shot workspace write access in "
                    f"{workspace.relative(resolved)}: {justification or 'no justification provided'}"
                ),
            )
        if requested == "danger-full-access":
            justification = str(args.get("justification") or "").strip()
            return PermissionDecision(
                risk=RiskLevel.level_2,
                allowed=False,
                requires_approval=True,
                reason=(
                    "PowerShell requests one-shot unrestricted Windows execution. "
                    "The command can read or write anywhere accessible to the current user: "
                    f"{justification or 'no justification provided'}"
                ),
            )
        return PermissionDecision(
            risk=RiskLevel.level_0,
            allowed=True,
            reason="PowerShell runs with a Windows read-only restricted token.",
        )

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
            reason=f"{tool_name} is a read-only project tool.",
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
        reason=f"{operation} is a low-risk project edit: {normalized}",
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
