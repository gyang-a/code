from __future__ import annotations

from dataclasses import dataclass

from code_agent.services.workspace import Workspace, WorkspaceError


@dataclass(frozen=True)
class PatchResult:
    path: str
    old_count: int
    changed: bool


def replace_exact_once(workspace: Workspace, path: str, old: str, new: str) -> PatchResult:
    if old == "":
        raise WorkspaceError("Old text cannot be empty.")

    content = workspace.read_text(path)
    count = content.count(old)
    if count != 1:
        return PatchResult(path=path, old_count=count, changed=False)

    updated = content.replace(old, new, 1)
    workspace.write_text(path, updated, overwrite=True)
    return PatchResult(path=path, old_count=count, changed=True)
