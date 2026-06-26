from __future__ import annotations

import subprocess

from langchain_core.tools import tool

from code_agent.services.patcher import replace_exact_once
from code_agent.services.summarizer import truncate
from code_agent.services.workspace import Workspace, WorkspaceError
from code_agent.tools.safety import approval_required, allowed, classify_file_operation, rejected


APPROVAL_TOKEN = "approved"


def _error(exc: Exception) -> str:
    return f"ERROR: {exc}"


def _format_decision_prefix(decision) -> str:
    if decision.risk.value == "level_1":
        return allowed(decision.risk, decision.reason)
    return decision.reason


def _git_diff_for(workspace: Workspace, path: str) -> str:
    try:
        resolved = workspace.resolve(path)
        rel = workspace.relative(resolved)
        result = subprocess.run(
            ["git", "diff", "--", rel],
            cwd=workspace.root,
            capture_output=True,
            text=True,
            timeout=10,
            shell=False,
        )
        diff = truncate(result.stdout + result.stderr, 4000)
        if diff:
            return "\n\nRecorded diff:\n" + diff
        if resolved.exists():
            return "\n\nRecorded diff: file is new or unchanged in git diff; use git_status for details."
        return "\n\nRecorded diff: no diff available."
    except (WorkspaceError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return f"\n\nDiff recording failed: {exc}"


def build_read_file_tool(workspace: Workspace):
    @tool
    def read_file(path: str) -> str:
        """Read a text file inside the workspace. Sensitive, binary, and large files are rejected."""
        try:
            return workspace.read_text(path)
        except WorkspaceError as exc:
            return _error(exc)

    return read_file


def build_list_files_tool(workspace: Workspace):
    @tool
    def list_files(path: str = ".") -> str:
        """List direct children under a workspace directory."""
        try:
            dir_path = workspace.resolve(path)
            if not dir_path.exists():
                return f"ERROR: Directory not found: {path}"
            if not dir_path.is_dir():
                return f"ERROR: Not a directory: {path}"

            lines = []
            for child in sorted(dir_path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
                if workspace.is_excluded(child):
                    continue
                suffix = "/" if child.is_dir() else ""
                lines.append(child.name + suffix)
            return "\n".join(lines)
        except WorkspaceError as exc:
            return _error(exc)

    return list_files


def build_get_file_tree_tool(workspace: Workspace):
    @tool
    def get_file_tree(path: str = ".", max_entries: int = 200) -> str:
        """Return a bounded file tree for the workspace path."""
        try:
            max_entries = min(max(max_entries, 1), 1000)
            return "\n".join(workspace.iter_tree(path, max_entries=max_entries))
        except WorkspaceError as exc:
            return _error(exc)

    return get_file_tree


def build_patch_file_tool(workspace: Workspace):
    @tool
    def patch_file(path: str, old: str, new: str, approval_token: str | None = None) -> str:
        """Replace one exact text block in a file. The old text must appear exactly once."""
        try:
            decision = classify_file_operation(workspace, "patch_file", path)
            if decision.risk.value == "level_3":
                return rejected(decision.risk, decision.reason)
            if decision.requires_approval and approval_token != APPROVAL_TOKEN:
                return approval_required(decision.risk, decision.reason)
            result = replace_exact_once(workspace, path, old, new)
            if not result.changed:
                return (
                    f"ERROR: Old text must appear exactly once in {path}; "
                    f"found {result.old_count} matches."
                )
            return f"{_format_decision_prefix(decision)}\nPatched {path}{_git_diff_for(workspace, path)}"
        except WorkspaceError as exc:
            return _error(exc)

    return patch_file


def build_create_file_tool(workspace: Workspace):
    @tool
    def create_file(path: str, content: str, approval_token: str | None = None) -> str:
        """Create a new text file inside the workspace. Existing files are rejected."""
        try:
            decision = classify_file_operation(workspace, "create_file", path)
            if decision.risk.value == "level_3":
                return rejected(decision.risk, decision.reason)
            if decision.requires_approval and approval_token != APPROVAL_TOKEN:
                return approval_required(decision.risk, decision.reason)
            workspace.write_text(path, content, overwrite=False)
            return f"{_format_decision_prefix(decision)}\nCreated {path}{_git_diff_for(workspace, path)}"
        except WorkspaceError as exc:
            return _error(exc)

    return create_file


def build_write_file_tool(workspace: Workspace):
    @tool
    def write_file(path: str, content: str, approval_token: str | None = None) -> str:
        """Overwrite a small text file inside the workspace. Prefer patch_file for edits."""
        try:
            decision = classify_file_operation(workspace, "write_file", path)
            if decision.risk.value == "level_3":
                return rejected(decision.risk, decision.reason)
            if decision.requires_approval and approval_token != APPROVAL_TOKEN:
                return approval_required(decision.risk, decision.reason)
            existing = workspace.resolve(path)
            if existing.exists() and existing.stat().st_size > 20_000:
                if approval_token != APPROVAL_TOKEN:
                    return approval_required(decision.risk, "Overwriting a file larger than 20KB requires confirmation.")
            workspace.write_text(path, content, overwrite=True)
            return f"{_format_decision_prefix(decision)}\nWrote {path}{_git_diff_for(workspace, path)}"
        except WorkspaceError as exc:
            return _error(exc)

    return write_file


def build_delete_file_tool(workspace: Workspace):
    @tool
    def delete_file(path: str, approval_token: str | None = None) -> str:
        """Request deletion of a workspace file. This MVP never deletes without human approval."""
        try:
            decision = classify_file_operation(workspace, "delete_file", path)
            if decision.risk.value == "level_3":
                return rejected(decision.risk, decision.reason)
            if approval_token != APPROVAL_TOKEN:
                return approval_required(decision.risk, decision.reason)
            file_path = workspace.resolve(path)
            if not file_path.exists():
                return f"ERROR: File not found: {path}"
            if not file_path.is_file():
                return f"ERROR: Not a file: {path}"
            before = _git_diff_for(workspace, path)
            file_path.unlink()
            return f"{allowed(decision.risk, decision.reason)}\nDeleted {path}{before}{_git_diff_for(workspace, path)}"
        except WorkspaceError as exc:
            return _error(exc)

    return delete_file
