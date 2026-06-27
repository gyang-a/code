from __future__ import annotations

import subprocess

from langchain_core.tools import tool

from code_agent.services.patcher import replace_exact_once
from code_agent.services.summarizer import truncate
from code_agent.services.workspace import Workspace, WorkspaceError
from code_agent.tools.safety import allowed, classify_tool_call, rejected
from code_agent.tools.schemas import (
    CreateFileInput,
    DeleteFileInput,
    FileTreeInput,
    ListFilesInput,
    PatchFileInput,
    ReadFileInput,
    WriteFileInput,
)


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
            encoding="utf-8",
            errors="replace",
            timeout=10,
            shell=False,
        )
        if result.returncode not in (0, 1):
            return "\n\nDiff: git diff is unavailable for this workspace."
        output = result.stdout if result.returncode == 0 else result.stdout + result.stderr
        diff = truncate(output, 4000)
        if diff:
            return "\n\nDiff:\n" + diff
        if resolved.exists():
            return "\n\nDiff: no tracked diff yet; use git_status if needed."
        return "\n\nDiff: no diff available."
    except (WorkspaceError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return f"\n\nDiff failed: {exc}"


def build_read_file_tool(
    workspace: Workspace,
    *,
    default_max_lines: int | None = None,
    output_limit: int | None = None,
):
    @tool(args_schema=ReadFileInput)
    def read_file(path: str, start_line: int = 1, max_lines: int | None = None) -> str:
        """Read a bounded text window from a file in the workspace."""
        try:
            decision = classify_tool_call(workspace, "read_file", {"path": path})
            if decision.risk.value == "level_3":
                return rejected(decision.risk, decision.reason)
            requested_lines = default_max_lines if max_lines is None else max_lines
            if requested_lines is not None:
                requested_lines = min(max(requested_lines, 1), 200)
            content = workspace.read_text_window(path, start_line=start_line, max_lines=requested_lines)
            return truncate(content, output_limit) if output_limit else content
        except WorkspaceError as exc:
            return _error(exc)

    return read_file


def build_list_files_tool(workspace: Workspace):
    @tool(args_schema=ListFilesInput)
    def list_files(path: str = ".") -> str:
        """List direct children of a workspace directory."""
        try:
            decision = classify_tool_call(workspace, "list_files", {"path": path})
            if decision.risk.value == "level_3":
                return rejected(decision.risk, decision.reason)
            dir_path = workspace.resolve(path)
            if not dir_path.exists():
                return f"ERROR: Directory does not exist: {path}"
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
    @tool(args_schema=FileTreeInput)
    def get_file_tree(path: str = ".", max_entries: int = 200) -> str:
        """Return a bounded file tree for a workspace path."""
        try:
            decision = classify_tool_call(workspace, "get_file_tree", {"path": path})
            if decision.risk.value == "level_3":
                return rejected(decision.risk, decision.reason)
            max_entries = min(max(max_entries, 1), 1000)
            return "\n".join(workspace.iter_tree(path, max_entries=max_entries))
        except WorkspaceError as exc:
            return _error(exc)

    return get_file_tree


def build_patch_file_tool(workspace: Workspace):
    @tool(args_schema=PatchFileInput)
    def patch_file(path: str, old: str, new: str) -> str:
        """Replace one exact text block in a workspace file."""
        try:
            decision = classify_tool_call(workspace, "patch_file", {"path": path})
            if decision.risk.value == "level_3":
                return rejected(decision.risk, decision.reason)
            result = replace_exact_once(workspace, path, old, new)
            if not result.changed:
                return (
                    f"ERROR: old text must appear exactly once in {path}; "
                    f"found {result.old_count} occurrences."
                )
            return f"{_format_decision_prefix(decision)}\nPatched {path}{_git_diff_for(workspace, path)}"
        except WorkspaceError as exc:
            return _error(exc)

    return patch_file


def build_create_file_tool(workspace: Workspace):
    @tool(args_schema=CreateFileInput)
    def create_file(path: str, content: str) -> str:
        """Create a new text file in the workspace."""
        try:
            decision = classify_tool_call(workspace, "create_file", {"path": path})
            if decision.risk.value == "level_3":
                return rejected(decision.risk, decision.reason)
            workspace.write_text(path, content, overwrite=False)
            return f"{_format_decision_prefix(decision)}\nCreated {path}{_git_diff_for(workspace, path)}"
        except WorkspaceError as exc:
            return _error(exc)

    return create_file


def build_write_file_tool(workspace: Workspace):
    @tool(args_schema=WriteFileInput)
    def write_file(path: str, content: str) -> str:
        """Overwrite a text file in the workspace."""
        try:
            decision = classify_tool_call(workspace, "write_file", {"path": path})
            if decision.risk.value == "level_3":
                return rejected(decision.risk, decision.reason)
            workspace.write_text(path, content, overwrite=True)
            return f"{_format_decision_prefix(decision)}\nWrote {path}{_git_diff_for(workspace, path)}"
        except WorkspaceError as exc:
            return _error(exc)

    return write_file


def build_delete_file_tool(workspace: Workspace):
    @tool(args_schema=DeleteFileInput)
    def delete_file(path: str) -> str:
        """Delete a workspace file."""
        try:
            decision = classify_tool_call(workspace, "delete_file", {"path": path})
            if decision.risk.value == "level_3":
                return rejected(decision.risk, decision.reason)
            file_path = workspace.resolve(path)
            if not file_path.exists():
                return f"ERROR: File does not exist: {path}"
            if not file_path.is_file():
                return f"ERROR: Not a file: {path}"
            before = _git_diff_for(workspace, path)
            file_path.unlink()
            return f"{allowed(decision.risk, decision.reason)}\nDeleted {path}{before}{_git_diff_for(workspace, path)}"
        except WorkspaceError as exc:
            return _error(exc)

    return delete_file
