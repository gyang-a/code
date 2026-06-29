from __future__ import annotations

import subprocess
from pathlib import Path

from langchain_core.tools import tool

from code_agent.services.patcher import replace_exact_once
from code_agent.services.summarizer import truncate
from code_agent.services.workspace import Workspace, WorkspaceError
from code_agent.tools.safety import RiskLevel, classify_tool_call, rejected
from code_agent.tools.schemas import (
    CreateFileInput,
    DeleteFileInput,
    GitDiffInput,
    GitStatusInput,
    ListFilesInput,
    PatchFileInput,
    ReadFileInput,
    WriteFileInput,
)


DEFAULT_READ_MAX_LINES = 120
DEFAULT_READ_OUTPUT_LIMIT = 12_000
DEFAULT_DIFF_OUTPUT_LIMIT = 8_000
MAX_WRITE_FILE_BYTES = 200_000


def _error(exc: Exception) -> str:
    return f"ERROR: {exc}"


def _decision_rejected(workspace: Workspace, tool_name: str, payload: dict) -> str | None:
    decision = classify_tool_call(workspace, tool_name, payload)
    if decision.risk == RiskLevel.level_3:
        return rejected(decision.risk, decision.reason)

    return None


def _run_git(
    workspace: Workspace,
    args: list[str],
    *,
    timeout: int = 10,
    output_limit: int = DEFAULT_DIFF_OUTPUT_LIMIT,
) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=workspace.root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        shell=False,
    )

    output = (result.stdout + result.stderr).strip()
    if result.returncode not in (0, 1):
        if "not a git repository" in output.lower():
            return None
        return f"ERROR: git {' '.join(args)} failed.\n{truncate(output, output_limit)}"

    return truncate(output, output_limit)


def _git_diff_for(workspace: Workspace, path: str) -> str:
    try:
        resolved = workspace.resolve(path)
        rel = workspace.relative(resolved)

        diff = _run_git(workspace, ["diff", "--", rel])
        if diff is None:
            return ""
        if diff and not diff.startswith("ERROR:"):
            return "\n\nDiff:\n" + diff

        # git diff does not show untracked files, so include concise status.
        status = _run_git(workspace, ["status", "--short", "--", rel], output_limit=2_000)
        if status is None:
            return ""
        if status:
            return "\n\nGit status:\n" + status

        if resolved.exists():
            return "\n\nDiff: no tracked diff."
        return "\n\nDiff: no diff available."

    except (WorkspaceError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return f"\n\nDiff failed: {exc}"


def _ensure_small_content(content: str) -> str | None:
    size = len(content.encode("utf-8", errors="replace"))
    if size > MAX_WRITE_FILE_BYTES:
        return f"ERROR: Refusing to write large file content over {MAX_WRITE_FILE_BYTES} bytes."
    return None


def build_read_file_tool(
    workspace: Workspace,
    *,
    default_max_lines: int | None = DEFAULT_READ_MAX_LINES,
    output_limit: int | None = DEFAULT_READ_OUTPUT_LIMIT,
):
    @tool(args_schema=ReadFileInput)
    def read_file(path: str, start_line: int = 1, max_lines: int | None = None) -> str:
        """Read a bounded text window from one workspace file. Prefer search_text/find_files first; do not bulk-read files for broad reviews."""
        try:
            rejection = _decision_rejected(workspace, "read_file", {"path": path})
            if rejection:
                return rejection

            requested_lines = default_max_lines if max_lines is None else max_lines
            if requested_lines is None:
                requested_lines = DEFAULT_READ_MAX_LINES

            requested_lines = min(max(requested_lines, 1), 200)

            content = workspace.read_text_window(
                path,
                start_line=start_line,
                max_lines=requested_lines,
            )

            return truncate(content, output_limit) if output_limit else content

        except WorkspaceError as exc:
            return _error(exc)

    return read_file


def build_list_files_tool(workspace: Workspace):
    @tool(args_schema=ListFilesInput)
    def list_files(path: str = ".", max_entries: int = 200) -> str:
        """List direct visible children of a workspace directory. Use find_files/search_text for precise locating."""
        try:
            rejection = _decision_rejected(workspace, "list_files", {"path": path})
            if rejection:
                return rejection

            dir_path = workspace.resolve(path)
            if not dir_path.exists():
                return f"ERROR: Directory does not exist: {path}"
            if not dir_path.is_dir():
                return f"ERROR: Not a directory: {path}"

            max_entries = min(max(max_entries, 1), 1000)

            lines: list[str] = []
            for child in sorted(dir_path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
                if workspace.is_excluded(child) or workspace.is_sensitive(child):
                    continue

                suffix = "/" if child.is_dir() else ""
                lines.append(child.name + suffix)

                if len(lines) >= max_entries:
                    lines.append(f"... truncated after {max_entries} visible entries ...")
                    break

            if not lines:
                rel = workspace.relative(dir_path)
                return f"EMPTY: {rel} has no visible entries."

            return "\n".join(lines)

        except WorkspaceError as exc:
            return _error(exc)

    return list_files


def build_patch_file_tool(workspace: Workspace):
    @tool(args_schema=PatchFileInput)
    def patch_file(path: str, old: str, new: str) -> str:
        """Replace one exact text block in a workspace file. Prefer this over write_file for edits."""
        try:
            rejection = _decision_rejected(workspace, "patch_file", {"path": path})
            if rejection:
                return rejection

            result = replace_exact_once(workspace, path, old, new)

            if result.changed:
                return f"Patched {path}{_git_diff_for(workspace, path)}"

            if result.old_count == 0:
                return (
                    f"ERROR: old text was not found in {path}. "
                    "Read the current file window again and retry with an exact block."
                )

            return (
                f"ERROR: old text appears {result.old_count} times in {path}. "
                "Use a larger surrounding block so the replacement is unique."
            )

        except WorkspaceError as exc:
            return _error(exc)

    return patch_file


def build_create_file_tool(workspace: Workspace):
    @tool(args_schema=CreateFileInput)
    def create_file(path: str, content: str) -> str:
        """Create a new text file in the workspace. Fails if the file already exists."""
        try:
            rejection = _decision_rejected(workspace, "create_file", {"path": path})
            if rejection:
                return rejection

            size_error = _ensure_small_content(content)
            if size_error:
                return size_error

            workspace.write_text(path, content, overwrite=False)
            return f"Created {path}{_git_diff_for(workspace, path)}"

        except WorkspaceError as exc:
            return _error(exc)

    return create_file


def build_write_file_tool(workspace: Workspace):
    @tool(args_schema=WriteFileInput)
    def write_file(path: str, content: str) -> str:
        """Overwrite a small text file in the workspace. Prefer patch_file for normal edits."""
        try:
            rejection = _decision_rejected(workspace, "write_file", {"path": path})
            if rejection:
                return rejection

            size_error = _ensure_small_content(content)
            if size_error:
                return size_error

            workspace.write_text(path, content, overwrite=True)
            return f"Wrote {path}{_git_diff_for(workspace, path)}"

        except WorkspaceError as exc:
            return _error(exc)

    return write_file


def build_delete_file_tool(workspace: Workspace):
    @tool(args_schema=DeleteFileInput)
    def delete_file(path: str) -> str:
        """Delete a workspace file. This should normally require approval in the safety policy."""
        try:
            rejection = _decision_rejected(workspace, "delete_file", {"path": path})
            if rejection:
                return rejection

            file_path = workspace.resolve(path)
            if not file_path.exists():
                return f"ERROR: File does not exist: {path}"
            if not file_path.is_file():
                return f"ERROR: Not a file: {path}"
            if workspace.is_excluded(file_path) or workspace.is_sensitive(file_path):
                return f"ERROR: Refusing to delete excluded or sensitive file: {path}"

            before = _git_diff_for(workspace, path)
            file_path.unlink()
            after = _git_diff_for(workspace, path)

            return f"Deleted {path}{before}{after}"

        except WorkspaceError as exc:
            return _error(exc)

    return delete_file


def build_git_diff_tool(workspace: Workspace):
    @tool(args_schema=GitDiffInput)
    def git_diff(path: str = ".") -> str:
        """Show git diff for a workspace path."""
        try:
            rejection = _decision_rejected(workspace, "git_diff", {"path": path})
            if rejection:
                return rejection

            resolved = workspace.resolve(path)
            rel = workspace.relative(resolved)
            output = _run_git(workspace, ["diff", "--", rel])
            if output is None:
                return "NOT_GIT_REPOSITORY"
            if output:
                return output

            status = _run_git(workspace, ["status", "--short", "--", rel], output_limit=2_000)
            if status is None:
                return "NOT_GIT_REPOSITORY"
            if status:
                return f"NO_TRACKED_DIFF\n\nGit status:\n{status}"

            return "NO_DIFF"

        except WorkspaceError as exc:
            return _error(exc)
        except subprocess.TimeoutExpired as exc:
            return _error(exc)

    return git_diff


def build_git_status_tool(workspace: Workspace):
    @tool(args_schema=GitStatusInput)
    def git_status() -> str:
        """Show concise git status for the workspace."""
        try:
            output = _run_git(workspace, ["status", "--short"], output_limit=6_000)
            if output is None:
                return "NOT_GIT_REPOSITORY"
            return output or "CLEAN"
        except subprocess.TimeoutExpired as exc:
            return _error(exc)

    return git_status
