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
        if result.returncode not in (0, 1):
            return "\n\nDiff 记录: 当前目录不是 git 仓库，或 git diff 不可用。"
        diff = truncate(result.stdout + result.stderr, 4000)
        if diff:
            return "\n\nDiff 记录:\n" + diff
        if resolved.exists():
            return "\n\nDiff 记录: 文件可能是新增文件，或 git diff 暂无变化；可使用 git_status 查看。"
        return "\n\nDiff 记录: 暂无可用 diff。"
    except (WorkspaceError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return f"\n\nDiff 记录失败: {exc}"


def build_read_file_tool(workspace: Workspace):
    @tool
    def read_file(path: str) -> str:
        """读取工作区内的文本文件；敏感文件、二进制文件和超大文件会被拒绝。"""
        try:
            return workspace.read_text(path)
        except WorkspaceError as exc:
            return _error(exc)

    return read_file


def build_list_files_tool(workspace: Workspace):
    @tool
    def list_files(path: str = ".") -> str:
        """列出工作区目录下的直接子项。"""
        try:
            dir_path = workspace.resolve(path)
            if not dir_path.exists():
                return f"ERROR: 目录不存在: {path}"
            if not dir_path.is_dir():
                return f"ERROR: 不是目录: {path}"

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
        """返回工作区路径下有数量上限的文件树。"""
        try:
            max_entries = min(max(max_entries, 1), 1000)
            return "\n".join(workspace.iter_tree(path, max_entries=max_entries))
        except WorkspaceError as exc:
            return _error(exc)

    return get_file_tree


def build_patch_file_tool(workspace: Workspace):
    @tool
    def patch_file(path: str, old: str, new: str, approval_token: str | None = None) -> str:
        """替换文件中的一个精确文本块；old 文本必须只出现一次。"""
        try:
            decision = classify_file_operation(workspace, "patch_file", path)
            if decision.risk.value == "level_3":
                return rejected(decision.risk, decision.reason)
            if decision.requires_approval and approval_token != APPROVAL_TOKEN:
                return approval_required(decision.risk, decision.reason)
            result = replace_exact_once(workspace, path, old, new)
            if not result.changed:
                return (
                    f"ERROR: old 文本在 {path} 中必须只出现一次；"
                    f"实际找到 {result.old_count} 处。"
                )
            return f"{_format_decision_prefix(decision)}\n已修补 {path}{_git_diff_for(workspace, path)}"
        except WorkspaceError as exc:
            return _error(exc)

    return patch_file


def build_create_file_tool(workspace: Workspace):
    @tool
    def create_file(path: str, content: str, approval_token: str | None = None) -> str:
        """在工作区内创建新的文本文件；如果文件已存在则拒绝。"""
        try:
            decision = classify_file_operation(workspace, "create_file", path)
            if decision.risk.value == "level_3":
                return rejected(decision.risk, decision.reason)
            if decision.requires_approval and approval_token != APPROVAL_TOKEN:
                return approval_required(decision.risk, decision.reason)
            workspace.write_text(path, content, overwrite=False)
            return f"{_format_decision_prefix(decision)}\n已创建 {path}{_git_diff_for(workspace, path)}"
        except WorkspaceError as exc:
            return _error(exc)

    return create_file


def build_write_file_tool(workspace: Workspace):
    @tool
    def write_file(path: str, content: str, approval_token: str | None = None) -> str:
        """覆盖工作区内的小文本文件；编辑已有文件时优先使用 patch_file。"""
        try:
            decision = classify_file_operation(workspace, "write_file", path)
            if decision.risk.value == "level_3":
                return rejected(decision.risk, decision.reason)
            if decision.requires_approval and approval_token != APPROVAL_TOKEN:
                return approval_required(decision.risk, decision.reason)
            existing = workspace.resolve(path)
            if existing.exists() and existing.stat().st_size > 20_000:
                if approval_token != APPROVAL_TOKEN:
                    return approval_required(decision.risk, "覆盖大于 20KB 的文件需要确认。")
            workspace.write_text(path, content, overwrite=True)
            return f"{_format_decision_prefix(decision)}\n已写入 {path}{_git_diff_for(workspace, path)}"
        except WorkspaceError as exc:
            return _error(exc)

    return write_file


def build_delete_file_tool(workspace: Workspace):
    @tool
    def delete_file(path: str, approval_token: str | None = None) -> str:
        """请求删除工作区文件；没有人工审批时不会删除。"""
        try:
            decision = classify_file_operation(workspace, "delete_file", path)
            if decision.risk.value == "level_3":
                return rejected(decision.risk, decision.reason)
            if approval_token != APPROVAL_TOKEN:
                return approval_required(decision.risk, decision.reason)
            file_path = workspace.resolve(path)
            if not file_path.exists():
                return f"ERROR: 文件不存在: {path}"
            if not file_path.is_file():
                return f"ERROR: 不是文件: {path}"
            before = _git_diff_for(workspace, path)
            file_path.unlink()
            return f"{allowed(decision.risk, decision.reason)}\n已删除 {path}{before}{_git_diff_for(workspace, path)}"
        except WorkspaceError as exc:
            return _error(exc)

    return delete_file
