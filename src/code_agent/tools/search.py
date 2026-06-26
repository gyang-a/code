from __future__ import annotations

import fnmatch
import subprocess
from pathlib import Path

from langchain_core.tools import tool

from code_agent.services.summarizer import truncate
from code_agent.services.workspace import Workspace, WorkspaceError


def _should_skip(workspace: Workspace, path: Path) -> bool:
    try:
        return workspace.is_excluded(path) or workspace.is_sensitive(path)
    except WorkspaceError:
        return True


def build_search_text_tool(workspace: Workspace):
    @tool
    def search_text(query: str, path: str = ".", max_results: int = 100) -> str:
        """Search text in the workspace using ripgrep when available."""
        try:
            search_root = workspace.resolve(path)
        except WorkspaceError as exc:
            return f"ERROR: {exc}"

        max_results = min(max(max_results, 1), 500)
        rg_command = [
            "rg",
            "--line-number",
            "--hidden",
            "--color",
            "never",
            "--glob",
            "!node_modules/**",
            "--glob",
            "!.git/**",
            "--glob",
            "!.venv/**",
            query,
            str(search_root),
        ]

        try:
            result = subprocess.run(
                rg_command,
                capture_output=True,
                text=True,
                cwd=workspace.root,
                timeout=10,
                shell=False,
            )
            if result.returncode not in (0, 1):
                return truncate(result.stderr)
            lines = result.stdout.splitlines()[:max_results]
            return truncate("\n".join(lines))
        except FileNotFoundError:
            matches: list[str] = []
            for file_path in search_root.rglob("*"):
                if not file_path.is_file() or _should_skip(workspace, file_path):
                    continue
                try:
                    text = file_path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                for line_no, line in enumerate(text.splitlines(), start=1):
                    if query in line:
                        matches.append(f"{workspace.relative(file_path)}:{line_no}:{line}")
                        if len(matches) >= max_results:
                            return truncate("\n".join(matches))
            return truncate("\n".join(matches))
        except subprocess.TimeoutExpired:
            return "ERROR: Search timed out."

    return search_text


def build_find_files_tool(workspace: Workspace):
    @tool
    def find_files(pattern: str, path: str = ".", max_results: int = 200) -> str:
        """Find files by glob-like pattern inside the workspace."""
        try:
            search_root = workspace.resolve(path)
        except WorkspaceError as exc:
            return f"ERROR: {exc}"

        max_results = min(max(max_results, 1), 1000)
        matches: list[str] = []
        for file_path in search_root.rglob("*"):
            if not file_path.is_file() or _should_skip(workspace, file_path):
                continue
            rel = workspace.relative(file_path)
            if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(file_path.name, pattern):
                matches.append(rel)
                if len(matches) >= max_results:
                    break
        return "\n".join(sorted(matches))

    return find_files
