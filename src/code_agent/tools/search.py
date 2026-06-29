from __future__ import annotations

import fnmatch
import os
import subprocess
from pathlib import Path

from langchain_core.tools import tool

from code_agent.services.summarizer import truncate
from code_agent.services.workspace import Workspace, WorkspaceError
from code_agent.tools.safety import RiskLevel, classify_tool_call, rejected
from code_agent.tools.schemas import FindFilesInput, SearchTextInput


MAX_FALLBACK_FILE_BYTES = 1_000_000


def _should_skip(workspace: Workspace, path: Path) -> bool:
    try:
        return workspace.is_excluded(path) or workspace.is_sensitive(path)
    except WorkspaceError:
        return True


def _decision_rejected(workspace: Workspace, tool_name: str, payload: dict) -> str | None:
    decision = classify_tool_call(workspace, tool_name, payload)
    if decision.risk == RiskLevel.level_3:
        return rejected(decision.risk, decision.reason)
    return None


def build_search_text_tool(workspace: Workspace):
    @tool(args_schema=SearchTextInput)
    def search_text(query: str, path: str = ".", max_results: int = 100) -> str:
        """Search visible workspace text by literal string. Use this before read_file when locating code."""
        try:
            rejection = _decision_rejected(workspace, "search_text", {"path": path, "query": query})
            if rejection:
                return rejection

            search_root = workspace.resolve(path)

            if not search_root.exists():
                return f"ERROR: Path does not exist: {path}"
            if not search_root.is_dir() and not search_root.is_file():
                return f"ERROR: Not a searchable path: {path}"
            if _should_skip(workspace, search_root):
                return f"ERROR: Refusing to search excluded or sensitive path: {path}"

        except WorkspaceError as exc:
            return f"ERROR: {exc}"

        max_results = min(max(max_results, 1), 500)
        search_target = workspace.relative(search_root) or "."

        rg_command = [
            "rg",
            "--line-number",
            "--color",
            "never",
            "--fixed-strings",
        ]

        for pattern in _rg_exclude_globs(workspace):
            rg_command.extend(["--glob", f"!{pattern}"])

        rg_command.extend([query, search_target])

        try:
            result = subprocess.run(
                rg_command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=workspace.root,
                timeout=10,
                shell=False,
            )

            if result.returncode not in (0, 1):
                return truncate(result.stderr)

            raw_lines = result.stdout.splitlines()

            filtered: list[str] = []
            skipped = 0

            for line in raw_lines:
                file_part = line.split(":", 1)[0]

                try:
                    abs_path = workspace.resolve(file_part)
                    if _should_skip(workspace, abs_path):
                        skipped += 1
                        continue
                except WorkspaceError:
                    skipped += 1
                    continue

                filtered.append(line)

                if len(filtered) >= max_results:
                    break

            if not filtered:
                return f"NO_MATCHES: {query!r} was not found under {workspace.relative(search_root)}."

            output = "\n".join(filtered)

            if len(raw_lines) > len(filtered) + skipped:
                output += f"\n... truncated after {len(filtered)} visible matches ..."

            return truncate(output)

        except FileNotFoundError:
            return _fallback_search(workspace, search_root, query, max_results)

        except subprocess.TimeoutExpired:
            return "ERROR: Search timed out."

    return search_text


def build_find_files_tool(workspace: Workspace):
    @tool(args_schema=FindFilesInput)
    def find_files(pattern: str, path: str = ".", max_results: int = 200) -> str:
        """Find visible files by filename or relative path pattern. Examples: '*.py', 'src/**/*.js'."""
        try:
            rejection = _decision_rejected(workspace, "find_files", {"path": path, "pattern": pattern})
            if rejection:
                return rejection

            search_root = workspace.resolve(path)

            if not search_root.exists():
                return f"ERROR: Path does not exist: {path}"
            if not search_root.is_dir():
                return f"ERROR: Not a directory: {path}"
            if _should_skip(workspace, search_root):
                return f"ERROR: Refusing to search excluded or sensitive path: {path}"

        except WorkspaceError as exc:
            return f"ERROR: {exc}"

        max_results = min(max(max_results, 1), 1000)

        matches: list[str] = []
        for file_path in _walk_visible_files(workspace, search_root):
            rel = workspace.relative(file_path).replace("\\", "/")
            name = file_path.name

            if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(name, pattern):
                matches.append(rel)

                if len(matches) >= max_results:
                    break

        if not matches:
            return f"NO_MATCHES: pattern {pattern!r} matched no visible files under {workspace.relative(search_root)}."

        output = "\n".join(sorted(matches))
        if len(matches) >= max_results:
            output += f"\n... truncated after {max_results} files ..."

        return output

    return find_files


def _fallback_search(workspace: Workspace, search_root: Path, query: str, max_results: int) -> str:
    matches: list[str] = []

    files = [search_root] if search_root.is_file() else _walk_visible_files(workspace, search_root)

    for file_path in files:
        if not file_path.is_file() or _should_skip(workspace, file_path):
            continue

        try:
            if file_path.stat().st_size > MAX_FALLBACK_FILE_BYTES:
                continue

            text = file_path.read_text(encoding="utf-8", errors="replace")

        except OSError:
            continue

        for line_no, line in enumerate(text.splitlines(), start=1):
            if query in line:
                matches.append(f"{workspace.relative(file_path)}:{line_no}:{line}")

                if len(matches) >= max_results:
                    return truncate(
                        "\n".join(matches)
                        + f"\n... truncated after {max_results} matches ..."
                    )

    if not matches:
        return f"NO_MATCHES: {query!r} was not found under {workspace.relative(search_root)}."

    return truncate("\n".join(matches))


def _walk_visible_files(workspace: Workspace, root: Path):
    for current_root, dir_names, file_names in os.walk(root):
        current_path = Path(current_root)

        visible_dirs: list[str] = []
        for dir_name in dir_names:
            dir_path = current_path / dir_name
            if not _should_skip(workspace, dir_path):
                visible_dirs.append(dir_name)

        dir_names[:] = visible_dirs

        for file_name in file_names:
            file_path = current_path / file_name
            if not _should_skip(workspace, file_path):
                yield file_path


def _rg_exclude_globs(workspace: Workspace) -> list[str]:
    patterns: list[str] = []

    for pattern in workspace.exclude_globs:
        normalized = pattern.replace("\\", "/").strip("/")
        if not normalized:
            continue

        variants = {
            normalized,
            f"{normalized}/**",
        }

        if not normalized.startswith("**/"):
            variants.add(f"**/{normalized}")
            variants.add(f"**/{normalized}/**")

        patterns.extend(sorted(variants))

    return sorted(set(patterns))
