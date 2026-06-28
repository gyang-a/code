from __future__ import annotations

import fnmatch
from pathlib import Path

from code_agent.config import (
    DEFAULT_EXCLUDE_GLOBS,
    DEFAULT_FILE_READ_MAX_LINES,
    DEFAULT_FILE_READ_LIMIT,
    SENSITIVE_FILE_NAMES,
    SENSITIVE_SUFFIXES,
)


class WorkspaceError(ValueError):
    """工作区操作违反沙箱策略时抛出。"""


class Workspace:
    def __init__(
        self,
        root: str | Path,
        *,
        read_limit: int = DEFAULT_FILE_READ_LIMIT,
        read_max_lines: int = DEFAULT_FILE_READ_MAX_LINES,
        exclude_globs: tuple[str, ...] = DEFAULT_EXCLUDE_GLOBS,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.read_limit = read_limit
        self.read_max_lines = read_max_lines
        self.exclude_globs = exclude_globs

        if not self.root.exists():
            raise WorkspaceError(f"工作区不存在: {self.root}")
        if not self.root.is_dir():
            raise WorkspaceError(f"工作区不是目录: {self.root}")

    def resolve(self, relative_path: str | Path) -> Path:
        candidate = Path(relative_path)

        if candidate.is_absolute():
            path = candidate.resolve()
        else:
            path = (self.root / candidate).resolve()

        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise WorkspaceError(f"路径逃逸工作区: {relative_path}") from exc

        return path

    def relative(self, path: str | Path) -> str:
        resolved = self.resolve(path)

        try:
            return resolved.relative_to(self.root).as_posix()
        except ValueError as exc:
            raise WorkspaceError(f"路径逃逸工作区: {path}") from exc

    def is_excluded(self, path: str | Path) -> bool:
        resolved = self.resolve(path)
        rel = self.relative(resolved)

        return any(
            _matches_exclude_glob(rel, pattern)
            for pattern in self.exclude_globs
        )

    def is_sensitive(self, path: str | Path) -> bool:
        resolved = self.resolve(path)
        name = resolved.name.lower()
        parts = {part.lower() for part in resolved.parts}

        return (
            name in SENSITIVE_FILE_NAMES
            or name.endswith(SENSITIVE_SUFFIXES)
            or ".ssh" in parts
        )

    def assert_visible_path(self, path: str | Path) -> Path:
        resolved = self.resolve(path)

        if self.is_excluded(resolved):
            raise WorkspaceError(f"拒绝访问 excluded path: {self.relative(resolved)}")

        if self.is_sensitive(resolved):
            raise WorkspaceError(f"拒绝访问敏感路径: {self.relative(resolved)}")

        return resolved

    def assert_directory(self, path: str | Path) -> Path:
        resolved = self.assert_visible_path(path)

        if not resolved.exists():
            raise WorkspaceError(f"目录不存在: {path}")

        if not resolved.is_dir():
            raise WorkspaceError(f"不是目录: {path}")

        return resolved

    def assert_readable_file(
        self,
        path: str | Path,
        *,
        enforce_size_limit: bool = True,
    ) -> Path:
        resolved = self.assert_visible_path(path)

        if not resolved.exists():
            raise WorkspaceError(f"文件不存在: {path}")

        if not resolved.is_file():
            raise WorkspaceError(f"不是文件: {path}")

        if enforce_size_limit and resolved.stat().st_size > self.read_limit:
            raise WorkspaceError(
                f"文件过大（{resolved.stat().st_size} bytes）。"
                f"请先搜索或读取更小范围: {self.relative(resolved)}"
            )

        return resolved

    def assert_text_file(
        self,
        path: str | Path,
        *,
        enforce_size_limit: bool = True,
    ) -> Path:
        resolved = self.assert_readable_file(
            path,
            enforce_size_limit=enforce_size_limit,
        )

        with resolved.open("rb") as handle:
            sample = handle.read(4096)

        if b"\x00" in sample:
            raise WorkspaceError(f"拒绝读取二进制文件: {self.relative(resolved)}")

        return resolved

    def read_text(self, path: str | Path) -> str:
        resolved = self.assert_text_file(path)
        data = resolved.read_bytes()
        return data.decode("utf-8", errors="replace")

    def read_text_window(
        self,
        path: str | Path,
        *,
        start_line: int = 1,
        max_lines: int | None = None,
    ) -> str:
        resolved = self.assert_text_file(path, enforce_size_limit=False)

        start_line = max(start_line, 1)
        max_lines = self.read_max_lines if max_lines is None else max(max_lines, 1)

        lines: list[str] = []
        seen_after_window = False
        end_line = start_line - 1

        with resolved.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line_number < start_line:
                    continue

                if len(lines) >= max_lines:
                    seen_after_window = True
                    break

                lines.append(f"{line_number}: {line.rstrip()}")
                end_line = line_number

        header = (
            f"FILE: {self.relative(resolved)}\n"
            f"LINES: {start_line}-{end_line if lines else start_line - 1}\n"
        )

        if not lines:
            return header + "No lines returned. The start_line may be past EOF."

        body = "\n".join(lines)

        if seen_after_window:
            body += f"\n... truncated; call read_file with start_line={end_line + 1} for more ..."

        return header + body

    def write_text(
        self,
        path: str | Path,
        content: str,
        *,
        overwrite: bool = True,
    ) -> Path:
        resolved = self.resolve(path)

        if self.is_excluded(resolved):
            raise WorkspaceError(f"拒绝写入 excluded path: {self.relative(resolved)}")

        if self.is_sensitive(resolved):
            raise WorkspaceError(f"拒绝写入敏感文件: {self.relative(resolved)}")

        if resolved.exists() and not overwrite:
            raise WorkspaceError(f"文件已存在: {self.relative(resolved)}")

        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8", newline="")

        return resolved


def _matches_exclude_glob(rel: str, pattern: str) -> bool:
    normalized_rel = rel.replace("\\", "/").strip("/")
    normalized_pattern = pattern.replace("\\", "/").strip("/")

    if not normalized_rel or not normalized_pattern:
        return False

    if fnmatch.fnmatch(normalized_rel, normalized_pattern):
        return True

    if fnmatch.fnmatch(normalized_rel, f"*/{normalized_pattern}"):
        return True

    if normalized_pattern.endswith("/**"):
        directory = normalized_pattern[:-3].rstrip("/")
        return (
            normalized_rel == directory
            or normalized_rel.startswith(f"{directory}/")
            or f"/{directory}/" in f"/{normalized_rel}/"
        )

    return False