from __future__ import annotations

from pydantic import BaseModel, Field


class ListFilesInput(BaseModel):
    path: str = Field(default=".", description="Directory path inside the workspace to list.")
    max_entries: int = Field(default=200, ge=1, le=1000, description="Maximum visible entries to return.")


class ReadFileInput(BaseModel):
    path: str = Field(description="Text file path inside the workspace.")
    start_line: int = Field(default=1, ge=1, description="1-based line number to start reading from.")
    max_lines: int | None = Field(
        default=None,
        ge=1,
        le=200,
        description="Maximum number of lines to return. Leave unset for the host default.",
    )


class SearchTextInput(BaseModel):
    query: str = Field(description="Literal text to search for.")
    path: str = Field(default=".", description="Workspace path to search within.")
    max_results: int = Field(default=100, ge=1, le=500, description="Maximum matching lines to return.")


class FindFilesInput(BaseModel):
    pattern: str = Field(description="Glob-style file pattern, for example '*.py' or 'src/**/*.js'.")
    path: str = Field(default=".", description="Workspace path to search within.")
    max_results: int = Field(default=200, ge=1, le=1000, description="Maximum file paths to return.")


class PatchFileInput(BaseModel):
    path: str = Field(description="File path inside the workspace to edit.")
    old: str = Field(description="Exact text block to replace. It must appear exactly once.")
    new: str = Field(description="Replacement text block.")


class CreateFileInput(BaseModel):
    path: str = Field(description="New file path inside the workspace.")
    content: str = Field(description="Complete text content for the new file.")


class WriteFileInput(BaseModel):
    path: str = Field(description="File path inside the workspace to overwrite.")
    content: str = Field(description="Complete replacement text content.")


class DeleteFileInput(BaseModel):
    path: str = Field(description="File path inside the workspace to delete.")


class RunShellInput(BaseModel):
    command: str = Field(
        description=(
            "Shell command to run inside the workspace sandbox. Use only for tests, builds, "
            "package installs/scaffolding, or short temporary scripts; do not use for file "
            "listing, reading, searching, diffing, editing, or deleting."
        )
    )
    timeout_seconds: int = Field(default=180, ge=1, le=180, description="Command timeout in seconds.")


class GitDiffInput(BaseModel):
    path: str = Field(default=".", description="Workspace path to show git diff for.")


class GitStatusInput(BaseModel):
    pass