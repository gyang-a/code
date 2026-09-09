from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ListFilesInput(BaseModel):
    path: str = Field(default=".", description="Directory path inside the current project folder to list.")
    max_entries: int = Field(default=200, ge=1, le=1000, description="Maximum visible entries to return.")


class ReadFileInput(BaseModel):
    path: str = Field(description="Text file path inside the current project folder.")
    start_line: int = Field(default=1, ge=1, description="1-based line number to start reading from.")
    max_lines: int | None = Field(
        default=None,
        ge=1,
        le=200,
        description="Maximum number of lines to return. Leave unset for the host default.",
    )


class SearchTextInput(BaseModel):
    query: str = Field(description="Literal text to search for.")
    path: str = Field(default=".", description="Current-project path to search within.")
    max_results: int = Field(default=100, ge=1, le=500, description="Maximum matching lines to return.")


class FindFilesInput(BaseModel):
    pattern: str = Field(description="Glob-style file pattern, for example '*.py' or 'src/**/*.js'.")
    path: str = Field(default=".", description="Current-project path to search within.")
    max_results: int = Field(default=200, ge=1, le=1000, description="Maximum file paths to return.")


class PatchFileInput(BaseModel):
    path: str = Field(description="File path inside the current project folder to edit.")
    old: str = Field(description="Exact text block to replace. It must appear exactly once.")
    new: str = Field(description="Replacement text block.")


class CreateFileInput(BaseModel):
    path: str = Field(description="New file path inside the current project folder.")
    content: str = Field(description="Complete text content for the new file.")


class WriteFileInput(BaseModel):
    path: str = Field(description="File path inside the current project folder to overwrite.")
    content: str = Field(description="Complete replacement text content.")


class DeleteFileInput(BaseModel):
    path: str = Field(description="File path inside the current project folder to delete.")


class GitDiffInput(BaseModel):
    path: str = Field(default=".", description="Current-project path to show git diff for.")


class GitStatusInput(BaseModel):
    pass


class ShellCommandInput(BaseModel):
    command: str = Field(
        min_length=1,
        description=(
            "PowerShell command executed by a fresh `pwsh -Command` process in the Windows sandbox. "
            "Use workdir instead of cd and `$env:NAME` for environment variables."
        ),
    )
    description: str = Field(
        min_length=1,
        description="Short user-facing description of what the command does.",
    )
    workdir: str = Field(
        default=".",
        description="Working directory inside the current project folder.",
    )
    timeout_ms: int | None = Field(
        default=None,
        ge=1,
        description="Optional timeout in milliseconds; the host applies a configured cap.",
    )
    model_config = {"extra": "forbid"}


class SkillsListInput(BaseModel):
    pass


class SkillViewInput(BaseModel):
    name: str = Field(description="Skill name to read, for example 'python-testing'.")
