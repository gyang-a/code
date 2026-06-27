# Code Agent

A workspace-safe CLI coding agent built with LangGraph tool calling.

## Quick Start

```bash
code-agent chat .
```

The CLI loads `.env` from the selected workspace. Set `DEEPSEEK_API_KEY` and optionally `CODE_AGENT_MODEL`.

## Graph Flow

```text
START
  -> agent
  -> execute
  -> tool_result_router
  -> context_manager
  -> agent | END
```

The `agent` node is the only LLM decision point. It decides whether to inspect, edit, validate, or answer. The graph provides runtime context, executes tools, handles human review, tracks changed files and validation output, and compacts old context when needed.

Slash commands such as `/help`, `/diff`, `/doctor`, `/tools`, and `/undo` are handled by the CLI before the graph runs.

## Permission Levels

- Level 0: read-only tools, such as `list_files`, `read_file`, `search_text`, `git_status`, and `git_diff`.
- Level 1: low-risk workspace edits or common test/build/read commands.
- Level 2: user review required, such as package metadata edits, dependency installs, Docker Compose, deletes, unknown shell commands, or writes outside `src/` and `tests/`.
- Level 3: rejected, such as `rm -rf`, `sudo`, `chmod 777`, `curl | bash`, `git reset --hard`, `.env`, and `.ssh`.

## Human Review

Level 2 tool calls follow the LangChain/LangGraph human-in-the-loop pattern:

1. After the model emits tool calls, the graph reviews the full batch before any tool executes.
2. If one or more calls need review, `execute` interrupts with one payload containing `action_requests` for all pending calls and matching `review_configs`.
3. The CLI resumes with a same-order `decisions` list. Each decision can be `approve`, `edit`, `reject`, or `respond`.
4. Approved or edited calls execute and produce normal `ToolMessage` content containing only the tool result.
5. Rejected or responded calls produce one synthetic `ToolMessage` for their corresponding tool call.
6. The agent continues from the resumed state.

Approval protocol data is not appended to `messages`; it lives only in the interrupt payload and resume value.

## Context Compaction

The graph compacts old messages when message count or estimated characters exceed configured limits. It keeps recent `AIMessage(tool_calls) + ToolMessage` blocks intact so tool observations are not orphaned, stores the summary in `state.context_summary`, and injects that summary only for later LLM calls.

## Tools

The model can call bounded filesystem, search, shell, and git tools. `read_file` returns a limited text window by default; the agent must request later `start_line` values to continue reading. Shell commands run through the configured sandbox backend.

## Useful Commands

```text
/doctor
/tools
/diff
/undo
/help
```
