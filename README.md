# Code Agent

A workspace-safe CLI coding agent built with LangGraph tool calling.

## Quick Start

```bash
code-agent .
```

The CLI loads `.env` from the selected workspace. Set `DEEPSEEK_API_KEY` and optionally `CODE_AGENT_MODEL`.

The selected workspace is the safety boundary. By default, `code-agent .`
binds to the current working directory. You can also pass the project folder
explicitly, for example `code-agent C:\path\to\project`.

## Graph Flow

```text
START
  -> agent
  -> execute
  -> tool_result_router
  -> context_manager
  -> agent | END
```

The `agent` node is the only LLM decision point. It decides whether to inspect, edit, or answer. The graph provides runtime context, executes project tools, handles human review, tracks changed files, and compacts old context when needed.

Slash commands such as `/help`, `/diff`, `/doctor`, `/tools`, `/resume`, and `/undo` are handled by the CLI before the graph runs.

## Permission Levels

- Level 0: read-only tools, such as `list_files`, `read_file`, `search_text`, `git_status`, and `git_diff`.
- Level 1: low-risk workspace edits.
- Level 2: user review required, such as package metadata edits, deletes, full-file overwrites, or writes outside `src/` and `tests/`.
- Level 3: rejected sensitive or excluded paths, such as `.env`, `.ssh`, and `.git`.

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

## Global Skills

Code Agent can use a reusable skill library across all workspaces. By default, skills
live in this Code Agent checkout, not in the workspace being edited:

```text
<code-agent-project>/.code-agent/skills/<skill-name>/SKILL.md
```

Set `CODE_AGENT_HOME`, `CODE_AGENT_SKILLS_DIR`, or `CODE_AGENT_PENDING_SKILLS_DIR` to
override those locations. Pending changes created by older versions under
`~/.code-agent/pending/skills` are still readable so they can be approved or rejected.

Each turn injects a compact global skill index into runtime metadata. The model can call
`skills_list` and `skill_view` to load a relevant skill on demand.

After sufficiently tool-heavy work, or after write tools are used, a background reviewer
checks whether durable procedural knowledge should be saved. The reviewer reads the
structured turn trace, not compressed conversation state, so tool calls, results, file
changes, errors, validation, final answer, and existing skills remain auditable. When it
finds a useful skill, the CLI presents the proposed change like a tool approval: it prints
the diff, asks for `y/n`, and writes the skill only after approval. Use:

```text
/skills list
/skills path
/skills view <name>
```

Legacy pending proposals are still supported for old local data:

```text
/skills pending
/skills diff <id>
/skills approve <id>
/skills reject <id>
```

## Turn Traces

Each agent turn appends an audit record to one project trace under the selected
workspace:

```text
.code-agent/traces/project_trace.json
```

The trace directory is local state. Code Agent writes `.code-agent/.gitignore` and, when
the workspace is a Git repository, adds `.code-agent/` to `.git/info/exclude` so traces
and checkpoints are not tracked by project Git history. Skill review reads this project
trace so user feedback and corrections across turns are visible. When the project trace
grows, review receives a bounded view: a structured historical summary plus the most
recent raw turns, rather than the full trace file.

## Undo

`/undo` is scoped to the latest agent turn. Each turn records the Git dirty paths that
already existed before the agent started, plus the files the agent actually changed.
When the agent edits a file that was already dirty, Code Agent saves a private before
snapshot under `.code-agent/undo/<turn-id>/` and restores that snapshot during undo. For
files that were clean at turn start, undo uses Git restore; for files the agent created,
undo removes only those paths.

## Tools

The model can call bounded filesystem, search, git, and skill tools. `read_file` returns a limited text window by default; the agent must request later `start_line` values to continue reading. Command execution is not available to the agent; if validation is useful, the final answer should suggest exact commands for the user to run locally.

## Useful Commands

```text
/doctor
/tools
/diff
/undo
/sessions
/resume [thread-id]
/help
```

Inside an active CLI session, `/sessions` lists saved conversation threads for
the workspace. `/resume` lists saved threads and prompts for a thread id, unique
prefix, or list number; `/resume [thread-id]` switches directly.
