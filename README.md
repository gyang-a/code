# Code Agent

A workspace-safe CLI code agent prototype built around LangGraph tool calling.

The project is intentionally scoped like a resume-grade MVP:

- fixed workspace sandbox
- bounded file reads and search results
- safe `patch_file` edits with exact-match checks
- structured validation commands instead of arbitrary shell
- git status/diff visibility
- Claude Code style slash commands in the CLI

## Install

```bash
pip install -e ".[dev]"
```

Set `OPENAI_API_KEY` before asking natural-language tasks.

## Run

```bash
code-agent .
```

or:

```bash
python -m code_agent.main .
```

## Slash Commands

- `/help` shows commands
- `/clear` starts a fresh thread
- `/model [name]` shows or changes the model
- `/status` shows session status
- `/tools` shows detected validation commands
- `/diff` shows current git diff
- `/doctor` checks local dependencies
- `/usage` shows local counters
- `/mcp` placeholder for future MCP integrations
- `/exit` quits

## Architecture

```text
START
  -> load_project_context
  -> route_input
      -> slash_command -> command_handler
      -> user_task -> plan_node
          -> agent_loop
          -> execute
          -> tool_result_router
              -> observe
              -> approval -> observe
              -> reject -> observe
          -> validation_node
          -> review_diff_node
          -> final_summary
```

The model never receives direct filesystem access. It can only call tools that
enforce path, size, command, and sensitive-file policies.

Slash commands are handled by the CLI control plane first, so `/help`, `/diff`,
`/doctor`, and similar commands do not require an LLM call.

## Permission Levels

- Level 0: read-only operations such as `list_files`, `read_file`, `search_text`, `git_status`, and `git_diff`
- Level 1: low-risk edits such as `patch_file`, creating files under `src/`, and updating tests; the tool records a git diff note
- Level 2: requires confirmation, including `package.json`, `pyproject.toml`, installs, Docker compose, deletion, and broad writes
- Level 3: always rejected, including `rm -rf`, `sudo`, `chmod 777`, `curl | bash`, `git reset --hard`, `.env`, and `.ssh`

The MVP returns `APPROVAL_REQUIRED[level_2]` for Level 2 actions. A later
checkpoint/interrupt node can resume approved actions without changing the
tool boundary.

## Approval Flow

Level 2 actions now use LangGraph interrupts:

1. The tool returns `APPROVAL_REQUIRED[level_2]` without executing.
2. The graph stores the pending tool name and arguments.
3. The CLI shows the reason and asks for `y/n`.
4. If approved, the graph resumes and executes the same tool with an internal
   approval token.
5. If denied, the agent receives an observation and must choose a safer path.

Try it with:

```text
帮我修改 pyproject.toml 的 description
```

Expected result: the CLI should pause for approval before writing.
