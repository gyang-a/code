from __future__ import annotations

import importlib.util
import os
import shlex
import shutil
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from collections.abc import Mapping

import typer
from rich.prompt import Prompt

from code_agent.config import AgentConfig, DEFAULT_MODEL
from code_agent.services.env import load_dotenv
from code_agent.services.persistence import (
    list_sessions,
    open_project_checkpointer,
    record_session,
    SessionRecord,
)
from code_agent.services.skills import SkillStore, format_pending_summary, format_skill_index
from code_agent.services.summarizer import truncate
from code_agent.services.usage import (
    TokenUsage,
    estimate_message_tokens,
    format_usage_line,
    usage_from_messages,
)
from code_agent.services.workspace import Workspace, WorkspaceError
from code_agent.ui.approval import format_approval_summary
from code_agent.ui.console import console, print_banner, print_help

app = typer.Typer()


@dataclass
class Session:
    workspace: str
    model: str = DEFAULT_MODEL
    env_file: str | None = None
    thread_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    interactions: int = 0
    tool_loops: int = 0
    context_compressions: int = 0
    last_context_tokens: int = 0
    model_usage: TokenUsage = field(default_factory=TokenUsage)

    def reset(self) -> None:
        self.thread_id = str(uuid.uuid4())
        self.interactions = 0
        self.tool_loops = 0
        self.context_compressions = 0
        self.last_context_tokens = 0
        self.model_usage = TokenUsage()


def _resolve_chat_workspace(workspace: str) -> str:
    raw_workspace = workspace.strip() if workspace else "."
    return str(Path(raw_workspace).expanduser().resolve())


def _initial_state(session: Session, user_input: str) -> dict:
    from langchain_core.messages import HumanMessage

    return {
        "messages": [HumanMessage(content=user_input)],
        "workspace": session.workspace,
        "thread_id": session.thread_id,
        "user_goal": user_input,
        "changed_files": [],
        "did_write": False,
        "tool_errors": [],
        "final_answer": None,
    }


def _run_git_diff(workspace: str) -> str:
    result = _run_git_command(workspace, ["diff", "--", "."])
    output = result.stdout if result.returncode == 0 else result.stdout + result.stderr
    return truncate(output)


def _run_git_command(workspace: str, args: list[str], timeout: int = 10) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=workspace,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            shell=False,
        )
    except FileNotFoundError as exc:
        return subprocess.CompletedProcess(["git", *args], 127, "", f"ERROR: {exc}")
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(["git", *args], 124, exc.stdout or "", f"ERROR: {exc}")


def _run_git_status_porcelain(workspace: str) -> str:
    result = _run_git_command(workspace, ["status", "--porcelain", "--", "."])
    output = result.stdout if result.returncode == 0 else result.stdout + result.stderr
    return truncate(output)


def _parse_undo_args(raw_arg: str) -> tuple[bool, list[str]]:
    include_untracked = False
    targets: list[str] = []
    for raw_part in shlex.split(raw_arg, posix=False):
        part = raw_part.strip("\"'")
        if part == "--include-untracked":
            include_untracked = True
        elif part:
            targets.append(part)
    return include_untracked, targets or ["."]


def _run_git_undo(workspace: str, raw_arg: str) -> str:
    include_untracked, targets = _parse_undo_args(raw_arg)
    restore = _run_git_command(workspace, ["restore", "--staged", "--worktree", "--", *targets], timeout=20)
    outputs = [restore.stdout if restore.returncode == 0 else restore.stdout + restore.stderr]
    if restore.returncode != 0:
        return truncate("".join(outputs))

    if include_untracked:
        clean = _run_git_command(workspace, ["clean", "-fd", "--", *targets], timeout=20)
        outputs.append(clean.stdout if clean.returncode == 0 else clean.stdout + clean.stderr)
        if clean.returncode != 0:
            return truncate("".join(outputs))

    remaining = _run_git_status_porcelain(workspace)
    if remaining:
        outputs.append("\nRestored tracked changes. Remaining changes:\n")
        outputs.append(remaining)
        if not include_untracked:
            outputs.append("\nTip: use /undo --include-untracked to remove untracked files.\n")
    else:
        outputs.append("Restored workspace changes.")
    return truncate("".join(outputs))


def _print_raw(text: str) -> None:
    console.print(text, markup=False, highlight=False)


def _doctor(workspace: str) -> None:
    env_path = Path(workspace) / ".env"
    table_rows = [
        ("workspace", "ok" if Path(workspace).is_dir() else "missing"),
        (".env", "found" if env_path.exists() else "missing"),
        ("DEEPSEEK_API_KEY", "set" if os.getenv("DEEPSEEK_API_KEY") else "missing"),
        ("CODE_AGENT_MODEL", os.getenv("CODE_AGENT_MODEL") or "unset"),
        ("git", "found" if shutil.which("git") else "missing"),
        ("rg", "found" if shutil.which("rg") else "missing"),
        ("langgraph", "found" if importlib.util.find_spec("langgraph") else "missing"),
        ("langchain", "found" if importlib.util.find_spec("langchain") else "missing"),
        ("langchain_deepseek", "found" if importlib.util.find_spec("langchain_deepseek") else "missing"),
    ]
    for name, status in table_rows:
        console.print(f"[bold]{name}[/bold]: {status}")


def _handle_slash(command: str, session: Session) -> bool:
    parts = command.strip().split(maxsplit=1)
    name = parts[0]
    arg = parts[1] if len(parts) > 1 else ""

    if name in {"/exit", "/quit"}:
        return False
    if name == "/help":
        print_help()
    elif name == "/clear":
        session.reset()
        console.print(f"Started a new thread: {session.thread_id}")
    elif name == "/model":
        if arg:
            session.model = arg
            console.print(f"Model set to: {session.model}")
        else:
            console.print(f"Current model: {session.model}")
    elif name == "/status":
        console.print(f"workspace: {session.workspace}")
        console.print(f"thread_id: {session.thread_id}")
        console.print(f"model: {session.model}")
        console.print(f"interactions: {session.interactions}")
    elif name == "/tools":
        from code_agent.services.permissions import describe_permission_policy

        _print_raw(describe_permission_policy())
    elif name == "/diff":
        diff = _run_git_diff(session.workspace)
        _print_raw(diff or "No git diff.")
    elif name in {"/undo", "/revert"}:
        status = _run_git_status_porcelain(session.workspace)
        if not status:
            _print_raw("No workspace changes to restore.")
            return True
        include_untracked, targets = _parse_undo_args(arg)
        scope = " ".join(targets)
        action = "restore tracked changes and delete untracked files" if include_untracked else "restore tracked changes"
        _print_raw(f"{action}: {scope}\n\nCurrent changes:\n{status}")
        approved = Prompt.ask("Confirm restore?", choices=["y", "n"], default="n")
        if approved == "y":
            _print_raw(_run_git_undo(session.workspace, arg))
        else:
            _print_raw("Restore cancelled.")
    elif name == "/doctor":
        _doctor(session.workspace)
    elif name == "/usage":
        console.print(f"interactions: {session.interactions}")
        console.print(f"thread_id: {session.thread_id}")
        _print_usage(session)
    elif name == "/skills":
        _handle_skills_command(arg)
    elif name == "/resume":
        _handle_resume_command(session, arg)
    elif name == "/sessions":
        _handle_sessions_command(session.workspace)
    elif name == "/mcp":
        console.print("MCP integration is not configured in this MVP.")
    else:
        console.print(f"Unknown slash command: {name}. Type /help to list commands.")

    return True


def _handle_skills_command(arg: str) -> None:
    store = SkillStore()
    parts = shlex.split(arg, posix=False) if arg else []
    command = parts[0].lower() if parts else "list"

    try:
        if command in {"list", "ls"}:
            _print_raw(format_skill_index(store.list_skills()))
        elif command == "path":
            _print_raw(f"skills: {store.skills_dir}\npending: {store.pending_dir}")
        elif command == "pending":
            _print_raw(format_pending_summary(store.list_pending()))
        elif command == "view" and len(parts) >= 2:
            _print_raw(store.read_skill(parts[1].strip("\"'")))
        elif command == "diff" and len(parts) >= 2:
            _print_raw(store.pending_diff(parts[1].strip("\"'")))
        elif command == "approve" and len(parts) >= 2:
            change = store.approve_pending(parts[1].strip("\"'"))
            _print_raw(f"Approved {change.id}; wrote global skill: {change.skill_name}")
        elif command == "reject" and len(parts) >= 2:
            change = store.reject_pending(parts[1].strip("\"'"))
            _print_raw(f"Rejected pending skill change: {change.id}")
        else:
            _print_raw(
                "Usage: /skills [list|path|pending|view <name>|diff <id>|approve <id>|reject <id>]"
            )
    except Exception as exc:
        _print_raw(f"ERROR: {exc}")


def _handle_sessions_command(workspace: str) -> None:
    records = list_sessions(workspace)
    _print_raw(_format_sessions(records))


def _handle_resume_command(session: Session, arg: str) -> None:
    thread_id = _select_session_thread_id(session.workspace, requested=arg.strip() or None)
    if thread_id is None:
        return

    session.thread_id = thread_id
    session.interactions = 0
    session.tool_loops = 0
    session.context_compressions = 0
    session.last_context_tokens = 0
    session.model_usage = TokenUsage()
    console.print(f"[dim]Resumed thread: {session.thread_id}[/dim]")


def _select_session_thread_id(workspace: str, *, requested: str | None = None) -> str | None:
    records = list_sessions(workspace)
    _print_raw(_format_sessions(records))
    if not records:
        return

    choice = requested
    if not choice:
        choice = Prompt.ask("Thread id, prefix, or number to resume").strip()
    thread_id = _resolve_session_choice(records, choice)
    if thread_id is None:
        _print_raw(f"ERROR: Unknown or ambiguous session: {choice}")
        return None
    return thread_id


def _format_sessions(records: list[SessionRecord]) -> str:
    if not records:
        return "No saved sessions for this workspace."

    lines = ["Saved sessions:"]
    for index, record in enumerate(records, start=1):
        lines.append(f"{index}. {record.thread_id}  {record.updated_at}  {record.title}")
    return "\n".join(lines)


def _resolve_session_choice(records: list[SessionRecord], choice: str) -> str | None:
    normalized = choice.strip()
    if not normalized:
        return None

    if normalized.isdigit():
        index = int(normalized)
        if 1 <= index <= len(records):
            return records[index - 1].thread_id

    exact = [record.thread_id for record in records if record.thread_id == normalized]
    if len(exact) == 1:
        return exact[0]

    prefix = [record.thread_id for record in records if record.thread_id.startswith(normalized)]
    if len(prefix) == 1:
        return prefix[0]

    return None


def _is_slash_command(user_input: str) -> bool:
    return user_input.lstrip().startswith("/")


def _prompt_approval_decisions(interrupt_value: Any) -> list[dict[str, Any]]:
    payload = interrupt_value if isinstance(interrupt_value, dict) else {}
    action_requests = payload.get("action_requests")
    if not isinstance(action_requests, list):
        legacy_action = payload.get("action")
        action_requests = [legacy_action] if legacy_action else []

    decisions: list[dict[str, Any]] = []
    for index, action in enumerate(action_requests, start=1):
        action = action if isinstance(action, dict) else {}
        reason = str(action.get("description") or action.get("reason") or "This action requires approval.")
        summary = format_approval_summary(action, reason)
        console.print(f"\n[bold yellow]Approval required[/bold yellow] [{index}/{len(action_requests)}] {summary}")
        approved = Prompt.ask("Approve this action?", choices=["y", "n"], default="n")
        if approved == "y":
            decisions.append({"type": "approve"})
        else:
            decisions.append({"type": "reject", "message": "Action rejected by user."})
    if not decisions:
        decisions.append({"type": "reject", "message": "Action rejected by user."})
    return decisions


@app.command()
def main(
    workspace: str = typer.Argument(".", help="Workspace directory for the code agent."),
    model: str | None = typer.Option(None, "--model", "-m", help="Override the default model name."),
) -> None:
    """Start an interactive workspace-safe code agent."""
    session, env_path, loaded_env = _create_session(workspace, model=model)
    _run_interactive_session(session, env_path=env_path, loaded_env=loaded_env, resumed=False)


def _create_session(
    workspace: str,
    *,
    model: str | None,
    thread_id: str | None = None,
) -> tuple[Session, Path, bool]:
    resolved_workspace = _validate_workspace(workspace)
    env_path = Path(resolved_workspace) / ".env"
    loaded_env = load_dotenv(env_path)
    selected_model = model or os.getenv("CODE_AGENT_MODEL", DEFAULT_MODEL)
    session = Session(
        workspace=resolved_workspace,
        model=selected_model,
        env_file=str(env_path) if loaded_env else None,
        thread_id=thread_id or str(uuid.uuid4()),
    )
    return session, env_path, loaded_env


def _validate_workspace(workspace: str) -> str:
    try:
        resolved_workspace = _resolve_chat_workspace(workspace)
        Workspace(resolved_workspace)
    except WorkspaceError as exc:
        raise typer.BadParameter(str(exc)) from exc
    return resolved_workspace


def _run_interactive_session(
    session: Session,
    *,
    env_path: Path,
    loaded_env: bool,
    resumed: bool,
) -> None:
    print_banner(session.workspace, session.model)
    console.print(f"[dim]{'Resumed' if resumed else 'Started'} thread: {session.thread_id}[/dim]")
    if loaded_env:
        console.print(f"[dim]Loaded environment variables from {env_path}[/dim]")

    with open_project_checkpointer(session.workspace) as checkpointer:
        graph = None
        while True:
            user_input = Prompt.ask("\n[bold cyan]you[/bold cyan]").strip()
            if not user_input:
                continue

            if _is_slash_command(user_input):
                if not _handle_slash(user_input, session):
                    break
                continue

            if graph is None:
                try:
                    from code_agent.graph import build_graph

                    graph = build_graph(
                        session.workspace,
                        AgentConfig(model=session.model),
                        checkpointer=checkpointer,
                    )
                except Exception as exc:
                    if "Missing credentials" in str(exc):
                        console.print("[red]Graph initialization failed[/red] Missing DeepSeek credentials.")
                        console.print("Set DEEPSEEK_API_KEY in .env or run /doctor to inspect the environment.")
                    else:
                        console.print(f"[red]Graph initialization failed[/red] {exc}")
                        console.print("Run /doctor to inspect dependencies and environment variables.")
                    continue

            record_session(session.workspace, session.thread_id, title=user_input)
            session.interactions += 1
            config = {"configurable": {"thread_id": session.thread_id}}
            console.print("[dim]Agent started[/dim]")
            from code_agent.services.trace import TurnTraceRecorder

            trace_recorder = TurnTraceRecorder(
                workspace=session.workspace,
                thread_id=session.thread_id,
                user_request=user_input,
            )

            try:
                final_answer = _run_graph_stream(
                    graph,
                    _initial_state(session, user_input),
                    config,
                    session,
                    trace_recorder=trace_recorder,
                )
                while final_answer is None:
                    state = graph.get_state(config)
                    interrupts = state.interrupts
                    if not interrupts:
                        values = state.values
                        final_answer = values.get("final_answer") or values["messages"][-1].content
                        break
                    decisions = _prompt_approval_decisions(interrupts[0].value)
                    if any(decision.get("type") in {"approve", "edit"} for decision in decisions):
                        console.print("[dim]Executing approved action(s).[/dim]")
                    from langgraph.types import Command

                    final_answer = _run_graph_stream(
                        graph,
                        Command(resume={"decisions": decisions}),
                        config,
                        session,
                        trace_recorder=trace_recorder,
                    )
            except Exception as exc:
                console.print(f"[red]Agent error:[/red] {exc}")
                _save_turn_trace(
                    session,
                    trace_recorder,
                    final_answer="",
                    user_feedback=f"Agent error: {exc}",
                )
                continue

            state = graph.get_state(config)
            record_session(session.workspace, session.thread_id)
            session.tool_loops += state.values.get("run_model_call_count", 0)
            session.last_context_tokens = estimate_message_tokens(state.values.get("messages", []))
            answer = final_answer or state.values.get("final_answer") or state.values["messages"][-1].content
            console.print("\n[bold green]Agent[/bold green]")
            _print_raw(str(answer))
            trace_payload = _save_turn_trace(session, trace_recorder, final_answer=str(answer))
            _review_trace_for_skills(session, trace_payload)
            _print_usage(session)


def _run_graph_stream(
    graph,
    graph_input,
    config: dict,
    session: Session | None = None,
    *,
    trace_recorder=None,
) -> str | None:
    from code_agent.ui.stream import final_answer_from_chunk, interrupt_from_chunk, render_stream_chunk

    final_answer = None
    for chunk in graph.stream(graph_input, config=config, stream_mode="updates"):
        if session is not None:
            _update_usage_from_chunk(session, chunk)
        if trace_recorder is not None:
            trace_recorder.record_chunk(chunk)
        render_stream_chunk(chunk)
        interrupt_value = interrupt_from_chunk(chunk)
        if interrupt_value is not None:
            return None
        chunk_answer = final_answer_from_chunk(chunk)
        if chunk_answer:
            final_answer = chunk_answer
    return final_answer


def _update_usage_from_chunk(session: Session, chunk: Mapping[str, Any]) -> None:
    from langchain_core.messages import BaseMessage

    for node_name, update in chunk.items():
        if isinstance(update, Mapping):
            messages = update.get("messages")
            if node_name in {"agent", "model"} and isinstance(messages, list):
                session.model_usage.add(
                    usage_from_messages(
                        [message for message in messages if isinstance(message, BaseMessage)]
                    )
                )
            if "SummarizationMiddleware" in node_name and _has_meaningful_update(update):
                session.context_compressions += 1


def _has_meaningful_update(update: Mapping[str, Any]) -> bool:
    return any(bool(value) for value in update.values())


def _print_usage(session: Session) -> None:
    console.print(
        f"[dim]{format_usage_line(context_tokens=session.last_context_tokens, model_usage=session.model_usage, compression_count=session.context_compressions)}[/dim]"
    )


def _save_turn_trace(
    session: Session,
    trace_recorder,
    *,
    final_answer: str,
    user_feedback: str = "",
) -> Mapping[str, Any]:
    from code_agent.services.trace import write_turn_trace

    existing_skills = SkillStore().list_skills()
    trace_payload = trace_recorder.build_trace(
        final_answer=final_answer,
        existing_skills=existing_skills,
        user_feedback=user_feedback,
    )
    path = write_turn_trace(session.workspace, trace_payload)
    try:
        rel_path = Path(path).resolve().relative_to(Path(session.workspace).resolve()).as_posix()
    except ValueError:
        rel_path = str(path)
    console.print(f"[dim]Trace saved: {rel_path}[/dim]")
    return trace_payload


def _review_trace_for_skills(session: Session, trace_payload: Mapping[str, Any]) -> None:
    from code_agent.services.skill_review import review_trace_for_skills

    result = review_trace_for_skills(
        trace=trace_payload,
        config=AgentConfig(model=session.model),
    )
    _print_skill_review_result(
        {
            "skill_review_status": result.status,
            "skill_review_message": result.message,
            "skill_review_pending_id": result.pending_id,
        }
    )


def _print_skill_review_result(state_values: Mapping[str, Any]) -> None:
    status = str(state_values.get("skill_review_status") or "")
    if status == "staged":
        pending_id = state_values.get("skill_review_pending_id")
        console.print(
            f"[dim]{state_values.get('skill_review_message')} "
            f"Use /skills diff {pending_id} to inspect.[/dim]"
        )
    elif status == "none":
        console.print(f"[dim]Skill review ran: {state_values.get('skill_review_message')}[/dim]")
    elif status == "error":
        console.print(f"[dim]{state_values.get('skill_review_message')}[/dim]")


if __name__ == "__main__":
    app()
