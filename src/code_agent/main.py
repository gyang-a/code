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
from code_agent.services.skills import SkillStore, format_pending_summary, format_skill_index, legacy_pending_skill_root
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


def _run_git_dirty_paths(workspace: str) -> list[str]:
    result = _run_git_command(workspace, ["status", "--porcelain=v1", "-z", "--", "."])
    if result.returncode != 0:
        return []
    return _parse_git_status_paths_z(result.stdout)


def _parse_git_status_paths_z(output: str) -> list[str]:
    records = [record for record in output.split("\0") if record]
    paths: list[str] = []
    index = 0
    while index < len(records):
        record = records[index]
        status = record[:2]
        path = record[3:] if len(record) > 3 else ""
        if path:
            paths.append(_normalize_git_path(path))
        if "R" in status or "C" in status:
            index += 2
        else:
            index += 1
    return _unique_preserving_order(paths)


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
    _include_untracked, targets = _parse_undo_args(raw_arg)
    undo_plan = _build_agent_undo_plan(workspace, targets)
    if undo_plan["error"]:
        return str(undo_plan["error"])

    restore_paths = list(undo_plan["restore_paths"])
    clean_paths = list(undo_plan["clean_paths"])
    snapshot_restores = list(undo_plan["snapshot_restores"])
    snapshot_deletes = list(undo_plan["snapshot_deletes"])
    skipped = list(undo_plan["skipped"])
    outputs: list[str] = []

    for snapshot in snapshot_restores:
        try:
            _restore_undo_snapshot(workspace, snapshot)
        except Exception as exc:
            return f"ERROR: Failed to restore undo snapshot for {snapshot.get('path')}: {exc}"

    for path in snapshot_deletes:
        try:
            _delete_workspace_file(workspace, str(path))
        except Exception as exc:
            return f"ERROR: Failed to restore deleted baseline state for {path}: {exc}"

    if restore_paths:
        restore = _run_git_command(workspace, ["restore", "--staged", "--worktree", "--", *restore_paths], timeout=20)
        outputs.append(restore.stdout if restore.returncode == 0 else restore.stdout + restore.stderr)
        if restore.returncode != 0:
            return truncate("".join(outputs))

    if clean_paths:
        clean = _run_git_command(workspace, ["clean", "-fd", "--", *clean_paths], timeout=20)
        outputs.append(clean.stdout if clean.returncode == 0 else clean.stdout + clean.stderr)
        if clean.returncode != 0:
            return truncate("".join(outputs))

    if restore_paths or clean_paths or snapshot_restores or snapshot_deletes:
        outputs.append("Restored agent changes.")
        if snapshot_restores:
            outputs.append(
                "\nRestored pre-agent dirty snapshots:\n"
                + "\n".join(f"- {snapshot.get('path')}" for snapshot in snapshot_restores)
            )
        if snapshot_deletes:
            outputs.append(
                "\nRestored pre-agent deleted paths:\n"
                + "\n".join(f"- {path}" for path in snapshot_deletes)
            )
        if restore_paths:
            outputs.append("\nRestored tracked paths:\n" + "\n".join(f"- {path}" for path in restore_paths))
        if clean_paths:
            outputs.append("\nRemoved agent-created untracked paths:\n" + "\n".join(f"- {path}" for path in clean_paths))
    else:
        outputs.append("No safe agent changes to restore.")

    if skipped:
        outputs.append("\nSkipped paths:\n" + "\n".join(f"- {item}" for item in skipped))

    remaining = _run_git_status_porcelain(workspace)
    if remaining:
        outputs.append("\nRemaining changes:\n")
        outputs.append(remaining)
    return truncate("".join(outputs))


def _build_agent_undo_plan(workspace: str, targets: list[str]) -> dict[str, Any]:
    from code_agent.services.trace import load_project_trace

    trace = load_project_trace(workspace)
    turns = trace.get("turns")
    if not isinstance(turns, list) or not turns:
        return {
            "error": "No agent trace found. Refusing to run broad workspace undo.",
            "restore_paths": [],
            "clean_paths": [],
            "snapshot_restores": [],
            "snapshot_deletes": [],
            "skipped": [],
        }

    latest_turn = turns[-1]
    if not isinstance(latest_turn, Mapping):
        return {
            "error": "Latest agent trace is invalid. Refusing to run broad workspace undo.",
            "restore_paths": [],
            "clean_paths": [],
            "snapshot_restores": [],
            "snapshot_deletes": [],
            "skipped": [],
        }

    baseline = latest_turn.get("git_baseline")
    if not isinstance(baseline, Mapping) or not isinstance(baseline.get("dirty_paths"), list):
        return {
            "error": "Latest agent trace has no git baseline. Refusing to risk user changes.",
            "restore_paths": [],
            "clean_paths": [],
            "snapshot_restores": [],
            "snapshot_deletes": [],
            "skipped": [],
        }

    workspace_obj = Workspace(workspace)
    target_paths = _normalize_undo_targets(workspace_obj, targets)
    baseline_dirty = {
        _normalize_git_path(str(path))
        for path in baseline.get("dirty_paths", [])
        if str(path)
    }

    restore_paths: list[str] = []
    clean_paths: list[str] = []
    snapshot_restores: list[dict[str, Any]] = []
    snapshot_deletes: list[str] = []
    skipped: list[str] = []
    snapshots = _undo_snapshots_by_path(latest_turn)
    for change in _latest_turn_file_changes(latest_turn):
        path = _normalize_workspace_path(workspace_obj, str(change.get("path") or ""))
        if not path:
            continue
        if not _path_matches_targets(path, target_paths):
            continue
        if path in baseline_dirty:
            snapshot = snapshots.get(path)
            if snapshot is None:
                skipped.append(f"{path} was already modified before this agent turn and has no undo snapshot")
                continue
            if snapshot.get("existed") is False:
                snapshot_deletes.append(path)
            else:
                snapshot_restores.append(snapshot)
            continue

        operation = str(change.get("operation") or "")
        if operation == "create":
            clean_paths.append(path)
        else:
            restore_paths.append(path)

    return {
        "error": "",
        "restore_paths": _unique_preserving_order(restore_paths),
        "clean_paths": _unique_preserving_order(clean_paths),
        "snapshot_restores": _unique_snapshot_restores(snapshot_restores),
        "snapshot_deletes": _unique_preserving_order(snapshot_deletes),
        "skipped": _unique_preserving_order(skipped),
    }


def _restore_undo_snapshot(workspace: str, snapshot: Mapping[str, Any]) -> None:
    workspace_obj = Workspace(workspace)
    target_rel = _normalize_workspace_path(workspace_obj, str(snapshot.get("path") or ""))
    snapshot_rel = _normalize_workspace_path(workspace_obj, str(snapshot.get("snapshot_path") or ""))
    if not target_rel or not snapshot_rel:
        raise WorkspaceError("Invalid undo snapshot path.")

    snapshot_path = workspace_obj.resolve(snapshot_rel)
    snapshot_path.relative_to(_undo_root_for_workspace(workspace_obj))
    if not snapshot_path.is_file():
        raise WorkspaceError(f"Undo snapshot not found: {snapshot_rel}")

    target_path = workspace_obj.resolve(target_rel)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(snapshot_path.read_bytes())


def _delete_workspace_file(workspace: str, path: str) -> None:
    workspace_obj = Workspace(workspace)
    target_rel = _normalize_workspace_path(workspace_obj, path)
    if not target_rel:
        raise WorkspaceError("Invalid undo target path.")
    target_path = workspace_obj.resolve(target_rel)
    if target_path.exists():
        if not target_path.is_file():
            raise WorkspaceError(f"Refusing to delete non-file undo target: {target_rel}")
        target_path.unlink()


def _undo_root_for_workspace(workspace: Workspace) -> Path:
    from code_agent.services.trace import undo_root

    return undo_root(workspace.root).resolve()


def _undo_snapshots_by_path(turn: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    snapshots = turn.get("undo_snapshots")
    if not isinstance(snapshots, list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for snapshot in snapshots:
        if not isinstance(snapshot, Mapping):
            continue
        path = _normalize_git_path(str(snapshot.get("path") or ""))
        if path and path not in result:
            result[path] = dict(snapshot)
    return result


def _unique_snapshot_restores(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for item in items:
        path = _normalize_git_path(str(item.get("path") or ""))
        if not path or path in seen:
            continue
        seen.add(path)
        unique.append(item)
    return unique


def _latest_turn_file_changes(turn: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    changes = turn.get("file_changes")
    if not isinstance(changes, list):
        return []
    return [change for change in changes if isinstance(change, Mapping)]


def _normalize_undo_targets(workspace: Workspace, targets: list[str]) -> list[str]:
    normalized: list[str] = []
    for target in targets or ["."]:
        normalized.append(_normalize_workspace_path(workspace, target) or ".")
    return _unique_preserving_order(normalized) or ["."]


def _normalize_workspace_path(workspace: Workspace, path: str) -> str:
    try:
        return _normalize_git_path(workspace.relative(workspace.resolve(path)))
    except WorkspaceError:
        return ""


def _path_matches_targets(path: str, targets: list[str]) -> bool:
    if "." in targets:
        return True
    return any(path == target or path.startswith(f"{target.rstrip('/')}/") for target in targets)


def _normalize_git_path(path: str) -> str:
    return path.replace("\\", "/").strip().strip("/")


def _unique_preserving_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique


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
        _include_untracked, targets = _parse_undo_args(arg)
        scope = " ".join(targets)
        _print_raw(
            f"Restore latest agent changes only: {scope}\n\n"
            f"Current changes:\n{status}"
        )
        approved = Prompt.ask("Confirm agent-scoped restore?", choices=["y", "n"], default="n")
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
            _print_raw(
                f"skills: {store.skills_dir}\n"
                f"pending: {store.pending_dir}\n"
                f"legacy pending fallback: {legacy_pending_skill_root()}"
            )
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
        lines.append(f"{index}. {record.thread_id}")
        lines.append(f"   updated_at: {record.updated_at}")
        lines.append(f"   title: {record.title}")
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
                git_baseline_status=_run_git_status_porcelain(session.workspace),
                git_baseline_dirty_paths=_run_git_dirty_paths(session.workspace),
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
                    trace_recorder.record_interrupt(interrupts)
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
                if _is_model_timeout_error(exc):
                    timeout_seconds = AgentConfig(model=session.model).model_timeout_seconds
                    error_message = f"模型 API 超时（{timeout_seconds:g} 秒），请重试。"
                    console.print(f"[red]{error_message}[/red]")
                else:
                    error_message = f"Agent error: {exc}"
                    console.print(f"[red]Agent error:[/red] {exc}")
                _save_turn_trace(
                    session,
                    trace_recorder,
                    final_answer="",
                    user_feedback=error_message,
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
    from code_agent.ui.stream import (
        ModelWaitIndicator,
        chunk_begins_model_wait,
        final_answer_from_chunk,
        interrupt_from_chunk,
        render_stream_chunk,
    )
    from code_agent.services.trace import reset_current_trace_recorder, set_current_trace_recorder

    final_answer = None
    wait_indicator = ModelWaitIndicator()
    token = set_current_trace_recorder(trace_recorder) if trace_recorder is not None else None
    try:
        if isinstance(graph_input, Mapping):
            wait_indicator.start()
        for chunk in graph.stream(graph_input, config=config, stream_mode="updates"):
            if chunk_begins_model_wait(chunk):
                wait_indicator.start()
            else:
                wait_indicator.stop()
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
    finally:
        wait_indicator.stop()
        if token is not None:
            reset_current_trace_recorder(token)
    return final_answer


def _is_model_timeout_error(exc: BaseException) -> bool:
    """Recognize timeout wrappers raised by httpx/OpenAI/LangChain clients."""

    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        name = type(current).__name__.lower()
        message = str(current).lower()
        if isinstance(current, TimeoutError) or "timeout" in name or "timed out" in message:
            return True
        current = current.__cause__ or current.__context__
    return False


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
    from code_agent.services.trace import append_turn_trace

    existing_skills = SkillStore().list_skills()
    turn_trace = trace_recorder.build_trace(
        final_answer=final_answer,
        existing_skills=existing_skills,
        user_feedback=user_feedback,
    )
    path, project_trace = append_turn_trace(session.workspace, turn_trace)
    try:
        rel_path = Path(path).resolve().relative_to(Path(session.workspace).resolve()).as_posix()
    except ValueError:
        rel_path = str(path)
    console.print(f"[dim]Trace saved: {rel_path}[/dim]")
    return project_trace


def _review_trace_for_skills(session: Session, trace_payload: Mapping[str, Any]) -> None:
    from code_agent.services.skill_review import review_trace_for_skills

    result = review_trace_for_skills(
        trace=trace_payload,
        config=AgentConfig(model=session.model),
    )
    if result.status == "proposed":
        _prompt_skill_review_approval(result)
        return

    _print_skill_review_result(
        {
            "skill_review_status": result.status,
            "skill_review_message": result.message,
        }
    )


def _prompt_skill_review_approval(result: Any) -> None:
    if not result.skill_name or result.content is None:
        _print_raw("Skill review failed: proposal is missing skill name or content.")
        return

    store = SkillStore()
    try:
        diff = store.skill_diff(result.skill_name, result.content, tofile=f"proposed:{result.skill_name}")
    except Exception as exc:
        _print_raw(f"Skill review failed: {exc}")
        return

    console.print(
        f"\n[bold yellow]Skill approval required[/bold yellow] "
        f"[{result.action or 'write'}] {result.skill_name}"
    )
    _print_raw(f"{result.message}\n\n{diff}")
    approved = Prompt.ask("Approve this skill change?", choices=["y", "n"], default="n")
    if approved != "y":
        _print_raw("Skill change rejected.")
        return

    try:
        path = store.write_skill(result.skill_name, result.content)
    except Exception as exc:
        _print_raw(f"Skill write failed: {exc}")
        return
    _print_raw(f"Approved skill change; wrote {path}")


def _print_skill_review_result(state_values: Mapping[str, Any]) -> None:
    status = str(state_values.get("skill_review_status") or "")
    if status == "none":
        console.print(f"[dim]Skill review ran: {state_values.get('skill_review_message')}[/dim]")
    elif status == "error":
        console.print(f"[dim]{state_values.get('skill_review_message')}[/dim]")


if __name__ == "__main__":
    app()
