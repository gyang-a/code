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

import typer
from langgraph.types import Command
from rich.prompt import Prompt

from code_agent.config import AgentConfig, DEFAULT_MODEL
from code_agent.services.env import load_dotenv
from code_agent.services.sandbox import SandboxPolicy, describe_sandbox_policy
from code_agent.services.summarizer import truncate
from code_agent.services.workspace import Workspace, WorkspaceError
from code_agent.tools.safety import describe_permission_policy
from code_agent.ui.approval import format_approval_summary
from code_agent.ui.console import console, print_banner, print_help
from code_agent.ui.stream import final_answer_from_chunk, interrupt_from_chunk, render_stream_chunk

app = typer.Typer(no_args_is_help=True)


@dataclass
class Session:
    workspace: str
    model: str = DEFAULT_MODEL
    env_file: str | None = None
    thread_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    interactions: int = 0
    tool_loops: int = 0

    def reset(self) -> None:
        self.thread_id = str(uuid.uuid4())
        self.interactions = 0
        self.tool_loops = 0


def _initial_state(session: Session, user_input: str) -> dict:
    from langchain_core.messages import HumanMessage

    return {
        "messages": [HumanMessage(content=user_input)],
        "workspace": session.workspace,
        "user_goal": user_input,
        "changed_files": [],
        "did_write": False,
        "test_command": None,
        "test_result": None,
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
    sandbox_policy = _sandbox_policy_from_env()
    table_rows = [
        ("workspace", "ok" if Path(workspace).is_dir() else "missing"),
        (".env", "found" if env_path.exists() else "missing"),
        ("DEEPSEEK_API_KEY", "set" if os.getenv("DEEPSEEK_API_KEY") else "missing"),
        ("CODE_AGENT_MODEL", os.getenv("CODE_AGENT_MODEL") or "unset"),
        ("shell_sandbox", describe_sandbox_policy(sandbox_policy)),
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
    elif name == "/mcp":
        console.print("MCP integration is not configured in this MVP.")
    else:
        console.print(f"Unknown slash command: {name}. Type /help to list commands.")

    return True


def _is_slash_command(user_input: str) -> bool:
    return user_input.lstrip().startswith("/")


def _sandbox_policy_from_env() -> SandboxPolicy:
    config = AgentConfig()
    return SandboxPolicy(
        backend=config.shell_sandbox_backend,
        docker_image=config.docker_image,
        allow_network=config.docker_allow_network,
    )


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
        if action.get("name") in {"run_shell", "run_command"} or action.get("tool") in {"run_shell", "run_command"}:
            console.print(
                "[yellow]  This approves the entire shell command. "
                "Package downloads, scaffolding, and child commands inside it will not prompt separately. "
                "File inspection/editing should use dedicated tools, not shell.[/yellow]"
            )
        approved = Prompt.ask("Approve this action?", choices=["y", "n"], default="n")
        if approved == "y":
            decisions.append({"type": "approve"})
        else:
            decisions.append({"type": "reject", "message": "Action rejected by user."})
    if not decisions:
        decisions.append({"type": "reject", "message": "Action rejected by user."})
    return decisions


@app.command()
def chat(
    workspace: str = typer.Argument(".", help="Workspace directory for the code agent."),
    model: str | None = typer.Option(None, "--model", "-m", help="Override the default model name."),
) -> None:
    """Start an interactive workspace-safe code agent."""
    try:
        resolved_workspace = str(Path(workspace).expanduser().resolve())
        Workspace(resolved_workspace)
    except WorkspaceError as exc:
        raise typer.BadParameter(str(exc)) from exc

    env_path = Path(resolved_workspace) / ".env"
    loaded_env = load_dotenv(env_path)
    selected_model = model or os.getenv("CODE_AGENT_MODEL", DEFAULT_MODEL)
    session = Session(
        workspace=resolved_workspace,
        model=selected_model,
        env_file=str(env_path) if loaded_env else None,
    )
    print_banner(session.workspace, session.model)
    console.print(f"[dim]shell sandbox: {describe_sandbox_policy(_sandbox_policy_from_env())}[/dim]")
    if loaded_env:
        console.print(f"[dim]Loaded environment variables from {env_path}[/dim]")

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

                graph = build_graph(session.workspace, AgentConfig(model=session.model))
            except Exception as exc:
                if "Missing credentials" in str(exc):
                    console.print("[red]Graph initialization failed[/red] Missing DeepSeek credentials.")
                    console.print("Set DEEPSEEK_API_KEY in .env or run /doctor to inspect the environment.")
                else:
                    console.print(f"[red]Graph initialization failed[/red] {exc}")
                    console.print("Run /doctor to inspect dependencies and environment variables.")
                continue

        session.interactions += 1
        config = {"configurable": {"thread_id": session.thread_id}}
        console.print("[dim]Agent started[/dim]")

        try:
            final_answer = _run_graph_stream(graph, _initial_state(session, user_input), config)
            while final_answer is None:
                state = graph.get_state(config)
                interrupts = state.interrupts
                if not interrupts:
                    values = state.values
                    final_answer = values.get("final_answer") or values["messages"][-1].content
                    break
                decisions = _prompt_approval_decisions(interrupts[0].value)
                if any(decision.get("type") in {"approve", "edit"} for decision in decisions):
                    console.print(
                        "[dim]Executing approved action(s). "
                        "Long-running shell output is captured and shown when the tool finishes.[/dim]"
                    )
                final_answer = _run_graph_stream(graph, Command(resume={"decisions": decisions}), config)
        except Exception as exc:
            console.print(f"[red]Agent error:[/red] {exc}")
            continue

        state = graph.get_state(config)
        session.tool_loops += state.values.get("run_model_call_count", 0)
        answer = final_answer or state.values.get("final_answer") or state.values["messages"][-1].content
        console.print("\n[bold green]Agent[/bold green]")
        _print_raw(str(answer))


def _run_graph_stream(graph, graph_input, config: dict) -> str | None:
    final_answer = None
    for chunk in graph.stream(graph_input, config=config, stream_mode="updates"):
        render_stream_chunk(chunk)
        interrupt_value = interrupt_from_chunk(chunk)
        if interrupt_value is not None:
            return None
        chunk_answer = final_answer_from_chunk(chunk)
        if chunk_answer:
            final_answer = chunk_answer
    return final_answer


if __name__ == "__main__":
    app()
