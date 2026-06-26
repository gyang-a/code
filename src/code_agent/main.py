from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import typer
from rich.prompt import Prompt

from code_agent.config import AgentConfig, DEFAULT_MAX_ITERATIONS, DEFAULT_MODEL
from code_agent.services.summarizer import truncate
from code_agent.services.workspace import Workspace, WorkspaceError
from code_agent.tools.safety import available_commands
from code_agent.ui.console import console, print_banner, print_help

app = typer.Typer(no_args_is_help=True)


@dataclass
class Session:
    workspace: str
    model: str = DEFAULT_MODEL
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
        "input_kind": None,
        "project_context": None,
        "plan": [],
        "current_step": None,
        "iteration_count": 0,
        "max_iterations": DEFAULT_MAX_ITERATIONS,
        "changed_files": [],
        "last_diff": None,
        "test_command": None,
        "test_result": None,
        "needs_approval": False,
        "approval_reason": None,
        "rejected_reason": None,
        "tool_errors": [],
        "diff_summary": None,
        "final_answer": None,
    }


def _run_git_diff(workspace: str) -> str:
    try:
        result = subprocess.run(
            ["git", "diff", "--", "."],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=10,
            shell=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return f"ERROR: {exc}"
    return truncate(result.stdout + result.stderr)


def _doctor(workspace: str) -> None:
    table_rows = [
        ("workspace", "ok" if Path(workspace).is_dir() else "missing"),
        ("OPENAI_API_KEY", "set" if os.getenv("OPENAI_API_KEY") else "missing"),
        ("git", "found" if shutil.which("git") else "missing"),
        ("rg", "found" if shutil.which("rg") else "missing; Python fallback will be used"),
        ("langgraph", "found" if importlib.util.find_spec("langgraph") else "missing"),
        ("langchain_openai", "found" if importlib.util.find_spec("langchain_openai") else "missing"),
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
        console.print(f"Started fresh thread: {session.thread_id}")
    elif name == "/model":
        if arg:
            session.model = arg
            console.print(f"Model set to {session.model}")
        else:
            console.print(f"Current model: {session.model}")
    elif name == "/status":
        console.print(f"workspace: {session.workspace}")
        console.print(f"thread_id: {session.thread_id}")
        console.print(f"model: {session.model}")
        console.print(f"interactions: {session.interactions}")
    elif name == "/tools":
        try:
            workspace = Workspace(session.workspace)
            commands = available_commands(workspace)
            if not commands:
                console.print("No validation commands detected.")
            for command_name, spec in commands.items():
                console.print(f"[bold]{command_name}[/bold]: {' '.join(spec.argv)}")
        except WorkspaceError as exc:
            console.print(f"[red]ERROR:[/red] {exc}")
    elif name == "/diff":
        diff = _run_git_diff(session.workspace)
        console.print(diff or "No git diff.")
    elif name == "/doctor":
        _doctor(session.workspace)
    elif name == "/usage":
        console.print(f"interactions: {session.interactions}")
        console.print(f"thread_id: {session.thread_id}")
    elif name == "/mcp":
        console.print("MCP integration is not configured in this MVP.")
    else:
        console.print(f"Unknown slash command: {name}. Type /help.")

    return True


@app.command()
def chat(
    workspace: str = typer.Argument(".", help="Workspace directory for the code agent."),
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m", help="OpenAI chat model name."),
) -> None:
    """Start an interactive workspace-safe code agent."""
    try:
        resolved_workspace = str(Path(workspace).expanduser().resolve())
        Workspace(resolved_workspace)
    except WorkspaceError as exc:
        raise typer.BadParameter(str(exc)) from exc

    session = Session(workspace=resolved_workspace, model=model)
    print_banner(session.workspace, session.model)

    graph = None
    while True:
        user_input = Prompt.ask("\n[bold cyan]you[/bold cyan]").strip()
        if not user_input:
            continue

        if user_input.startswith("/"):
            if not _handle_slash(user_input, session):
                break
            continue

        if graph is None:
            try:
                from code_agent.graph import build_graph

                graph = build_graph(session.workspace, AgentConfig(model=session.model))
            except Exception as exc:
                if "Missing credentials" in str(exc):
                    console.print("[red]Failed to initialize graph:[/red] missing OpenAI credentials.")
                    console.print("Set OPENAI_API_KEY before natural-language tasks, or run /doctor.")
                else:
                    console.print(f"[red]Failed to initialize graph:[/red] {exc}")
                    console.print("Run /doctor to inspect missing dependencies or environment variables.")
                continue

        session.interactions += 1
        config = {"configurable": {"thread_id": session.thread_id}}
        console.print("[dim]agent is inspecting the workspace...[/dim]")

        try:
            result = graph.invoke(_initial_state(session, user_input), config=config)
        except Exception as exc:
            console.print(f"[red]Agent error:[/red] {exc}")
            continue

        session.tool_loops += result.get("iteration_count", 0)
        answer = result.get("final_answer") or result["messages"][-1].content
        console.print("\n[bold green]agent[/bold green]")
        console.print(answer)


if __name__ == "__main__":
    app()
