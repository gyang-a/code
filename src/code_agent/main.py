from __future__ import annotations

import importlib.util
import os
import shlex
import shutil
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import typer
from langgraph.types import Command
from rich.prompt import Prompt

from code_agent.config import AgentConfig, DEFAULT_MAX_ITERATIONS, DEFAULT_MODEL
from code_agent.services.env import load_dotenv
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
        "context_summary": None,
        "recent_files": [],
        "compaction_count": 0,
        "iteration_count": 0,
        "max_iterations": DEFAULT_MAX_ITERATIONS,
        "changed_files": [],
        "did_write": False,
        "test_command": None,
        "test_result": None,
        "needs_approval": False,
        "approval_reason": None,
        "pending_approval": None,
        "rejected_reason": None,
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
        outputs.append("\n已回滚 tracked 变更。仍有未回滚的变更：\n")
        outputs.append(remaining)
        if not include_untracked:
            outputs.append("\n提示：如需删除未跟踪的新文件，使用 /undo --include-untracked。")
    else:
        outputs.append("已回滚工作区变更。")
    return truncate("".join(outputs))


def _print_raw(text: str) -> None:
    console.print(text, markup=False, highlight=False)


def _doctor(workspace: str) -> None:
    env_path = Path(workspace) / ".env"
    table_rows = [
        ("工作区", "正常" if Path(workspace).is_dir() else "缺失"),
        (".env", "已找到" if env_path.exists() else "缺失"),
        ("DEEPSEEK_API_KEY", "已设置" if os.getenv("DEEPSEEK_API_KEY") else "缺失"),
        ("CODE_AGENT_MODEL", os.getenv("CODE_AGENT_MODEL") or "未设置"),
        ("git", "已找到" if shutil.which("git") else "缺失"),
        ("rg", "已找到" if shutil.which("rg") else "缺失，将使用 Python fallback"),
        ("langgraph", "已找到" if importlib.util.find_spec("langgraph") else "缺失"),
        ("langchain", "已找到" if importlib.util.find_spec("langchain") else "缺失"),
        ("langchain_deepseek", "已找到" if importlib.util.find_spec("langchain_deepseek") else "缺失"),
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
        console.print(f"已开启新的会话线程: {session.thread_id}")
    elif name == "/model":
        if arg:
            session.model = arg
            console.print(f"模型已设置为: {session.model}")
        else:
            console.print(f"当前模型: {session.model}")
    elif name == "/status":
        console.print(f"工作区: {session.workspace}")
        console.print(f"thread_id: {session.thread_id}")
        console.print(f"模型: {session.model}")
        console.print(f"交互次数: {session.interactions}")
    elif name == "/tools":
        _print_raw(describe_permission_policy())
    elif name == "/diff":
        diff = _run_git_diff(session.workspace)
        _print_raw(diff or "当前没有 git diff。")
    elif name in {"/undo", "/revert"}:
        status = _run_git_status_porcelain(session.workspace)
        if not status:
            _print_raw("当前没有可回滚的 git 工作区变更。")
            return True
        include_untracked, targets = _parse_undo_args(arg)
        scope = " ".join(targets)
        action = "回滚 tracked 文件改动并删除未跟踪文件" if include_untracked else "回滚 tracked 文件改动"
        _print_raw(f"{action}: {scope}\n\n当前文件改动:\n{status}")
        approved = Prompt.ask("确认回滚这些文件改动？", choices=["y", "n"], default="n")
        if approved == "y":
            _print_raw(_run_git_undo(session.workspace, arg))
        else:
            _print_raw("已取消回滚。")
    elif name == "/doctor":
        _doctor(session.workspace)
    elif name == "/usage":
        console.print(f"交互次数: {session.interactions}")
        console.print(f"thread_id: {session.thread_id}")
    elif name == "/mcp":
        console.print("当前 MVP 尚未配置 MCP 集成。")
    else:
        console.print(f"未知 slash command: {name}。输入 /help 查看命令。")

    return True


def _is_slash_command(user_input: str) -> bool:
    return user_input.lstrip().startswith("/")


@app.command()
def chat(
    workspace: str = typer.Argument(".", help="代码智能体使用的工作区目录。"),
    model: str | None = typer.Option(None, "--model", "-m", help="覆盖默认模型名称。"),
) -> None:
    """启动一个带工作区沙箱的交互式代码智能体。"""
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
    if loaded_env:
        console.print(f"[dim]已从 {env_path} 加载环境变量[/dim]")

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
                    console.print("[red]Graph 初始化失败:[/red] 缺少 DeepSeek 凭证。")
                    console.print("请先在 .env 中设置 DEEPSEEK_API_KEY，或运行 /doctor 检查环境。")
                else:
                    console.print(f"[red]Graph 初始化失败:[/red] {exc}")
                    console.print("请运行 /doctor 检查依赖和环境变量。")
                continue

        session.interactions += 1
        config = {"configurable": {"thread_id": session.thread_id}}
        console.print("[dim]Agent 已启动[/dim]")

        try:
            final_answer = _run_graph_stream(graph, _initial_state(session, user_input), config)
            while final_answer is None:
                state = graph.get_state(config)
                interrupts = state.interrupts
                if not interrupts:
                    values = state.values
                    final_answer = values.get("final_answer") or values["messages"][-1].content
                    break
                interrupt_value = interrupts[0].value
                action = interrupt_value.get("action")
                reason = interrupt_value.get("reason", "该操作需要确认。")
                console.print(f"\n[bold yellow]需要审批[/bold yellow] {format_approval_summary(action, reason)}")
                approved = Prompt.ask("是否批准这个 Level 2 操作？", choices=["y", "n"], default="n")
                final_answer = _run_graph_stream(graph, Command(resume={"approved": approved == "y"}), config)
        except Exception as exc:
            console.print(f"[red]Agent 错误:[/red] {exc}")
            continue

        state = graph.get_state(config)
        session.tool_loops += state.values.get("iteration_count", 0)
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
