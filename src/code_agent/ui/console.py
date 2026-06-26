from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def print_banner(workspace: str, model: str) -> None:
    console.print(
        Panel.fit(
            f"[bold]Code Agent[/bold]\n工作区: {workspace}\n模型: {model}\n输入 /help 查看命令。",
            border_style="cyan",
        )
    )


def print_help() -> None:
    table = Table(title="Slash Commands", show_header=True, header_style="bold cyan")
    table.add_column("命令")
    table.add_column("说明")
    rows = [
        ("/help", "显示命令列表。"),
        ("/clear", "开启新的会话线程。"),
        ("/model [name]", "查看或设置模型。"),
        ("/status", "查看当前会话状态。"),
        ("/tools", "查看工具权限和 shell 风险分级规则。"),
        ("/diff", "查看当前 git diff。"),
        ("/doctor", "检查依赖、环境变量和工作区状态。"),
        ("/usage", "查看本地交互计数。"),
        ("/mcp", "预留的 MCP 集成入口。"),
        ("/exit", "退出 CLI。"),
    ]
    for command, description in rows:
        table.add_row(command, description)
    console.print(table)
