from __future__ import annotations

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

console = Console()

BANNER_ART = """
  CCCC      OOO     DDDDD      EEEEE
 C        O     O   D     D    E
C         O     O   D     D    EEEE
 C        O     O   D     D    E
  CCCC      OOO     DDDDD      EEEEE
"""


def print_banner(workspace: str, model: str) -> None:
    console.print(
        Panel.fit(
            (
                f"[cyan]{BANNER_ART}[/cyan]"
                "\n[bold]Code Agent[/bold]\n"
                f"工作区: {escape(workspace)}\n"
                f"模型: {escape(model)}\n"
                "输入 /help 查看命令。"
            ),
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
        ("/undo [path]", "回滚 git tracked 文件改动；加 --include-untracked 可删除新文件。"),
        ("/doctor", "检查依赖、环境变量和工作区状态。"),
        ("/usage", "查看本地交互计数。"),
        ("/mcp", "预留的 MCP 集成入口。"),
        ("/exit", "退出 CLI。"),
    ]
    for command, description in rows:
        table.add_row(command, description)
    console.print(table)
