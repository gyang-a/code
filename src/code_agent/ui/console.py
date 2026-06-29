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
                f"Workspace: {escape(workspace)}\n"
                f"Model: {escape(model)}\n"
                "Type /help to list commands."
            ),
            border_style="cyan",
        )
    )


def print_help() -> None:
    table = Table(title="Slash Commands", show_header=True, header_style="bold cyan")
    table.add_column("Command")
    table.add_column("Description")
    rows = [
        ("/help", "Show this command list."),
        ("/clear", "Start a new conversation thread."),
        ("/model [name]", "Show or set the model."),
        ("/status", "Show current workspace and thread state."),
        ("/tools", "Show tool permission rules."),
        ("/diff", "Show current git diff."),
        ("/undo [path]", "Restore tracked git changes; add --include-untracked to delete new files."),
        ("/doctor", "Check dependencies, environment variables, and workspace status."),
        ("/usage", "Show local interaction and token usage."),
        ("/skills", "List saved skills; legacy pending review commands are available."),
        ("/sessions", "List saved conversation threads for this workspace."),
        ("/resume [id]", "List saved threads or switch to a saved thread."),
        ("/mcp", "Reserved MCP integration entry point."),
        ("/exit", "Exit the CLI."),
    ]
    for command, description in rows:
        table.add_row(command, description)
    console.print(table)
