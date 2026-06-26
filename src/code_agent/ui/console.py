from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def print_banner(workspace: str, model: str) -> None:
    console.print(
        Panel.fit(
            f"[bold]Code Agent[/bold]\nworkspace: {workspace}\nmodel: {model}\nType /help for commands.",
            border_style="cyan",
        )
    )


def print_help() -> None:
    table = Table(title="Slash Commands", show_header=True, header_style="bold cyan")
    table.add_column("Command")
    table.add_column("Description")
    rows = [
        ("/help", "Show slash commands."),
        ("/clear", "Start a fresh thread."),
        ("/model [name]", "Show or set the model."),
        ("/status", "Show session status."),
        ("/tools", "Show available safe commands."),
        ("/diff", "Show current git diff."),
        ("/doctor", "Check dependencies and workspace basics."),
        ("/usage", "Show local interaction counters."),
        ("/mcp", "Placeholder for future MCP integrations."),
        ("/exit", "Quit the CLI."),
    ]
    for command, description in rows:
        table.add_row(command, description)
    console.print(table)
