from __future__ import annotations

from rich.prompt import Confirm


def confirm_action(reason: str) -> bool:
    return Confirm.ask(reason, default=False)
