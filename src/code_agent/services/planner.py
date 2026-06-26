from __future__ import annotations


def default_plan(user_goal: str) -> list[str]:
    return [
        "Inspect project structure and relevant manifests.",
        "Search for files related to the user's goal.",
        "Read the smallest useful set of files.",
        "Apply targeted edits through safe tools if needed.",
        "Run an available validation command.",
        "Summarize changes, validation, and remaining risks.",
    ]
