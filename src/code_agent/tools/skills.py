from __future__ import annotations

from langchain_core.tools import tool

from code_agent.services.skills import SkillStore, format_skill_index
from code_agent.services.workspace import Workspace, WorkspaceError
from code_agent.tools.schemas import SkillsListInput, SkillViewInput


def build_skills_list_tool(workspace: Workspace):
    _ = workspace
    store = SkillStore()

    @tool(args_schema=SkillsListInput)
    def skills_list() -> str:
        """List available project skills by name and description."""
        return format_skill_index(store.list_skills())

    return skills_list


def build_skill_view_tool(workspace: Workspace):
    _ = workspace
    store = SkillStore()

    @tool(args_schema=SkillViewInput)
    def skill_view(name: str) -> str:
        """Read one project skill's SKILL.md instructions when its description is relevant."""
        try:
            return store.read_skill(name)
        except WorkspaceError as exc:
            return f"ERROR: {exc}"

    return skill_view
