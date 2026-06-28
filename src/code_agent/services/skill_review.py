from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from code_agent.config import AgentConfig
from code_agent.services.skills import SkillStore, format_skill_index, normalize_skill_name
from code_agent.services.summarizer import truncate


DEFAULT_REVIEW_TOOL_THRESHOLD = 5
WRITE_TOOL_NAMES = {"patch_file", "create_file", "write_file", "delete_file"}


@dataclass(frozen=True)
class SkillReviewResult:
    status: str
    message: str
    pending_id: str | None = None


def count_tool_results(messages: list[BaseMessage]) -> int:
    return sum(isinstance(message, ToolMessage) for message in messages)


def should_review_skills(
    messages: list[BaseMessage],
    *,
    reviewed_tool_count: int,
    threshold: int = DEFAULT_REVIEW_TOOL_THRESHOLD,
) -> bool:
    current_tool_count = count_tool_results(messages)
    if current_tool_count - reviewed_tool_count >= threshold:
        return True
    return _has_write_tool_call(_messages_after_tool_count(messages, reviewed_tool_count))


def review_turn_for_skills(
    *,
    messages: list[BaseMessage],
    config: AgentConfig,
    thread_id: str,
    project_folder_name: str,
    reviewed_tool_count: int,
) -> SkillReviewResult:
    if not should_review_skills(messages, reviewed_tool_count=reviewed_tool_count):
        return SkillReviewResult(status="skipped", message="Skill review threshold not reached.")

    store = SkillStore()
    prompt = _build_review_prompt(
        messages=messages,
        skill_index=format_skill_index(store.list_skills()),
        project_folder_name=project_folder_name,
        reviewed_tool_count=reviewed_tool_count,
    )

    llm_kwargs: dict[str, Any] = {"temperature": 0}
    if config.api_key:
        llm_kwargs["api_key"] = config.api_key
    llm = init_chat_model(config.model, model_provider="deepseek", **llm_kwargs)

    try:
        response = llm.invoke(
            [
                SystemMessage(content=_REVIEW_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
        )
        data = _parse_json_object(str(response.content))
        decision = str(data.get("decision", "none")).lower()
        if decision != "stage":
            return SkillReviewResult(status="none", message=str(data.get("reason") or "No reusable skill update."))

        skill_name = normalize_skill_name(str(data.get("skill_name") or "learned-skill"))
        content = str(data.get("content") or "")
        reason = str(data.get("reason") or "Background review proposed a reusable skill update.")
        change = store.stage_skill_change(
            skill_name=skill_name,
            content=content,
            reason=reason,
            source={
                "thread_id": thread_id,
                "project_folder_name": project_folder_name,
                "tool_results_reviewed_from": reviewed_tool_count,
                "tool_results_total": count_tool_results(messages),
            },
        )
        return SkillReviewResult(
            status="staged",
            message=f"Staged pending skill change {change.id} for {change.skill_name}.",
            pending_id=change.id,
        )
    except Exception as exc:
        return SkillReviewResult(status="error", message=f"Skill review failed: {exc}")


def _build_review_prompt(
    *,
    messages: list[BaseMessage],
    skill_index: str,
    project_folder_name: str,
    reviewed_tool_count: int,
) -> str:
    transcript = _format_messages_for_review(messages, reviewed_tool_count=reviewed_tool_count)
    return f"""
Project folder: {project_folder_name}

Available global skills:
{skill_index}

Review the recent agent work below and decide whether a reusable global skill should be
created or updated. Only propose a skill when the transcript reveals durable procedural
knowledge that would help on future tasks across workspaces.

Recent transcript:
{truncate(transcript, 14_000)}
""".strip()


def _format_messages_for_review(
    messages: list[BaseMessage],
    *,
    reviewed_tool_count: int,
) -> str:
    lines: list[str] = []
    seen_tool_results = 0
    include = reviewed_tool_count <= 0

    for message in messages:
        if isinstance(message, ToolMessage):
            seen_tool_results += 1
            if seen_tool_results > reviewed_tool_count:
                include = True
        if not include:
            continue
        lines.append(_format_message(message))

    return "\n\n".join(lines)


def _format_message(message: BaseMessage) -> str:
    if isinstance(message, HumanMessage):
        role = "user"
    elif isinstance(message, AIMessage):
        role = "agent"
    elif isinstance(message, ToolMessage):
        role = "tool"
    else:
        role = message.type

    content = str(message.content or "")
    if isinstance(message, AIMessage):
        tool_calls = getattr(message, "tool_calls", None) or []
        if tool_calls:
            calls = [
                f"{call.get('name', 'tool')}({json.dumps(call.get('args') or {}, ensure_ascii=False)})"
                for call in tool_calls
            ]
            content = "\n".join(calls)

    return f"{role}:\n{truncate(content, 3000)}"


def _has_write_tool_call(messages: list[BaseMessage]) -> bool:
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        for tool_call in getattr(message, "tool_calls", None) or []:
            if tool_call.get("name") in WRITE_TOOL_NAMES:
                return True
    return False


def _messages_after_tool_count(
    messages: list[BaseMessage],
    reviewed_tool_count: int,
) -> list[BaseMessage]:
    if reviewed_tool_count <= 0:
        return messages

    seen_tool_results = 0
    for index, message in enumerate(messages):
        if isinstance(message, ToolMessage):
            seen_tool_results += 1
            if seen_tool_results >= reviewed_tool_count:
                return messages[index + 1 :]
    return []


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)

    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise
        data = json.loads(match.group(0))

    if not isinstance(data, dict):
        raise ValueError("Skill review response must be a JSON object.")
    return data


_REVIEW_SYSTEM_PROMPT = """
You are a background skill reviewer for a coding agent.

Decide whether the transcript contains durable, reusable procedural knowledge worth
turning into a global skill. Good skill material includes: repeated multi-step workflows,
non-obvious tool usage, user corrections about how future work should be done, recurring
failure recovery, or domain procedures likely to apply across future tasks.

Do not create a skill for one-off project facts, transient errors, secrets, private data,
or generic advice an expert coding agent already knows. Prefer updating an existing
umbrella skill over creating a narrow fragment when the skill index shows a good home.

Respond with only one JSON object. For no change:
{"decision":"none","reason":"..."}

For a proposed change:
{
  "decision":"stage",
  "skill_name":"lowercase-hyphen-name",
  "reason":"why this is reusable",
  "content":"---\\nname: lowercase-hyphen-name\\ndescription: What this skill does and when to use it.\\n---\\n\\n# Title\\n\\nConcise imperative instructions...\\n"
}

The content must be a complete SKILL.md file. The YAML frontmatter must contain only
name and description. Keep the skill concise, procedural, and free of project secrets.
""".strip()
