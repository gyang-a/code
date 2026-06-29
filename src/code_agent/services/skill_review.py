from __future__ import annotations

import json
import re
from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from code_agent.config import AgentConfig
from code_agent.services.skills import SkillStore, format_skill_index, normalize_skill_name
from code_agent.services.summarizer import truncate
from code_agent.services.trace import reviewable_project_trace, should_review_trace


DEFAULT_REVIEW_TOOL_THRESHOLD = 4
WRITE_TOOL_NAMES = {"patch_file", "create_file", "write_file", "delete_file"}
WRITE_RESULT_PREFIXES = ("Patched ", "Created ", "Wrote ", "Deleted ")


@dataclass(frozen=True)
class SkillReviewResult:
    status: str
    message: str
    pending_id: str | None = None


def count_tool_results(messages: list[BaseMessage]) -> int:
    return sum(_message_type(message) == "tool" for message in messages)


def should_review_skills(
    messages: list[BaseMessage],
    *,
    reviewed_tool_count: int,
    threshold: int = DEFAULT_REVIEW_TOOL_THRESHOLD,
) -> bool:
    return skill_review_trigger_reason(
        messages,
        reviewed_tool_count=reviewed_tool_count,
        threshold=threshold,
    ) is not None


def skill_review_trigger_reason(
    messages: list[BaseMessage],
    *,
    reviewed_tool_count: int,
    threshold: int = DEFAULT_REVIEW_TOOL_THRESHOLD,
) -> str | None:
    current_tool_count = count_tool_results(messages)
    effective_reviewed_tool_count = _effective_reviewed_tool_count(
        reviewed_tool_count,
        current_tool_count=current_tool_count,
    )
    new_tool_count = current_tool_count - effective_reviewed_tool_count
    if new_tool_count >= threshold:
        return f"{new_tool_count} new tool results reached threshold {threshold}."

    recent_messages = _messages_after_tool_count(messages, effective_reviewed_tool_count)
    if _has_write_tool_call(recent_messages):
        return "Recent write tool call detected."
    if _has_write_tool_result(recent_messages):
        return "Recent write tool result detected."
    return None


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
        content = _normalize_skill_markdown(str(data.get("content") or ""))
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


def review_trace_for_skills(
    *,
    trace: Mapping[str, Any],
    config: AgentConfig,
) -> SkillReviewResult:
    if not should_review_trace(trace):
        return SkillReviewResult(status="skipped", message="Skill review threshold not reached.")

    store = SkillStore()
    prompt = _build_trace_review_prompt(
        trace=trace,
        skill_index=format_skill_index(store.list_skills()),
    )

    llm_kwargs: dict[str, Any] = {"temperature": 0}
    if config.api_key:
        llm_kwargs["api_key"] = config.api_key
    llm = init_chat_model(config.model, model_provider="deepseek", **llm_kwargs)

    try:
        response = llm.invoke(
            [
                SystemMessage(content=_TRACE_REVIEW_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
        )
        data = _parse_json_object(str(response.content))
        if not _bool_from_decision(data):
            return SkillReviewResult(status="none", message=str(data.get("reason") or "No reusable skill update."))

        target_skill = _target_skill_name(str(data.get("target_skill") or data.get("skill_name") or "learned-skill"))
        content = _normalize_skill_markdown(str(data.get("proposed_content") or data.get("content") or ""))
        reason = str(data.get("reason") or "Trace review proposed a reusable skill update.")
        operation = _operation_from_decision(data)
        confidence = _confidence_from_decision(data)
        latest_turn = _latest_turn(trace)
        change = store.stage_skill_change(
            skill_name=target_skill,
            content=content,
            reason=f"{reason} Confidence: {confidence:.2f}.",
            action=operation,
            source={
                "kind": "project_trace",
                "trace_path": str(trace.get("trace_path") or ""),
                "turn_id": str(latest_turn.get("turn_id") or trace.get("turn_id") or ""),
                "latest_turn_id": str(trace.get("latest_turn_id") or latest_turn.get("turn_id") or ""),
                "thread_id": str(latest_turn.get("thread_id") or trace.get("thread_id") or ""),
                "turn_count": int(trace.get("turn_count") or 0),
                "skill_type": str(data.get("skill_type") or "global"),
                "confidence": confidence,
                "decision": {
                    "should_create_skill": True,
                    "reason": reason,
                    "skill_type": str(data.get("skill_type") or "global"),
                    "target_skill": target_skill,
                    "operation": operation,
                    "confidence": confidence,
                },
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


def _build_trace_review_prompt(
    *,
    trace: Mapping[str, Any],
    skill_index: str,
) -> str:
    review_trace = reviewable_project_trace(trace)
    trace_json = json.dumps(review_trace, indent=2, ensure_ascii=False)
    return f"""
Available global skills:
{skill_index}

Review this structured project trace. It is the source of truth for the user's goals,
feedback across turns, tool calls, tool results, file changes, errors, validation, final
answers, and existing skills. Do not infer tool execution from compressed conversation
memory.

Project trace:
{truncate(trace_json, 18_000)}
""".strip()


def _format_messages_for_review(
    messages: list[BaseMessage],
    *,
    reviewed_tool_count: int,
) -> str:
    lines: list[str] = []
    seen_tool_results = 0
    include = reviewed_tool_count <= 0

    effective_reviewed_tool_count = _effective_reviewed_tool_count(
        reviewed_tool_count,
        current_tool_count=count_tool_results(messages),
    )

    for message in messages:
        if _message_type(message) == "tool":
            seen_tool_results += 1
            if seen_tool_results > effective_reviewed_tool_count:
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
        role = _message_type(message)

    content = str(_message_content(message) or "")
    if role == "agent" or _message_type(message) == "ai":
        tool_calls = _message_tool_calls(message)
        if tool_calls:
            calls = [
                f"{_tool_call_name(call)}({json.dumps(_tool_call_args(call), ensure_ascii=False)})"
                for call in tool_calls
            ]
            content = "\n".join(calls)

    return f"{role}:\n{truncate(content, 3000)}"


def _has_write_tool_call(messages: list[BaseMessage]) -> bool:
    for message in messages:
        if _message_type(message) != "ai":
            continue
        for tool_call in _message_tool_calls(message):
            if _tool_call_name(tool_call) in WRITE_TOOL_NAMES:
                return True
    return False


def _has_write_tool_result(messages: list[BaseMessage]) -> bool:
    for message in messages:
        if _message_type(message) != "tool":
            continue
        if _tool_message_name(message) in WRITE_TOOL_NAMES:
            return True
        content = str(_message_content(message) or "")
        if content.startswith(WRITE_RESULT_PREFIXES):
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
        if _message_type(message) == "tool":
            seen_tool_results += 1
            if seen_tool_results >= reviewed_tool_count:
                return messages[index + 1 :]
    return []


def _effective_reviewed_tool_count(reviewed_tool_count: int, *, current_tool_count: int) -> int:
    if reviewed_tool_count > current_tool_count:
        return 0
    return max(0, reviewed_tool_count)


def _message_type(message: BaseMessage) -> str:
    if isinstance(message, dict):
        return str(message.get("type") or message.get("role") or "")
    return str(getattr(message, "type", ""))


def _message_content(message: BaseMessage) -> Any:
    if isinstance(message, dict):
        return message.get("content")
    return getattr(message, "content", None)


def _message_tool_calls(message: BaseMessage) -> list[Any]:
    if isinstance(message, dict):
        tool_calls = message.get("tool_calls") or []
    else:
        tool_calls = getattr(message, "tool_calls", None) or []
    return list(tool_calls) if isinstance(tool_calls, list) else []


def _tool_message_name(message: BaseMessage) -> str:
    if isinstance(message, dict):
        return str(message.get("name") or "")
    return str(getattr(message, "name", "") or "")


def _tool_call_name(tool_call: Any) -> str:
    if isinstance(tool_call, dict):
        return str(tool_call.get("name") or "")
    return str(getattr(tool_call, "name", "") or "")


def _tool_call_args(tool_call: Any) -> Any:
    if isinstance(tool_call, dict):
        return tool_call.get("args") or {}
    return getattr(tool_call, "args", {}) or {}


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


def _bool_from_decision(data: Mapping[str, Any]) -> bool:
    value = data.get("should_create_skill")
    if isinstance(value, bool):
        return value
    decision = str(data.get("decision") or "").lower()
    return decision == "stage"


def _target_skill_name(value: str) -> str:
    name = value.replace("\\", "/").rsplit("/", 1)[-1]
    if name.lower().endswith(".md"):
        name = name[:-3]
    if name.lower() == "skill":
        name = "learned-skill"
    return normalize_skill_name(name)


def _operation_from_decision(data: Mapping[str, Any]) -> str:
    operation = str(data.get("operation") or "").lower()
    if operation in {"create", "update"}:
        return operation
    return ""


def _confidence_from_decision(data: Mapping[str, Any]) -> float:
    try:
        confidence = float(data.get("confidence", 0))
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(confidence, 1.0))


def _latest_turn(trace: Mapping[str, Any]) -> Mapping[str, Any]:
    turns = trace.get("turns")
    if isinstance(turns, list) and turns and isinstance(turns[-1], Mapping):
        return turns[-1]
    return trace


def _normalize_skill_markdown(content: str) -> str:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:markdown|md)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
    if text and not text.endswith("\n"):
        text += "\n"
    return text


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


_TRACE_REVIEW_SYSTEM_PROMPT = """
You are a background skill reviewer for a coding agent.

Read the structured project trace and decide whether it reveals reusable procedural
knowledge that should become a skill. Pay attention to user feedback and corrections
across multiple turns, not only the latest request. Prefer durable workflows, repeated
recovery patterns, user corrections, workspace conventions that future runs should obey,
and non-obvious tool sequences. Do not create skills for one-off facts, secrets, private
data, or generic coding advice.

Respond with only one JSON object matching this schema:
{
  "should_create_skill": false,
  "reason": "why no reusable skill is needed",
  "skill_type": "global",
  "target_skill": "",
  "operation": "create",
  "proposed_content": "",
  "confidence": 0.0
}

When a skill is useful, set should_create_skill to true. skill_type must be "global" or
"project". operation must be "create" or "update". target_skill must be a concise
lowercase filename-like name, for example "frontend-style-editing.md". proposed_content
must be a complete SKILL.md file with YAML frontmatter containing only name and
description. Keep proposed_content concise, procedural, and free of secrets.
""".strip()
