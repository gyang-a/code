from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

from code_agent.config import AgentConfig
from code_agent.services.skills import SkillStore, format_skill_index, normalize_skill_name
from code_agent.services.summarizer import truncate
from code_agent.services.trace import reviewable_project_trace, should_review_trace


@dataclass(frozen=True)
class SkillReviewResult:
    status: str
    message: str
    skill_name: str | None = None
    action: str | None = None
    content: str | None = None
    source: dict[str, Any] | None = None


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
        content = _normalize_skill_markdown(
            str(data.get("proposed_content") or data.get("content") or ""),
            skill_name=target_skill,
        )
        reason = str(data.get("reason") or "Trace review proposed a reusable skill update.")
        operation = _operation_from_decision(data)
        confidence = _confidence_from_decision(data)
        latest_turn = _latest_turn(trace)
        action = operation or ("update" if store.skill_exists(target_skill) else "create")
        source = {
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
        }
        store.skill_diff(target_skill, content, tofile=f"proposed:{target_skill}")
        return SkillReviewResult(
            status="proposed",
            message=f"{reason} Confidence: {confidence:.2f}.",
            skill_name=target_skill,
            action=action,
            content=content,
            source=source,
        )
    except Exception as exc:
        return SkillReviewResult(status="error", message=f"Skill review failed: {exc}")


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


def _normalize_skill_markdown(content: str, *, skill_name: str | None = None) -> str:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:markdown|md)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
    if skill_name:
        text = _rewrite_skill_frontmatter(text, skill_name=skill_name)
    if text and not text.endswith("\n"):
        text += "\n"
    return text


def _rewrite_skill_frontmatter(content: str, *, skill_name: str) -> str:
    description = _extract_frontmatter_description(content)
    frontmatter = f"---\nname: {skill_name}\ndescription: {description}\n---\n\n"

    if not content.startswith("---\n"):
        return frontmatter + content.lstrip()

    end = content.find("\n---", 4)
    if end == -1:
        return frontmatter + content.lstrip("- \n")

    body_start = content.find("\n", end + 4)
    body = content[body_start + 1 :] if body_start != -1 else ""
    return frontmatter + body.lstrip()


def _extract_frontmatter_description(content: str) -> str:
    description = "Use when this learned workflow is relevant to the current task."
    if not content.startswith("---\n"):
        return description

    end = content.find("\n---", 4)
    if end == -1:
        return description

    for line in content[4:end].splitlines():
        if line.strip().lower().startswith("description:"):
            value = line.split(":", 1)[1].strip().strip("\"'")
            if value:
                return _clean_frontmatter_value(value)
    return description


def _clean_frontmatter_value(value: str) -> str:
    text = " ".join(value.split())
    text = text.replace('"', "'")
    return text[:240] or "Use when this learned workflow is relevant to the current task."


_TRACE_REVIEW_SYSTEM_PROMPT = """
You are a background skill reviewer for a coding agent.

Read the structured project trace and decide whether it reveals reusable procedural
knowledge that should become a skill. Pay attention to user feedback and corrections
across multiple turns, not only the latest request. Prefer durable workflows, repeated
recovery patterns, user corrections, workspace conventions that future runs should obey,
and non-obvious tool sequences. Do not create skills for one-off facts, secrets, private
data, or generic coding advice. If the user explicitly requests that relevant content be
written into the skill, you shall comply.

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
lowercase filename-like name, for example "frontend-viteconfig.md" or
"frontend-style-editing.md". proposed_content must be a complete SKILL.md file with YAML
frontmatter containing only name and description. The frontmatter name must be lowercase
letters, digits, and hyphens only, without ".md"; for target_skill
"frontend-viteconfig.md", use name: frontend-viteconfig. Keep proposed_content concise,
procedural, and free of secrets.
""".strip()
