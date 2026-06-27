from __future__ import annotations

import subprocess
from functools import partial
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from code_agent.config import AgentConfig
from code_agent.prompts import SYSTEM_PROMPT
from code_agent.services.context import compact_messages, should_compact_messages
from code_agent.services.metadata import build_turn_metadata
from code_agent.services.permissions import classify_tool_call
from code_agent.services.sandbox import SandboxPolicy
from code_agent.services.summarizer import truncate
from code_agent.services.workspace import Workspace
from code_agent.state import AgentState
from code_agent.tools import (
    build_create_file_tool,
    build_delete_file_tool,
    build_find_files_tool,
    build_git_diff_tool,
    build_git_status_tool,
    build_list_files_tool,
    build_patch_file_tool,
    build_read_file_tool,
    build_run_command_tool,
    build_search_text_tool,
    build_write_file_tool,
)


WRITE_TOOL_NAMES = {"patch_file", "create_file", "write_file", "delete_file"}
WRITE_COMMAND_PREFIXES = (
    "npm install",
    "npm i",
    "pnpm add",
    "pnpm install",
    "yarn add",
    "yarn install",
    "pip install",
    "python -m pip install",
    "uv add",
    "uv pip install",
    "docker compose up",
    "docker-compose up",
)


def build_tools(
    workspace: Workspace,
    *,
    read_max_lines: int | None = None,
    tool_output_limit: int | None = None,
    sandbox_policy: SandboxPolicy | None = None,
):
    return [
        build_list_files_tool(workspace),
        build_read_file_tool(
            workspace,
            default_max_lines=read_max_lines,
            output_limit=tool_output_limit,
        ),
        build_search_text_tool(workspace),
        build_find_files_tool(workspace),
        build_patch_file_tool(workspace),
        build_create_file_tool(workspace),
        build_write_file_tool(workspace),
        build_delete_file_tool(workspace),
        build_run_command_tool(workspace, sandbox_policy=sandbox_policy),
        build_git_status_tool(workspace),
        build_git_diff_tool(workspace),
    ]


def _execute_node(
    state: AgentState,
    *,
    workspace: Workspace,
    tools_by_name: dict,
    max_tool_calls_per_turn: int,
):
    last_message = state["messages"][-1]
    tool_calls = list(getattr(last_message, "tool_calls", None) or [])
    if len(tool_calls) > max_tool_calls_per_turn:
        return {
            "messages": [
                ToolMessage(
                    content=(
                        "TOOL_LIMIT_EXCEEDED: Too many tool calls in one assistant turn. "
                        f"Limit is {max_tool_calls_per_turn}. "
                        "Please continue with a smaller batch."
                    ),
                    tool_call_id=tool_call["id"],
                )
                for tool_call in tool_calls
                if "id" in tool_call
            ]
        }

    reviews = [_review_tool_call(workspace, tools_by_name, tool_call) for tool_call in tool_calls]
    action_requests = [review["action_request"] for review in reviews if review.get("requires_approval")]
    decisions_by_id: dict[str, dict[str, Any]] = {}
    if action_requests:
        resume_value = interrupt(_approval_interrupt_payload(action_requests))
        decisions = _normalize_approval_decisions(resume_value, len(action_requests))
        decisions_by_id = {
            str(action_request["tool_call_id"]): decision
            for action_request, decision in zip(action_requests, decisions, strict=False)
        }

    results: list[ToolMessage] = []
    for tool_call, review in zip(tool_calls, reviews, strict=False):
        tool_name = tool_call.get("name")
        tool_call_id = tool_call.get("id")
        args = dict(tool_call.get("args") or {})
        if not tool_call_id:
            continue
        if tool_name not in tools_by_name:
            results.append(ToolMessage(content=f"ERROR: Unknown tool: {tool_name}", tool_call_id=tool_call_id))
            continue
        if review.get("blocked"):
            results.append(ToolMessage(content=f"REJECTED[level_3]: {review['reason']}", tool_call_id=tool_call_id))
            continue
        if review.get("requires_approval"):
            decision = decisions_by_id.get(str(tool_call_id), {"type": "reject", "message": "Action rejected."})
            decision_type = str(decision.get("type") or "").lower()
            if decision_type in {"approve", "approved", "accept"}:
                pass
            elif decision_type == "edit":
                args = _edited_tool_args(args, decision)
            else:
                results.append(
                    ToolMessage(
                        content=_synthetic_tool_message_from_decision(decision),
                        tool_call_id=tool_call_id,
                    )
                )
                continue

        results.append(ToolMessage(content=_invoke_tool(tools_by_name[tool_name], args), tool_call_id=tool_call_id))

    return {"messages": results}


def _review_tool_call(workspace: Workspace, tools_by_name: dict, tool_call: dict) -> dict[str, Any]:
    tool_name = str(tool_call.get("name") or "")
    args = dict(tool_call.get("args") or {})
    tool_call_id = str(tool_call.get("id") or "")
    if tool_name not in tools_by_name:
        return {}
    try:
        decision = classify_tool_call(workspace, tool_name, args)
    except Exception as exc:
        return {"blocked": True, "reason": str(exc)}

    if decision.risk.value == "level_3" or (not decision.allowed and not decision.requires_approval):
        return {"blocked": True, "reason": decision.reason}
    if not decision.requires_approval:
        return {}

    return {
        "requires_approval": True,
        "action_request": {
            "name": tool_name,
            "tool": tool_name,
            "args": args,
            "description": decision.reason,
            "reason": decision.reason,
            "tool_call_id": tool_call_id,
        },
    }


def _approval_interrupt_payload(action_requests: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "action_requests": action_requests,
        "review_configs": [
            {
                "action_name": request["name"],
                "tool_call_id": request["tool_call_id"],
                "allowed_decisions": ["approve", "edit", "reject", "respond"],
            }
            for request in action_requests
        ],
    }


def _normalize_approval_decisions(value: Any, expected_count: int) -> list[dict[str, Any]]:
    if isinstance(value, dict) and isinstance(value.get("decisions"), list):
        raw_decisions = value["decisions"]
    elif isinstance(value, list):
        raw_decisions = value
    elif isinstance(value, dict) and "approved" in value:
        decision_type = "approve" if value.get("approved") else "reject"
        raw_decisions = [{"type": decision_type, "message": value.get("message")} for _ in range(expected_count)]
    elif isinstance(value, bool):
        raw_decisions = [{"type": "approve" if value else "reject"} for _ in range(expected_count)]
    else:
        raw_decisions = [value]

    decisions = [_normalize_decision(decision) for decision in raw_decisions[:expected_count]]
    while len(decisions) < expected_count:
        decisions.append({"type": "reject", "message": "Action rejected."})
    return decisions


def _normalize_decision(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        normalized = dict(value)
        normalized["type"] = str(normalized.get("type") or "reject").lower()
        return normalized
    if isinstance(value, str):
        return {"type": value.lower()}
    if isinstance(value, bool):
        return {"type": "approve" if value else "reject"}
    return {"type": "reject", "message": "Action rejected."}


def _edited_tool_args(original_args: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    if isinstance(decision.get("args"), dict):
        return dict(decision["args"])
    edited_action = decision.get("edited_action")
    if isinstance(edited_action, dict) and isinstance(edited_action.get("args"), dict):
        return dict(edited_action["args"])
    return dict(original_args)


def _synthetic_tool_message_from_decision(decision: dict[str, Any]) -> str:
    message = decision.get("message") or decision.get("content")
    if message:
        return str(message)
    decision_type = str(decision.get("type") or "reject").lower()
    if decision_type == "respond":
        return "No tool execution. User response provided."
    return "Action rejected by user."


def _invoke_tool(tool, args: dict[str, Any]) -> str:
    try:
        return str(tool.invoke(args))
    except Exception as exc:
        return f"ERROR: Tool execution failed: {exc}"


def _git_output(workspace: Workspace, args: list[str], timeout: int = 10) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=workspace.root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            shell=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return f"ERROR: {exc}"
    output = result.stdout if result.returncode == 0 else result.stdout + result.stderr
    return truncate(output)


def _recent_tool_messages(state: AgentState) -> list[ToolMessage]:
    messages = state["messages"]
    recent: list[ToolMessage] = []
    for message in reversed(messages):
        if isinstance(message, ToolMessage):
            recent.append(message)
            continue
        if recent:
            break
    return list(reversed(recent))


def build_graph(workspace_path: str, config: AgentConfig | None = None):
    agent_config = config or AgentConfig()
    workspace = Workspace(
        workspace_path,
        read_limit=agent_config.file_read_limit,
        read_max_lines=agent_config.file_read_max_lines,
        exclude_globs=agent_config.exclude_globs,
    )
    tools = build_tools(
        workspace,
        read_max_lines=agent_config.file_read_max_lines,
        tool_output_limit=agent_config.tool_output_limit,
        sandbox_policy=SandboxPolicy(
            backend=agent_config.shell_sandbox_backend,
            docker_image=agent_config.docker_image,
            allow_network=agent_config.docker_allow_network,
        ),
    )
    tools_by_name = {tool.name: tool for tool in tools}
    llm_kwargs = {"temperature": 0}
    if agent_config.api_key:
        llm_kwargs["api_key"] = agent_config.api_key
    llm = init_chat_model(
        agent_config.model,
        model_provider="deepseek",
        **llm_kwargs,
    )
    summary_llm = init_chat_model(
        agent_config.model,
        model_provider="deepseek",
        **llm_kwargs,
    )
    llm_with_tools = llm.bind_tools(tools)
    compaction_char_limit = min(
        agent_config.context_char_limit,
        int(agent_config.context_window_chars * agent_config.context_compaction_ratio),
    )

    def agent_node(state: AgentState):
        iteration_count = state.get("iteration_count", 0) + 1
        if iteration_count > state.get("max_iterations", agent_config.max_iterations):
            return {
                "iteration_count": iteration_count,
                "final_answer": "Stopped: agent reached the maximum tool-iteration count.",
            }
        response = llm_with_tools.invoke(_messages_for_agent(state, workspace, agent_config.max_tool_calls_per_turn))
        update = {"messages": [response], "iteration_count": iteration_count}
        if not getattr(response, "tool_calls", None):
            update["final_answer"] = response.content
        return update

    def context_manager(state: AgentState):
        messages = list(state["messages"])
        if _has_unanswered_tool_calls(messages):
            return {}
        if not should_compact_messages(
            messages,
            max_messages=agent_config.context_message_limit,
            max_chars=compaction_char_limit,
        ):
            return {}

        compaction = compact_messages(
            messages,
            existing_summary=state.get("context_summary"),
            changed_files=state.get("changed_files", []),
            test_result=state.get("test_result"),
            keep_recent=agent_config.context_keep_recent,
        )
        if not compaction.compacted:
            return {}

        summary = _generate_context_summary(summary_llm, compaction.context_summary)
        return {
            "messages": compaction.messages,
            "context_summary": summary,
            "recent_files": compaction.recent_files,
            "compaction_count": state.get("compaction_count", 0) + 1,
        }

    def after_agent(state: AgentState):
        if state.get("final_answer"):
            return END
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "execute"
        return END

    def tool_result_router(state: AgentState):
        errors: list[str] = []
        changed_files = set(state.get("changed_files", []))
        did_write = bool(state.get("did_write"))
        tool_calls_by_id = _recent_tool_calls_by_id(state)

        for message in _recent_tool_messages(state):
            content = str(message.content)
            first_line = _first_line(content)
            if first_line.startswith(("ERROR:", "REJECTED[")):
                errors.append(first_line)

            tool_call = tool_calls_by_id.get(message.tool_call_id)
            if _tool_call_is_write(tool_call) and not first_line.startswith(("ERROR:", "REJECTED[")):
                did_write = True
                path_arg = _path_arg_from_tool_call(tool_call)
                if path_arg:
                    changed_files.add(path_arg)
                changed_files.update(_changed_files_from_git(workspace))

            if tool_call and tool_call.get("name") in {"run_command", "run_shell"} and first_line.startswith("ALLOWED["):
                args = dict(tool_call.get("args") or {})
                update_test_command = str(args.get("command") or "")
                if update_test_command:
                    test_command = update_test_command
                    test_result = content

        update = {
            "changed_files": sorted(changed_files),
            "did_write": did_write,
            "tool_errors": [*state.get("tool_errors", []), *errors],
        }
        if "test_command" in locals():
            update["test_command"] = test_command
            update["test_result"] = test_result
        return update

    def should_continue(state: AgentState):
        if state.get("final_answer"):
            return END
        last_message = state["messages"][-1]
        if last_message.type == "ai" and not getattr(last_message, "tool_calls", None):
            return END
        return "agent"

    def route_after_context(state: AgentState):
        last_message = state["messages"][-1]
        if last_message.type == "ai":
            return after_agent(state)
        return should_continue(state)

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("context_manager", context_manager)
    graph.add_node(
        "execute",
        partial(
            _execute_node,
            workspace=workspace,
            tools_by_name=tools_by_name,
            max_tool_calls_per_turn=agent_config.max_tool_calls_per_turn,
        ),
    )
    graph.add_node("tool_result_router", tool_result_router)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        after_agent,
        {
            "execute": "execute",
            END: END,
        },
    )
    graph.add_edge("execute", "tool_result_router")
    graph.add_edge("tool_result_router", "context_manager")
    graph.add_conditional_edges(
        "context_manager",
        route_after_context,
        {
            "agent": "agent",
            "execute": "execute",
            END: END,
        },
    )

    return graph.compile(checkpointer=InMemorySaver())


def _changed_files_from_git(workspace: Workspace) -> list[str]:
    output = _git_output(workspace, ["diff", "--name-only", "--", "."])
    if output.startswith("ERROR:"):
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


def _messages_for_agent(state: AgentState, workspace: Workspace, max_tool_calls_per_turn: int = 2) -> list:
    messages = _messages_for_llm_with_context_summary(state)
    prepared = _with_single_base_system_prompt(messages)

    runtime_message = SystemMessage(
        content=(
            "Host-provided turn metadata follows. Treat it as current runtime context; "
            "do not call tools just to rediscover these facts. Use tools only when you "
            "need file contents, command output, or to make changes.\n\n"
            f"{build_turn_metadata(workspace)}\n\n"
            "Operational constraints:\n"
            "- You are the agent node: choose whether to inspect, edit, validate, or answer.\n"
            f"- Use at most {max_tool_calls_per_turn} tool calls per turn; inspect in small batches.\n"
            "- Do not read README or list trees just to orient yourself; use host metadata first.\n"
            "- read_file returns a bounded line window by default; request later start_line values as needed.\n"
            "- If you changed files, decide whether a focused validation command is useful before final answer.\n"
            "- Stop and answer when the task is complete; the host enforces a max tool-iteration limit."
        )
    )
    insert_at = 1 if prepared and prepared[0].type == "system" else 0
    return [*prepared[:insert_at], runtime_message, *prepared[insert_at:]]


def _with_single_base_system_prompt(messages: list) -> list:
    non_base_messages = [
        message
        for message in messages
        if not (message.type == "system" and message.content == SYSTEM_PROMPT)
    ]
    return [SystemMessage(content=SYSTEM_PROMPT), *non_base_messages]


def _messages_for_llm_with_context_summary(state: AgentState) -> list:
    messages = list(state["messages"])
    summary = state.get("context_summary")
    if not summary:
        return messages

    summary_message = SystemMessage(
        content=(
            "Historical context summary generated by a separate summarizer. "
            "Use it only as continuity context for the current task:\n"
            f"{summary}"
        )
    )
    insert_at = 1 if messages and messages[0].type == "system" else 0
    return [*messages[:insert_at], summary_message, *messages[insert_at:]]


def _has_unanswered_tool_calls(messages: list) -> bool:
    if not messages:
        return False
    last_message = messages[-1]
    return last_message.type == "ai" and bool(getattr(last_message, "tool_calls", None))


def _first_line(content: str) -> str:
    return content.splitlines()[0] if content.splitlines() else ""


def _tool_call_is_write(tool_call: dict | None) -> bool:
    if not tool_call:
        return False
    tool_name = str(tool_call.get("name") or "")
    if tool_name in WRITE_TOOL_NAMES:
        return True
    if tool_name not in {"run_command", "run_shell"}:
        return False
    args = dict(tool_call.get("args") or {})
    command = " ".join(str(args.get("command") or "").lower().split())
    return any(command.startswith(prefix) for prefix in WRITE_COMMAND_PREFIXES)


def _path_arg_from_tool_call(tool_call: dict | None) -> str | None:
    if not tool_call:
        return None
    args = dict(tool_call.get("args") or {})
    value = args.get("path")
    return str(value) if value else None


def _generate_context_summary(llm, compaction_material: str) -> str:
    try:
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a context compressor for a code agent. Summarize the provided "
                        "history for another agent that will continue the same task. Preserve "
                        "the user goal, files read or changed, important tool observations, "
                        "rejections, validation results, and remaining risks. Do not invent facts."
                    )
                ),
                HumanMessage(
                    content=(
                        "Summarize this conversation context:\n\n"
                        f"{truncate(compaction_material, 24000)}"
                    )
                ),
            ]
        )
        return truncate(str(response.content), 8000)
    except Exception:
        return truncate(compaction_material, 8000)


def _recent_tool_calls_by_id(state: AgentState) -> dict[str, dict]:
    for message in reversed(state["messages"]):
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            return {
                tool_call["id"]: tool_call
                for tool_call in tool_calls
                if "id" in tool_call
            }
    return {}
