from __future__ import annotations

import subprocess
from functools import partial

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from code_agent.config import AgentConfig
from code_agent.prompts import SYSTEM_PROMPT
from code_agent.services.context import compact_messages, should_compact_messages
from code_agent.services.metadata import build_turn_metadata
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


def build_tools(
    workspace: Workspace,
    *,
    read_max_lines: int | None = None,
    tool_output_limit: int | None = None,
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
        build_run_command_tool(workspace),
        build_git_status_tool(workspace),
        build_git_diff_tool(workspace),
    ]


def _execute_node(
    state: AgentState,
    *,
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

    results = []
    for tool_call in tool_calls:
        tool_name = tool_call.get("name")
        tool_call_id = tool_call.get("id")
        args = dict(tool_call.get("args") or {})
        if not tool_call_id:
            continue
        if tool_name not in tools_by_name:
            results.append(ToolMessage(content=f"ERROR: 未知工具: {tool_name}", tool_call_id=tool_call_id))
            continue
        try:
            content = str(tools_by_name[tool_name].invoke(args))
        except Exception as exc:
            content = f"ERROR: 工具执行失败: {exc}"
        results.append(ToolMessage(content=content, tool_call_id=tool_call_id))
    return {"messages": results}


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
                "final_answer": "已停止：Agent 达到最大工具循环次数。",
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
        approval_reasons: list[str] = []
        rejection_reasons: list[str] = []
        errors: list[str] = []
        changed_files = set(state.get("changed_files", []))
        did_write = bool(state.get("did_write"))
        pending_approval = None
        tool_calls_by_id = _recent_tool_calls_by_id(state)

        for message in _recent_tool_messages(state):
            content = str(message.content)
            first_line = _first_line(content)
            if first_line.startswith("APPROVAL_REQUIRED[level_2]"):
                approval_reasons.append(first_line)
                tool_call = tool_calls_by_id.get(message.tool_call_id)
                if tool_call and pending_approval is None:
                    pending_approval = {
                        "tool": tool_call["name"],
                        "args": dict(tool_call.get("args") or {}),
                        "reason": first_line,
                        "tool_call_id": message.tool_call_id,
                        "tool_message_id": message.id,
                    }
            elif first_line.startswith("REJECTED[level_3]"):
                rejection_reasons.append(first_line)
            elif first_line.startswith("ERROR:"):
                errors.append(first_line)
            tool_call = tool_calls_by_id.get(message.tool_call_id)
            if _tool_call_is_write(tool_call):
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
        if approval_reasons:
            update.update(
                {
                    "needs_approval": True,
                    "approval_reason": "\n".join(approval_reasons),
                    "pending_approval": pending_approval,
                }
            )
        if rejection_reasons:
            update.update({"rejected_reason": "\n".join(rejection_reasons)})
        return update

    def route_tool_result(state: AgentState):
        if state.get("approval_reason"):
            return "approval"
        if state.get("rejected_reason"):
            return "reject"
        return "context_manager"

    def approval_node(state: AgentState):
        reason = state.get("approval_reason") or "该操作需要确认。"
        pending = state.get("pending_approval")
        decision = interrupt(
            {
                "risk": "level_2",
                "reason": reason,
                "action": pending,
            }
        )
        approved = bool(decision.get("approved")) if isinstance(decision, dict) else bool(decision)
        if not approved:
            return {
                "messages": [
                    _tool_message_for_pending(
                        pending,
                        content=(
                            "DENIED[level_2]: User rejected the requested tool action.\n"
                            f"{reason}\n"
                            "Choose a safer alternative or explain that the action stopped."
                        ),
                    )
                ],
                "needs_approval": False,
                "approval_reason": None,
                "pending_approval": None,
            }

        execution_result = _execute_approved_tool(tools_by_name, pending)
        did_write = bool(state.get("did_write") or _pending_action_is_write(pending))
        changed_files = set(state.get("changed_files", []))
        if did_write:
            path_arg = _path_arg_from_pending(pending)
            if path_arg:
                changed_files.add(path_arg)
            changed_files.update(_changed_files_from_git(workspace))
        return {
            "messages": [
                _tool_message_for_pending(
                    pending,
                    content=(
                        "APPROVED[level_2]: User approved the requested tool action.\n"
                        f"{reason}\n"
                        f"Execution result:\n{execution_result}"
                    ),
                )
            ],
            "needs_approval": False,
            "approval_reason": None,
            "pending_approval": None,
            "did_write": did_write,
            "changed_files": sorted(changed_files),
        }

    def reject_node(state: AgentState):
        return {
            "rejected_reason": None,
        }

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
            tools_by_name=tools_by_name,
            max_tool_calls_per_turn=agent_config.max_tool_calls_per_turn,
        ),
    )
    graph.add_node("tool_result_router", tool_result_router)
    graph.add_node("approval", approval_node)
    graph.add_node("reject", reject_node)

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
    graph.add_conditional_edges(
        "tool_result_router",
        route_tool_result,
        {
            "approval": "approval",
            "reject": "reject",
            "context_manager": "context_manager",
        },
    )
    graph.add_edge("approval", "context_manager")
    graph.add_edge("reject", "context_manager")
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
            "历史上下文摘要（由独立 summary LLM 压缩生成，仅作为继续任务的参考）：\n"
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


def _path_arg_from_pending(pending: dict | None) -> str | None:
    if not pending:
        return None
    args = dict(pending.get("args") or {})
    value = args.get("path")
    return str(value) if value else None


def _pending_action_is_write(pending: dict | None) -> bool:
    if not pending:
        return False
    return _tool_call_is_write(
        {
            "name": pending.get("tool"),
            "args": dict(pending.get("args") or {}),
        }
    )


def _tool_message_for_pending(pending: dict | None, *, content: str) -> ToolMessage:
    pending = pending or {}
    tool_call_id = str(pending.get("tool_call_id") or "unknown_tool_call")
    message_id = pending.get("tool_message_id")
    kwargs = {"content": content, "tool_call_id": tool_call_id}
    if message_id:
        kwargs["id"] = message_id
    return ToolMessage(**kwargs)


def _generate_context_summary(llm, compaction_material: str) -> str:
    try:
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "你是代码智能体的上下文压缩器，运行在独立会话中。\n"
                        "请把输入材料压缩成中文工作摘要，供另一个主 Agent 继续任务。\n"
                        "必须保留：用户目标、已经读过/改过的文件、关键工具结果、审批/拒绝信息、验证结果、剩余风险。\n"
                        "不要输出寒暄，不要新增事实，不要假装执行了工具。"
                    )
                ),
                HumanMessage(
                    content=(
                        "请压缩以下历史上下文材料：\n\n"
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


def _execute_approved_tool(tools_by_name: dict, pending: dict | None) -> str:
    if not pending:
        return "ERROR: 未找到待审批的 Level 2 操作。"
    tool_name = pending.get("tool")
    args = dict(pending.get("args") or {})
    if tool_name not in tools_by_name:
        return f"ERROR: 未知的待审批工具: {tool_name}"
    args["approval_token"] = "approved"
    try:
        return str(tools_by_name[tool_name].invoke(args))
    except Exception as exc:
        return f"ERROR: 已审批工具执行失败: {exc}"
