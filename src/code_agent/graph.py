from __future__ import annotations

import subprocess

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

from code_agent.config import AgentConfig
from code_agent.prompts import SYSTEM_PROMPT
from code_agent.services.planner import default_plan
from code_agent.services.summarizer import truncate
from code_agent.services.workspace import Workspace
from code_agent.state import AgentState
from code_agent.tools import (
    build_create_file_tool,
    build_delete_file_tool,
    build_find_files_tool,
    build_get_file_tree_tool,
    build_git_diff_tool,
    build_git_status_tool,
    build_list_files_tool,
    build_patch_file_tool,
    build_read_file_tool,
    build_run_command_tool,
    build_search_text_tool,
    build_write_file_tool,
)
from code_agent.tools.safety import available_commands, run_argv


def build_tools(workspace: Workspace):
    return [
        build_list_files_tool(workspace),
        build_get_file_tree_tool(workspace),
        build_read_file_tool(workspace),
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


def _git_output(workspace: Workspace, args: list[str], timeout: int = 10) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=workspace.root,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return f"ERROR: {exc}"
    return truncate(result.stdout + result.stderr)


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


def _has_changes(workspace: Workspace) -> bool:
    status = _git_output(workspace, ["status", "--short"])
    return bool(status.strip()) and not status.startswith("ERROR:")


def build_graph(workspace_path: str, config: AgentConfig | None = None):
    agent_config = config or AgentConfig()
    workspace = Workspace(
        workspace_path,
        read_limit=agent_config.file_read_limit,
        exclude_globs=agent_config.exclude_globs,
    )
    tools = build_tools(workspace)
    tools_by_name = {tool.name: tool for tool in tools}
    llm_kwargs = {"temperature": 0}
    if agent_config.api_key:
        llm_kwargs["api_key"] = agent_config.api_key
    llm = init_chat_model(
        agent_config.model,
        model_provider="deepseek",
        **llm_kwargs,
    )
    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)

    def load_project_context(state: AgentState):
        tree = "\n".join(workspace.iter_tree(".", max_entries=80))
        manifest_names = [
            name
            for name in ("README.md", "pyproject.toml", "package.json", "uv.lock", "pnpm-lock.yaml")
            if (workspace.root / name).exists()
        ]
        context = (
            f"工作区: {workspace.root}\n"
            f"检测到的清单文件: {', '.join(manifest_names) or '无'}\n"
            f"文件树:\n{tree}"
        )
        return {"project_context": context}

    def route_input(state: AgentState):
        user_goal = state["user_goal"].strip()
        return {"input_kind": "slash_command" if user_goal.startswith("/") else "ai_routed"}

    def command_handler(state: AgentState):
        command = state["user_goal"].strip().split(maxsplit=1)[0]
        if command == "/help":
            answer = "可用 slash commands: /help, /clear, /model, /status, /tools, /diff, /doctor, /usage, /mcp, /exit。"
        elif command == "/diff":
            answer = _git_output(workspace, ["diff", "--", "."], timeout=20) or "当前没有 git diff。"
        elif command == "/tools":
            commands = available_commands(workspace)
            answer = "\n".join(f"{name}: {' '.join(spec.argv)}" for name, spec in commands.items()) or "未检测到验证命令。"
        else:
            answer = f"Slash command {command} 已由 CLI 控制层处理。"
        return {"final_answer": answer}

    def direct_response(state: AgentState):
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "你是一个 CLI 代码智能体的对话入口。"
                        "对于不属于代码任务的消息，请直接、简短地用中文回答。"
                        f"当前配置的模型名是 {agent_config.model}。"
                        "不要检查文件，不要声称已经检查文件，也不要进行代码修改。"
                    )
                ),
                HumanMessage(content=state["user_goal"]),
            ]
        )
        return {"messages": [response], "final_answer": response.content}

    def route_after_input(state: AgentState) -> str:
        if state["user_goal"].strip().startswith("/"):
            return "slash_command"
        return _classify_user_intent(llm, state["user_goal"])

    def route_after_input_for_graph(state: AgentState) -> str:
        intent = route_after_input(state)
        if intent in {"casual_chat", "general_question"}:
            return "direct_response"
        if intent == "slash_command":
            return "command_handler"
        return "load_project_context"

    def plan_node(state: AgentState):
        messages = list(state["messages"])
        has_system = any(message.type == "system" for message in messages)
        plan = state.get("plan") or _generate_task_plan(
            llm,
            project_context=state["project_context"] or "",
            user_goal=state["user_goal"],
        )
        context_messages = []
        if not has_system:
            context_messages.append(SystemMessage(content=SYSTEM_PROMPT))
            context_messages.append(
                HumanMessage(
                    content=(
                        f"{state['project_context']}\n\n"
                        f"任务: {state['user_goal']}\n\n"
                        "AI 生成的执行计划：\n"
                        + "\n".join(f"{index}. {item}" for index, item in enumerate(plan, start=1))
                        + "\n\n请按计划推进。执行过程中如果发现计划不准确，可以根据工具结果调整。"
                    )
                )
            )
        return {
            "messages": context_messages,
            "plan": plan,
        }

    def agent_loop(state: AgentState):
        iteration_count = state.get("iteration_count", 0) + 1
        if iteration_count > state.get("max_iterations", agent_config.max_iterations):
            return {
                "iteration_count": iteration_count,
                "final_answer": "已停止：Agent 达到最大工具循环次数。",
            }
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response], "iteration_count": iteration_count}

    def after_agent(state: AgentState):
        if state.get("final_answer"):
            return "final_summary"
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "execute"
        return "validation_node" if _has_changes(workspace) else "review_diff_node"

    def tool_result_router(state: AgentState):
        approval_reasons: list[str] = []
        rejection_reasons: list[str] = []
        errors: list[str] = []
        changed_files = set(state.get("changed_files", []))
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
                    }
            elif first_line.startswith("REJECTED[level_3]"):
                rejection_reasons.append(first_line)
            elif first_line.startswith("ERROR:"):
                errors.append(first_line)
            if "Patched " in content or "Created " in content or "Wrote " in content:
                changed_files.update(_changed_files_from_git(workspace))

        update = {
            "changed_files": sorted(changed_files),
            "tool_errors": [*state.get("tool_errors", []), *errors],
        }
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
        return "observe"

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
                    HumanMessage(
                        content=(
                            "用户拒绝了 Level 2 操作。\n"
                            f"{reason}\n"
                            "请选择更安全的替代方案，或明确说明已停止。"
                        )
                    )
                ],
                "needs_approval": False,
                "approval_reason": None,
                "pending_approval": None,
            }

        execution_result = _execute_approved_tool(tools_by_name, pending)
        return {
            "messages": [
                HumanMessage(
                    content=(
                        "用户批准了 Level 2 操作。\n"
                        f"{reason}\n"
                        f"执行结果:\n{execution_result}"
                    )
                )
            ],
            "needs_approval": False,
            "approval_reason": None,
            "pending_approval": None,
        }

    def reject_node(state: AgentState):
        reason = state.get("rejected_reason") or "操作已被安全策略拒绝。"
        return {
            "messages": [
                HumanMessage(
                        content=(
                        "上一个工具请求被判定为 Level 3 风险并已拒绝。\n"
                        f"{reason}\n"
                        "不要再次尝试该操作。请选择更安全的替代方案。"
                    )
                )
            ],
            "rejected_reason": None,
        }

    def observe_node(state: AgentState):
        return {}

    def should_continue(state: AgentState):
        if state.get("final_answer"):
            return "final_summary"
        last_message = state["messages"][-1]
        if last_message.type == "ai" and not getattr(last_message, "tool_calls", None):
            return "validation_node" if _has_changes(workspace) else "review_diff_node"
        return "agent_loop"

    def validation_node(state: AgentState):
        if not _has_changes(workspace):
            return {"test_result": "已跳过验证：未检测到工作区变更。"}

        commands = available_commands(workspace)
        preferred = ["python_compile", "pytest", "npm_build", "npm_test", "pnpm_build", "pnpm_test"]
        selected = next((name for name in preferred if name in commands), None)
        if selected is None:
            return {"test_result": "已跳过验证：未检测到安全的验证命令。"}

        spec = commands[selected]
        try:
            result = run_argv(spec.argv, workspace.root, timeout=120)
            output = truncate(result.stdout + result.stderr)
            test_result = f"$ {' '.join(spec.argv)}\nexit_code={result.returncode}\n{output}"
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            test_result = f"验证命令运行失败: {exc}"

        return {"test_command": selected, "test_result": test_result}

    def review_diff_node(state: AgentState):
        diff = _git_output(workspace, ["diff", "--", "."], timeout=20)
        changed_files = _changed_files_from_git(workspace)
        return {
            "changed_files": changed_files,
            "last_diff": diff or None,
            "diff_summary": _summarize_diff(changed_files, diff),
        }

    def final_summary(state: AgentState):
        if state.get("final_answer"):
            return {}

        final_prompt = HumanMessage(
            content=(
                "现在生成最终中文回复。请包含：修改了哪些文件、改了什么、验证结果、剩余风险。\n\n"
                f"变更文件: {', '.join(state.get('changed_files', [])) or '无'}\n"
                f"验证结果: {state.get('test_result') or '未运行'}\n"
                f"Diff 摘要: {state.get('diff_summary') or '无 diff'}"
            )
        )
        response = llm.invoke([*state["messages"], final_prompt])
        return {"messages": [response], "final_answer": response.content}

    graph = StateGraph(AgentState)
    graph.add_node("load_project_context", load_project_context)
    graph.add_node("route_input", route_input)
    graph.add_node("command_handler", command_handler)
    graph.add_node("direct_response", direct_response)
    graph.add_node("plan_node", plan_node)
    graph.add_node("agent_loop", agent_loop)
    graph.add_node("execute", tool_node)
    graph.add_node("tool_result_router", tool_result_router)
    graph.add_node("approval", approval_node)
    graph.add_node("reject", reject_node)
    graph.add_node("observe", observe_node)
    graph.add_node("validation_node", validation_node)
    graph.add_node("review_diff_node", review_diff_node)
    graph.add_node("final_summary", final_summary)

    graph.add_edge(START, "route_input")
    graph.add_conditional_edges(
        "route_input",
        route_after_input_for_graph,
        {
            "command_handler": "command_handler",
            "direct_response": "direct_response",
            "load_project_context": "load_project_context",
        },
    )
    graph.add_edge("load_project_context", "plan_node")
    graph.add_edge("command_handler", END)
    graph.add_edge("direct_response", END)
    graph.add_edge("plan_node", "agent_loop")
    graph.add_conditional_edges(
        "agent_loop",
        after_agent,
        {
            "execute": "execute",
            "validation_node": "validation_node",
            "review_diff_node": "review_diff_node",
            "final_summary": "final_summary",
        },
    )
    graph.add_edge("execute", "tool_result_router")
    graph.add_conditional_edges(
        "tool_result_router",
        route_tool_result,
        {
            "approval": "approval",
            "reject": "reject",
            "observe": "observe",
        },
    )
    graph.add_edge("approval", "observe")
    graph.add_edge("reject", "observe")
    graph.add_conditional_edges(
        "observe",
        should_continue,
        {
            "agent_loop": "agent_loop",
            "validation_node": "validation_node",
            "review_diff_node": "review_diff_node",
            "final_summary": "final_summary",
        },
    )
    graph.add_edge("validation_node", "review_diff_node")
    graph.add_edge("review_diff_node", "final_summary")
    graph.add_edge("final_summary", END)

    return graph.compile(checkpointer=InMemorySaver())


def _changed_files_from_git(workspace: Workspace) -> list[str]:
    output = _git_output(workspace, ["diff", "--name-only", "--", "."])
    if output.startswith("ERROR:"):
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


def _summarize_diff(changed_files: list[str], diff: str) -> str:
    if not changed_files and not diff:
        return "当前没有 git diff。"
    file_summary = ", ".join(changed_files) if changed_files else "没有已跟踪文件的 diff。"
    line_count = len(diff.splitlines()) if diff else 0
    return f"变更文件: {file_summary}。Diff 行数: {line_count}。"


def _first_line(content: str) -> str:
    return content.splitlines()[0] if content.splitlines() else ""


def _classify_user_intent(llm, user_goal: str) -> str:
    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "请为 CLI 代码智能体判断用户消息意图。"
                    "只返回一个标签：casual_chat、general_question 或 code_task。\n"
                    "- casual_chat：问候、感谢、闲聊，或不需要检查工作区的普通对话。\n"
                    "- general_question：关于助手、模型、使用方式、能力或概念的问题，不需要读取工作区。\n"
                    "- code_task：要求检查、解释、修改、调试、测试、运行、搜索或推理工作区文件的请求。\n"
                    "只能返回标签本身。"
                )
            ),
            HumanMessage(content=user_goal),
        ]
    )
    return _normalize_intent_label(str(response.content))


def _generate_task_plan(llm, *, project_context: str, user_goal: str) -> list[str]:
    try:
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "你是 CLI 代码智能体的规划节点。"
                        "请根据用户目标和项目上下文生成 3 到 6 步中文执行计划。"
                        "计划必须针对当前任务，不要输出固定模板。"
                        "每一步只写一句话，不要展开解释。"
                        "不要使用工具，不要假装已经读取文件。"
                    )
                ),
                HumanMessage(
                    content=(
                        f"用户目标：{user_goal}\n\n"
                        f"项目上下文：\n{project_context}\n\n"
                        "请只输出步骤列表，每行一步。"
                    )
                ),
            ]
        )
        plan = _parse_plan_lines(str(response.content))
        if plan:
            return plan
    except Exception:
        pass
    return default_plan(user_goal)


def _parse_plan_lines(raw_plan: str) -> list[str]:
    steps: list[str] = []
    for raw_line in raw_plan.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = line.lstrip("-* ")
        line = line.lstrip("0123456789.、)） ")
        line = line.strip()
        if line:
            steps.append(line)
        if len(steps) >= 6:
            break
    return steps


def _normalize_intent_label(raw_label: str) -> str:
    label = raw_label.strip().lower().split()[0] if raw_label.strip() else "code_task"
    label = label.strip("`'\".,:;")
    if label in {"casual_chat", "general_question", "code_task"}:
        return label
    return "code_task"


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
