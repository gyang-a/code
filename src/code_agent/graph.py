from __future__ import annotations

import subprocess

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

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
    llm = ChatOpenAI(model=agent_config.model, temperature=0)
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
            f"Workspace: {workspace.root}\n"
            f"Detected manifests: {', '.join(manifest_names) or 'none'}\n"
            f"File tree:\n{tree}"
        )
        return {"project_context": context}

    def route_input(state: AgentState):
        user_goal = state["user_goal"].strip()
        return {"input_kind": "slash_command" if user_goal.startswith("/") else "user_task"}

    def command_handler(state: AgentState):
        command = state["user_goal"].strip().split(maxsplit=1)[0]
        if command == "/help":
            answer = "Available slash commands: /help, /clear, /model, /status, /tools, /diff, /doctor, /usage, /mcp, /exit."
        elif command == "/diff":
            answer = _git_output(workspace, ["diff", "--", "."], timeout=20) or "No git diff."
        elif command == "/tools":
            commands = available_commands(workspace)
            answer = "\n".join(f"{name}: {' '.join(spec.argv)}" for name, spec in commands.items()) or "No validation commands detected."
        else:
            answer = f"Slash command {command} is handled by the CLI command layer."
        return {"final_answer": answer}

    def plan_node(state: AgentState):
        messages = list(state["messages"])
        has_system = any(message.type == "system" for message in messages)
        context_messages = []
        if not has_system:
            context_messages.append(SystemMessage(content=SYSTEM_PROMPT))
            context_messages.append(
                HumanMessage(
                    content=(
                        f"{state['project_context']}\n\n"
                        f"Task: {state['user_goal']}\n\n"
                        "Follow this workflow: inspect relevant files, edit only through tools, "
                        "respect permission levels, validate if files changed, then summarize."
                    )
                )
            )
        return {
            "messages": context_messages,
            "plan": state.get("plan") or default_plan(state["user_goal"]),
        }

    def agent_loop(state: AgentState):
        iteration_count = state.get("iteration_count", 0) + 1
        if iteration_count > state.get("max_iterations", agent_config.max_iterations):
            return {
                "iteration_count": iteration_count,
                "final_answer": "Stopped because the agent reached the maximum tool loop count.",
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

        for message in _recent_tool_messages(state):
            content = str(message.content)
            if "APPROVAL_REQUIRED[level_2]" in content:
                approval_reasons.append(content.splitlines()[0])
            elif "REJECTED[level_3]" in content:
                rejection_reasons.append(content.splitlines()[0])
            elif content.startswith("ERROR:"):
                errors.append(content.splitlines()[0])
            if "Patched " in content or "Created " in content or "Wrote " in content:
                changed_files.update(_changed_files_from_git(workspace))

        update = {
            "changed_files": sorted(changed_files),
            "tool_errors": [*state.get("tool_errors", []), *errors],
        }
        if approval_reasons:
            update.update({"needs_approval": True, "approval_reason": "\n".join(approval_reasons)})
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
        reason = state.get("approval_reason") or "Operation requires confirmation."
        return {
            "messages": [
                HumanMessage(
                    content=(
                        "The last tool request required Level 2 approval and was not executed.\n"
                        f"{reason}\n"
                        "Choose a safe alternative, or explain what confirmation is needed."
                    )
                )
            ],
            "needs_approval": False,
            "approval_reason": None,
        }

    def reject_node(state: AgentState):
        reason = state.get("rejected_reason") or "Operation rejected by safety policy."
        return {
            "messages": [
                HumanMessage(
                    content=(
                        "The last tool request was rejected as Level 3 risk.\n"
                        f"{reason}\n"
                        "Do not try that operation again. Choose a safe alternative."
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
            return {"test_result": "Skipped validation because no workspace changes were detected."}

        commands = available_commands(workspace)
        preferred = ["python_compile", "pytest", "npm_build", "npm_test", "pnpm_build", "pnpm_test"]
        selected = next((name for name in preferred if name in commands), None)
        if selected is None:
            return {"test_result": "Skipped validation because no safe validation command was detected."}

        spec = commands[selected]
        try:
            result = run_argv(spec.argv, workspace.root, timeout=120)
            output = truncate(result.stdout + result.stderr)
            test_result = f"$ {' '.join(spec.argv)}\nexit_code={result.returncode}\n{output}"
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            test_result = f"Validation failed to run: {exc}"

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
                "Produce the final response now. Include files changed, what changed, "
                "validation result, and remaining risks.\n\n"
                f"Changed files: {', '.join(state.get('changed_files', [])) or 'none'}\n"
                f"Validation: {state.get('test_result') or 'not run'}\n"
                f"Diff summary: {state.get('diff_summary') or 'no diff'}"
            )
        )
        response = llm.invoke([*state["messages"], final_prompt])
        return {"messages": [response], "final_answer": response.content}

    graph = StateGraph(AgentState)
    graph.add_node("load_project_context", load_project_context)
    graph.add_node("route_input", route_input)
    graph.add_node("command_handler", command_handler)
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

    graph.add_edge(START, "load_project_context")
    graph.add_edge("load_project_context", "route_input")
    graph.add_conditional_edges(
        "route_input",
        lambda state: state.get("input_kind") or "user_task",
        {
            "slash_command": "command_handler",
            "user_task": "plan_node",
        },
    )
    graph.add_edge("command_handler", END)
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
        return "No git diff."
    file_summary = ", ".join(changed_files) if changed_files else "No tracked file diff."
    line_count = len(diff.splitlines()) if diff else 0
    return f"Changed files: {file_summary}. Diff lines: {line_count}."
