from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    ModelRequest,
    ModelResponse,
    SummarizationMiddleware,
    TodoListMiddleware,
    ToolCallLimitMiddleware,
)
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import InMemorySaver

from code_agent.config import AgentConfig
from code_agent.middleware import (
    ModelRetryMiddleware,
    PerModelToolCallLimitMiddleware,
    ToolErrorMiddleware,
)
from code_agent.prompts import SYSTEM_PROMPT
from code_agent.services.metadata import build_turn_metadata
from code_agent.services.permissions import classify_tool_call
from code_agent.services.skills import SkillStore, format_skill_index
from code_agent.services.workspace import Workspace
from code_agent.state import AgentState
from code_agent.ui.console import console
from code_agent.tools import (
    build_create_file_tool,
    build_delete_file_tool,
    build_find_files_tool,
    build_git_diff_tool,
    build_git_status_tool,
    build_list_files_tool,
    build_patch_file_tool,
    build_read_file_tool,
    build_search_text_tool,
    build_shell_command_tool,
    build_skill_view_tool,
    build_skills_list_tool,
    build_write_file_tool,
)


def build_tools(
    workspace: Workspace,
    *,
    read_max_lines: int | None = None,
    tool_output_limit: int | None = None,
    shell_timeout_ms: int = 10_000,
    shell_max_timeout_ms: int = 120_000,
    shell_output_limit: int = 12_000,
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
        build_skills_list_tool(workspace),
        build_skill_view_tool(workspace),
        build_git_status_tool(workspace),
        build_git_diff_tool(workspace),
        build_shell_command_tool(
            workspace,
            default_timeout_ms=shell_timeout_ms,
            max_timeout_ms=shell_max_timeout_ms,
            output_limit=shell_output_limit,
        ),
    ]


def build_graph(workspace_path: str, config: AgentConfig | None = None, checkpointer=None):
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
        shell_timeout_ms=agent_config.shell_timeout_ms,
        shell_max_timeout_ms=agent_config.shell_max_timeout_ms,
        shell_output_limit=agent_config.shell_output_limit,
    )
    llm_kwargs = {
        "temperature": 0,
        "timeout": agent_config.model_request_timeout_seconds,
    }
    if agent_config.api_key:
        llm_kwargs["api_key"] = agent_config.api_key
    llm = init_chat_model(agent_config.model, model_provider="deepseek", **llm_kwargs)
    fallback_llm = None
    if agent_config.fallback_model and agent_config.fallback_model != agent_config.model:
        fallback_llm = init_chat_model(
            agent_config.fallback_model,
            model_provider="deepseek",
            **llm_kwargs,
        )

    return create_agent(
        llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        middleware=[
            RuntimeMetadataMiddleware(
                workspace=workspace,
                max_tool_calls_per_turn=agent_config.max_tool_calls_per_turn,
            ),
            ModelRetryMiddleware(
                max_retries=agent_config.model_max_retries,
                total_timeout_seconds=agent_config.model_timeout_seconds,
                base_delay_seconds=agent_config.model_retry_base_delay_seconds,
                max_delay_seconds=agent_config.model_retry_max_delay_seconds,
                fallback_model=fallback_llm,
                on_retry=_print_model_retry,
                on_fallback=_print_model_fallback,
            ),
            PerModelToolCallLimitMiddleware(agent_config.max_tool_calls_per_turn),
            TodoListMiddleware(),
            HumanInTheLoopMiddleware(
                interrupt_on=_approval_interrupt_config(workspace, tools),
                description_prefix="Tool execution requires approval",
            ),
            ToolErrorMiddleware(),
            SummarizationMiddleware(
                llm,
                trigger=[
                    ("messages", agent_config.context_message_limit),
                    ("tokens", agent_config.context_token_limit),
                ],
                keep=("messages", agent_config.context_keep_recent),
            ),
            ToolCallLimitMiddleware(
                run_limit=agent_config.max_total_tool_calls_per_run,
                exit_behavior="continue",
            ),
            ModelCallLimitMiddleware(
                run_limit=agent_config.max_iterations,
                exit_behavior="end",
            ),
        ],
        state_schema=AgentState,
        checkpointer=checkpointer or InMemorySaver(),
    )


def _print_model_retry(
    next_attempt: int,
    total_attempts: int,
    delay_seconds: float,
    exc: Exception,
) -> None:
    console.print(
        f"[dim yellow]模型请求暂时失败，{delay_seconds:.1f} 秒后重试 "
        f"{next_attempt}/{total_attempts}（{type(exc).__name__}）…[/dim yellow]"
    )


def _print_model_fallback(exc: Exception) -> None:
    console.print(
        f"[dim yellow]主模型连续失败，正在切换备用模型（{type(exc).__name__}）…[/dim yellow]"
    )


class RuntimeMetadataMiddleware(AgentMiddleware):
    def __init__(
        self,
        *,
        workspace: Workspace,
        max_tool_calls_per_turn: int,
    ) -> None:
        super().__init__()
        self.workspace = workspace
        self.max_tool_calls_per_turn = max_tool_calls_per_turn

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler,
    ) -> ModelResponse | Any:
        base = request.system_message.text if request.system_message else SYSTEM_PROMPT
        state = request.state if isinstance(request.state, dict) else {}
        memory_query = str(state.get("user_goal") or "")
        runtime_metadata = (
            "Runtime metadata:\n"
            "The host provides the current project folder snapshot below. Treat it as current context, "
            "not as conversation history.\n\n"
            f"{build_turn_metadata(self.workspace, memory_query=memory_query)}\n\n"
            "Available global skills:\n"
            f"{format_skill_index(SkillStore().list_skills())}\n\n"
            "Skill loading rules:\n"
            "- Compare the current user request with the available global skills before taking action.\n"
            "- If a listed skill is relevant, call skill_view for that skill before inspecting or editing project files.\n"
            "- If no listed skill is relevant, continue without calling skill_view.\n\n"
            "Command execution: use shell_command. It runs PowerShell under the Windows read-only "
            "restricted-token sandbox by default. After a real read-only file denial, use workspace-write "
            "for local writes, but use direct danger-full-access for package managers such as npm install or uv add "
            "that need external runtimes/caches. Retry the exact command with a justification. Never request "
            "workspace-write speculatively or work around a rejected escalation. Each call is a fresh "
            "PowerShell process; controlled modes use ConstrainedLanguage and workdir replaces cd. Any workspace-write "
            "denial, or a read-only process-pipe denial such as spawn EPERM, permits one exact retry with sandbox_permissions='danger-full-access' "
            "and separate approval. Never request it speculatively or change the command spelling.\n"
            f"Tool call budget: the host executes at most {self.max_tool_calls_per_turn} tool calls from each model response. "
            "If more work remains, continue it in a later response."
        )
        return handler(
            request.override(
                system_message=SystemMessage(content=f"{base}\n\n{runtime_metadata}")
            )
        )


def _approval_interrupt_config(workspace: Workspace, tools: list) -> dict[str, dict[str, Any]]:
    return {
        tool.name: {
            "allowed_decisions": ["approve", "edit", "reject", "respond"],
            "when": _approval_required(workspace, tool),
            "description": _approval_description(workspace),
        }
        for tool in tools
    }


def _approval_required(workspace: Workspace, tool=None):
    def when(request) -> bool:
        tool_call = request.tool_call
        args = dict(tool_call.get("args") or {})
        decision = classify_tool_call(
            workspace,
            str(tool_call.get("name") or ""),
            args,
        )
        eligibility_check = getattr(tool, "_approval_eligible", None)
        if decision.requires_approval and callable(eligibility_check):
            return bool(eligibility_check(args))
        return bool(decision.requires_approval)

    return when


def _approval_description(workspace: Workspace):
    def describe(tool_call, state, runtime) -> str:
        decision = classify_tool_call(
            workspace,
            str(tool_call.get("name") or ""),
            dict(tool_call.get("args") or {}),
        )
        return decision.reason

    return describe
