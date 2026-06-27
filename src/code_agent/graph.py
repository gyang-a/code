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
    ToolCallLimitMiddleware,
)
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import InMemorySaver

from code_agent.config import AgentConfig
from code_agent.prompts import SYSTEM_PROMPT
from code_agent.services.metadata import build_turn_metadata
from code_agent.services.permissions import classify_tool_call
from code_agent.services.sandbox import SandboxPolicy, describe_sandbox_policy
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


def build_graph(workspace_path: str, config: AgentConfig | None = None):
    agent_config = config or AgentConfig()
    workspace = Workspace(
        workspace_path,
        read_limit=agent_config.file_read_limit,
        read_max_lines=agent_config.file_read_max_lines,
        exclude_globs=agent_config.exclude_globs,
    )
    sandbox_policy = SandboxPolicy(
        backend=agent_config.shell_sandbox_backend,
        docker_image=agent_config.docker_image,
        allow_network=agent_config.docker_allow_network,
    )
    tools = build_tools(
        workspace,
        read_max_lines=agent_config.file_read_max_lines,
        tool_output_limit=agent_config.tool_output_limit,
        sandbox_policy=sandbox_policy,
    )
    llm_kwargs = {"temperature": 0}
    if agent_config.api_key:
        llm_kwargs["api_key"] = agent_config.api_key
    llm = init_chat_model(agent_config.model, model_provider="deepseek", **llm_kwargs)

    return create_agent(
        llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        middleware=[
            RuntimeMetadataMiddleware(
                workspace=workspace,
                shell_sandbox=describe_sandbox_policy(sandbox_policy),
                max_tool_calls_per_turn=agent_config.max_tool_calls_per_turn,
            ),
            HumanInTheLoopMiddleware(
                interrupt_on=_approval_interrupt_config(workspace, tools),
                description_prefix="Tool execution requires approval",
            ),
            SummarizationMiddleware(
                llm,
                trigger=[
                    ("messages", agent_config.context_message_limit),
                    ("tokens", agent_config.context_token_limit),
                ],
                keep=("messages", agent_config.context_keep_recent),
            ),
            ToolCallLimitMiddleware(
                run_limit=agent_config.max_iterations * agent_config.max_tool_calls_per_turn,
                exit_behavior="continue",
            ),
            ModelCallLimitMiddleware(
                run_limit=agent_config.max_iterations,
                exit_behavior="end",
            ),
        ],
        state_schema=AgentState,
        checkpointer=InMemorySaver(),
    )


class RuntimeMetadataMiddleware(AgentMiddleware):
    def __init__(
        self,
        *,
        workspace: Workspace,
        shell_sandbox: str,
        max_tool_calls_per_turn: int,
    ) -> None:
        super().__init__()
        self.workspace = workspace
        self.shell_sandbox = shell_sandbox
        self.max_tool_calls_per_turn = max_tool_calls_per_turn

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler,
    ) -> ModelResponse | Any:
        base = request.system_message.text if request.system_message else SYSTEM_PROMPT
        runtime_metadata = (
            "Runtime metadata:\n"
            "The host provides the current workspace snapshot below. Treat it as current context, "
            "not as conversation history.\n\n"
            f"{build_turn_metadata(self.workspace)}\n\n"
            f"Shell sandbox: {self.shell_sandbox}.\n"
            f"Tool call budget hint: prefer at most {self.max_tool_calls_per_turn} tool calls per model turn."
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
            "when": _approval_required(workspace),
            "description": _approval_description(workspace),
        }
        for tool in tools
    }


def _approval_required(workspace: Workspace):
    def when(request) -> bool:
        tool_call = request.tool_call
        decision = classify_tool_call(
            workspace,
            str(tool_call.get("name") or ""),
            dict(tool_call.get("args") or {}),
        )
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
