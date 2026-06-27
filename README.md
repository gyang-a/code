# Code Agent

一个基于 LangGraph 工具调用的、带工作区沙箱的 CLI 代码智能体原型。

这个项目按“可以写进简历的 MVP”来设计，重点不是堆功能，而是把代码智能体最容易翻车的边界做好：

- 固定工作区沙箱
- 有上限的文件读取和搜索结果
- 基于精确匹配的安全 `patch_file`
- 受限 `run_shell`，由 Agent 自己选择搜索、查看、测试、构建或验证命令
- 可见的 git status / diff
- 类似 Claude Code 的 slash commands
- Level 0-3 风险分级与人工审批
- 流式展示节点进度、工具调用和工具结果

## 安装

```bash
pip install -e ".[dev]"
```

在运行自然语言任务前，请先配置 `DEEPSEEK_API_KEY`。

## 运行

在项目根目录创建本地 `.env` 文件：

```bash
DEEPSEEK_API_KEY=your-deepseek-api-key-here
# CODE_AGENT_MODEL=deepseek-v4-flash
```

启动 CLI：

```bash
code-agent .
```

或者：

```bash
python -m code_agent.main .
```

自然语言任务会以流式方式展示 LangGraph 工作流进度，例如：

```text
> 已加载项目上下文
> 已生成任务计划
  工具调用: read_file(path='...')
  工具结果: ...
> 已请求 Agent 决定验证方式
> 已准备最终总结
```

## Slash Commands

- `/help` 显示命令列表
- `/clear` 开启新的会话线程
- `/model [name]` 查看或切换模型
- `/status` 查看当前会话状态
- `/tools` 查看工具权限和 shell 风险分级规则
- `/diff` 查看当前 git diff
- `/doctor` 检查本地依赖和环境变量
- `/usage` 查看本地交互计数
- `/mcp` 预留的 MCP 集成入口
- `/exit` 退出 CLI

## 架构

```text
START
  -> route_input
      -> slash_command -> command_handler
      -> agent
          -> context_manager
          -> execute
          -> tool_result_router
              -> observe
              -> approval -> observe
              -> reject -> observe
```

`agent` 是唯一负责思考和决策的 LLM 节点：是否读取文件、是否制定计划、是否验证、何时停止，都由 agent 根据任务和工具结果自己决定。Graph 只提供运行外壳：每轮注入工作区、分支、git status、顶层目录、项目标记文件和 memory 等元信息，执行工具，处理审批，压缩上下文，并用最大迭代次数防止无限循环。

模型永远不会直接访问文件系统。它只能调用工具层暴露的函数，而工具内部负责路径、读取窗口、输出长度、命令和敏感文件策略。`read_file` 默认只返回有限行数，Agent 需要用 `start_line` 按需继续读取。
写入后的验证不由 Graph 硬编码触发。Agent 自己按项目文件和报错上下文判断是否需要运行验证命令，再交给 permission rules 分级，最后由受限 `run_shell` 在 sandbox 中执行。

Slash commands 由 CLI 控制面优先处理，所以 `/help`、`/diff`、`/doctor` 这类命令不需要调用 LLM。

## 分层边界

第一层是 permission rules：在工具执行前判断某个 tool call 或某条 shell 命令属于 Level 0-3。这里不维护项目命令表，不替 Agent 选择 `pytest`、`npm test` 或 `uv run pytest`；它只审查 Agent 自己写出的命令是否允许、需要确认或必须拒绝。

第二层是 sandbox：只负责运行 `run_shell` 及其子进程。当前 MVP 使用固定 cwd、`shell=False`、UTF-8 输出解码、环境标记和明显路径越界拦截；后续可以把 `ShellSandbox` 替换成真正的 OS/container sandbox。

第三层是 memory：读取 `CLAUDE.md`、`AGENTS.md`、`.code-agent/memory.md` 作为上下文提示。memory 不参与权限判断，也不能放行任何工具或命令。

## 上下文压缩

项目使用 LangGraph 节点自己管理上下文压缩，而不是直接套 LangChain 总结中间件。原因是代码任务状态不只有聊天消息，还包括 `changed_files`、`test_result`、审批状态和最近文件等结构化字段。

当消息数量或字符数超过阈值时，`context_manager` 会收集旧消息、工具调用、变更文件和验证结果，使用一个独立的 DeepSeek summary LLM 调用生成中文摘要。这个 summary LLM 不绑定工具，也不接收主会话完整 messages，避免压缩提示污染主 Agent 线程。压缩完成后，旧消息通过 LangGraph `RemoveMessage` 删除，只保留最近消息；摘要存入 `state.context_summary`，并在后续主 Agent LLM 调用时临时注入，不作为真实历史消息追加。

## 权限等级

- Level 0：只读操作，例如 `list_files`、`read_file`、`search_text`、`git_status`、`git_diff`
- Level 1：低风险写入，例如 `patch_file`、在 `src/` 下创建文件、更新测试；工具会记录 diff 提示
- Level 2：需要用户确认，例如修改 `package.json`、`pyproject.toml`、安装依赖、Docker compose、删除文件、大范围写入
- Level 3：直接拒绝，例如 `rm -rf`、`sudo`、`chmod 777`、`curl | bash`、`git reset --hard`、访问 `.env` 和 `.ssh`

Level 2 操作会返回机器可读的 `APPROVAL_REQUIRED[level_2]` 标记，并通过 LangGraph interrupt 暂停，等待用户确认后恢复执行。

## 审批流程

Level 2 操作使用 LangGraph interrupt：

1. 工具先返回 `APPROVAL_REQUIRED[level_2]`，不执行真实操作。
2. Graph 保存待执行工具名和参数。
3. CLI 展示风险原因，并询问 `y/n`。
4. 如果用户批准，Graph 恢复执行，并用内部 approval token 调用同一个工具。
5. 批准或拒绝都会写回同一个 `ToolMessage` 观察结果，而不是伪造成用户输入。
6. Agent 基于工具观察结果继续选择更安全的路径或停止。

可以这样测试：

```text
帮我修改 pyproject.toml 的 description
```

预期结果：CLI 会在写入前暂停，等待 Level 2 审批。

## 演示脚本

在 CLI 中依次尝试：

```text
/doctor
/tools
你好
你是什么模型
帮我解释这个项目的权限系统在哪里实现
帮我修改 pyproject.toml 的 description
/diff
```

其中 `你好` 和 `你是什么模型` 会由 AI router 判定为普通对话或一般问题，不进入代码工具链；`pyproject.toml` 修改请求会触发 Level 2 审批。Agent 运行过程中，CLI 会持续打印节点进度、工具调用、工具结果、验证决策和 diff 检查信息。
