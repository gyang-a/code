# Code Agent

基于 LangGraph 工具调用构建的命令行编程智能体，以工作区为安全边界。

## 快速开始

```bash
code-agent .
```

CLI 会从所选工作区加载 `.env`。请设置 `DEEPSEEK_API_KEY`，也可以通过 `CODE_AGENT_MODEL` 指定模型。

所选工作区就是安全边界。默认情况下，`code-agent .` 使用当前工作目录。也可以显式指定项目目录，例如 `code-agent C:\path\to\project`。

## 图执行流程

```text
START
  -> agent
  -> execute
  -> tool_result_router
  -> context_manager
  -> agent | END
```

`agent` 节点是唯一由大语言模型作出决策的节点，负责决定检查文件、修改代码还是直接回答。图负责提供运行时上下文、执行项目工具、处理人工审批、跟踪文件变更，并在需要时压缩旧上下文。

`/help`、`/diff`、`/doctor`、`/tools`、`/resume` 和 `/undo` 等斜杠命令由 CLI 在图运行前处理。

## 权限等级

- 等级 0：只读工具，例如 `list_files`、`read_file`、`search_text`、`git_status` 和 `git_diff`。
- 等级 1：工作区内的低风险修改。
- 等级 2：需要用户审批，例如修改包元数据、删除文件、覆盖整个文件，或写入 `src/` 和 `tests/` 之外的位置。
- 等级 3：拒绝访问敏感或排除路径，例如 `.env`、`.ssh` 和 `.git`。

## 人工审批

等级 2 的工具调用遵循 LangChain/LangGraph 的人工介入流程：

1. 模型生成工具调用后，图会先检查整批调用，再执行任何工具。
2. 如果有调用需要审批，`execute` 会中断执行，并在同一个载荷中提供所有待处理调用的 `action_requests` 及对应的 `review_configs`。
3. CLI 使用顺序一致的 `decisions` 列表恢复执行。每项决策可以是 `approve`（批准）、`edit`（修改）、`reject`（拒绝）或 `respond`（直接回应）。
4. 获准或修改后的调用会执行，并生成仅包含工具结果的正常 `ToolMessage`。
5. 被拒绝或直接回应的调用会各自生成一条合成的 `ToolMessage`，与对应的工具调用匹配。
6. 智能体从恢复后的状态继续运行。

审批协议数据不会追加到 `messages` 中，只存在于中断载荷和恢复值中。

## 上下文压缩

当消息数量或估算字符数超过配置限制时，图会压缩旧消息。压缩时会完整保留近期的 `AIMessage(tool_calls) + ToolMessage` 消息块，避免工具结果与调用脱节；摘要保存在 `state.context_summary` 中，仅在后续模型调用时注入。

`AGENTS.md` 中的项目指令始终会被注入。`.code-agent/memory.md` 中的长期笔记按 Markdown 标题拆分，并在 `.code-agent/memory.sqlite3` 中建立索引；只有与用户当前目标相关的章节会加入模型上下文。

## 可靠性控制

每次模型响应在到达工具执行器之前，最多保留 5 个工具调用。超出的调用会被省略，并附上运行程序的说明，让模型在后续响应中继续剩余工作。此外，单次智能体运行还有独立的工具调用总量限制。

模型遇到超时、连接错误、HTTP 429 或 HTTP 5xx 等临时故障时，会采用指数退避和随机抖动重试两次。认证失败、校验失败及其他非临时故障不会重试。设置 `CODE_AGENT_FALLBACK_MODEL` 后，主模型重试耗尽时会使用另一个 DeepSeek 模型进行最后一次尝试。

默认每次 API 请求的超时时间为 120 秒，重试时间窗口为 390 秒。底层 OpenAI 兼容客户端的自动重试已禁用，以便 CLI 清楚展示每次尝试和重试次数。可通过以下配置调整这些参数：

- `CODE_AGENT_MODEL_REQUEST_TIMEOUT_SECONDS`
- `CODE_AGENT_MODEL_TOTAL_TIMEOUT_SECONDS`
- `CODE_AGENT_MODEL_MAX_RETRIES`

工具失败会以可读的 `ToolMessage` 返回，同时作为结构化 `AgentError` 记录到图状态和轮次追踪中。结构化记录包含来源、类别、稳定错误码、是否可重试、尝试次数、调用 ID，以及进程退出码等详细信息。

## 全局技能

Code Agent 可以在所有工作区之间复用技能库。默认情况下，技能保存在 Code Agent 自身的项目目录中，而不是正在编辑的工作区中：

```text
<code-agent-project>/.code-agent/skills/<skill-name>/SKILL.md
```

可通过 `CODE_AGENT_HOME`、`CODE_AGENT_SKILLS_DIR` 或 `CODE_AGENT_PENDING_SKILLS_DIR` 覆盖这些位置。旧版本保存在 `~/.code-agent/pending/skills` 下的待处理变更仍可读取，并可批准或拒绝。

每一轮都会将精简的全局技能索引注入运行时元数据。模型可以调用 `skills_list` 和 `skill_view` 按需加载相关技能。

当一轮工作使用了较多工具，或使用过写入工具时，后台审查智能体会判断是否有值得长期保存的操作知识。审查依据是结构化轮次追踪，而非压缩后的会话状态，因此工具调用、执行结果、文件变更、错误、验证、最终回答及已有技能都可供审计。发现有价值的技能后，CLI 会像工具审批一样展示变更建议：打印差异，询问 `y/n`，仅在批准后写入技能。可使用以下命令：

```text
/skills list
/skills path
/skills view <name>
```

为兼容旧的本地数据，仍支持以下待处理提案命令：

```text
/skills pending
/skills diff <id>
/skills approve <id>
/skills reject <id>
```

## 轮次追踪

智能体每一轮都会向项目本地的 SQLite 追踪数据库追加一条审计记录：

```text
.code-agent/traces.sqlite3
```

追踪数据库属于本地状态。Code Agent 会写入 `.code-agent/.gitignore`；如果工作区是 Git 仓库，还会将 `.code-agent/` 加入 `.git/info/exclude`，以避免追踪记录和检查点进入项目 Git 历史。技能审查会读取项目追踪记录，从而了解跨轮次的用户反馈和纠正。

随着项目追踪记录增长，审查使用的数据范围会受到限制：结构化历史摘要加上最近的原始轮次记录。最近 100 轮保留完整载荷，更早的轮次则由聚合摘要表示。已有的 `.code-agent/traces/project_trace.json` 文件会被导入一次，并保留为旧版备份。

程序还会自动刷新一份大小受限、便于人工阅读的镜像文件：

```text
.code-agent/traces/project_trace.readable.json
```

其中包含聚合摘要和最近 20 轮的完整记录。在 CLI 中使用 `/trace` 或 `/trace 10`，可查看精简的近期轮次记录及镜像文件路径。

## 撤销修改

`/undo` 仅作用于智能体最近一轮的修改。每轮开始时会记录 Git 中已有未提交改动的路径，并跟踪智能体实际修改的文件。

如果智能体修改的文件在本轮开始前就有未提交改动，Code Agent 会在 `.code-agent/undo/<turn-id>/` 下保存修改前的私有快照，撤销时恢复该快照。对于本轮开始时干净的文件，撤销使用 Git 恢复；对于智能体新建的文件，撤销只删除这些文件。

## 工具

模型可以调用受限的文件系统、搜索、Git、技能和 Windows PowerShell 工具。`read_file` 默认只返回有限范围的文本；智能体需要指定后续的 `start_line` 才能继续读取。

`shell_command` 采用三种执行决策：明确允许的操作直接执行；删除、安装、任意脚本等需要审批；已识别的越界、敏感路径和禁止操作直接拒绝。PowerShell AST 解析器会检查复合命令，而不是只匹配第一个命令名。解析失败时拒绝执行。

沙箱模式与审批策略彼此独立，默认 `workspace-write + on-risk`。不再要求先以只读模式执行失败，也不再提供 `danger-full-access` 重试。审批只允许本次具体操作，不改变沙箱权限；批准后修改命令仍会重新检查硬性禁止规则。

```powershell
# 默认：已识别读取直接运行，其余未授权命令审批
code-agent .

# 明确信任当前项目的两个开发入口，避免反复审批
code-agent . --allow-shell 'npm run build' --allow-shell 'npm test'

# 工作区只读；需要审批的命令直接拒绝
code-agent . --sandbox-mode read-only --approval-policy never

# 不采用项目命令豁免，只让已识别的读取操作自动执行
code-agent . --approval-policy untrusted
```

`--allow-shell` 精确匹配整条命令，仅在本次 CLI 会话中有效；明确删除命令、动态脚本和禁止路径不会因此放行。它表示用户信任项目代码，**不表示已证明构建/测试脚本安全**。不要把未知项目的执行入口加入允许列表。模式和规则由宿主配置，不从 Agent 可写的项目规则文件加载。

文件工具拒绝修改 `.git`、`.agents`、`.codex`、`.code-agent`。Shell 静态检查保守地拒绝显式访问这些路径，以及 `.env`、私钥等敏感路径。Windows 后端在执行前给已存在的受保护路径添加针对沙箱 SID 的拒绝写入/删除 ACE，并限制父目录删除子项的权限；普通宿主进程不持有这个 SID。发现链接或 junction 时拒绝初始化，避免修改链接目标的 ACL。能力 ACL 会保留在工作区中。

每次调用启动新的 PowerShell 进程，显式定位到 `workdir`。临时目录在两种模式下均可写，npm、uv、pip 缓存重定向到本次临时目录；执行结束后清理。超时和 Job Object 继续负责清理子进程。沙箱初始化失败时拒绝执行，不回退到无约束进程。输出中的权限关键词只用于诊断，不再赋予提权重试资格。

**现有 Windows 后端仍是部分写入隔离。** 它保留 `WRITE_RESTRICTED` 令牌：动态脚本的读取行为和网络访问没有完整的操作系统隔离；Windows 对 Everyone 等主体开放写入的对象以及硬链接仍有平台限制。敏感路径读取的静态检查不能看透任意 Python/Node 脚本；未来在执行中创建的敏感路径也不是通配符 ACL 保护。批准任意脚本不能被理解为安全证明。`read-only` 指工作区写约束，允许本次私有临时目录写入。真正的完整读取/网络隔离需要专用低权限用户或独立隔离后端。

验证：`python -m pytest`；Windows 原生访问测试使用临时工作区及模拟普通项目的 ACL，覆盖普通写入、保护路径写入/删除、工作区外写入、只读写入拒绝。若测试运行器自身被外层沙箱限制，需从正常 Windows 用户终端运行原生测试。

## 常用命令

```text
/doctor
/tools
/diff
/undo
/sessions
/trace [n]
/resume [thread-id]
/help
```

在正在运行的 CLI 会话中，`/sessions` 会列出工作区已保存的会话线程。`/resume` 会列出已保存的线程，并提示输入线程 ID、唯一前缀或列表序号；`/resume [thread-id]` 则直接切换到指定线程。
