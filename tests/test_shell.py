import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from pydantic import ValidationError
from code_agent.models import SandboxMode
from code_agent.services.shell_policy import classify_shell
from code_agent.services.shell_approval import ShellApprovalMiddleware
from code_agent.services.windows_sandbox import ShellExecutionResult, _sanitized_environment
from code_agent.services.workspace import Workspace
from code_agent.tools.schemas import ShellCommandInput
from code_agent.tools.shell import build_shell_command_tool


class FakeSandbox:
    def __init__(self):
        self.calls = []

    def run(self, spec):
        self.calls.append(spec)
        return ShellExecutionResult(0, 'ok', '', False, False, spec.mode)


class ShellTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.workspace = Workspace(self.tmp.name)
        self.executor = FakeSandbox()
        self.tool = build_shell_command_tool(self.workspace, executor=self.executor)

    def args(self, command):
        return {'command': command, 'description': 'Test command'}

    def test_reads_run_in_default_workspace_write_without_approval(self):
        self.tool.invoke(self.args('Get-Content README.md'))
        self.assertEqual(self.executor.calls[0].mode, SandboxMode.workspace_write)

    def test_approval_does_not_change_read_only_mode(self):
        self.workspace.shell_mode = 'read-only'
        args = self.args('python task.py')
        self.tool._authorize(args)
        self.tool.invoke(args)
        self.assertEqual(self.executor.calls[0].mode, SandboxMode.read_only)

    def test_unknown_script_requires_exact_one_shot_host_approval(self):
        args = self.args('python task.py')
        self.assertIn('Approval required', self.tool.invoke(args))
        self.tool._authorize(args)
        self.assertIn('Approval required', self.tool.invoke(self.args('python other.py')))
        self.assertIn('exit code: 0', self.tool.invoke(args))
        self.assertIn('Approval required', self.tool.invoke(args))
        self.assertEqual(len(self.executor.calls), 1)

    def test_host_allowed_build_does_not_need_failure_or_approval(self):
        self.workspace.shell_allowed_commands = ('npm run build',)
        self.tool.invoke(self.args('npm run build'))
        self.assertEqual(len(self.executor.calls), 1)

    def test_untrusted_and_never_are_independent_of_sandbox(self):
        self.workspace.shell_allowed_commands = ('npm run build',)
        self.workspace.shell_approval_policy = 'untrusted'
        self.assertTrue(classify_shell(self.workspace, self.args('npm run build')).requires_approval)
        self.workspace.shell_approval_policy = 'never'
        self.assertEqual(classify_shell(self.workspace, self.args('npm run build')).risk.value, 'level_3')

    def test_denied_targets_never_receive_approval(self):
        for command in ['Remove-Item . -Recurse -Force', 'Get-Content ../secret.txt',
                        'Set-Content .git/config x', 'Get-Content .env',
                        'Remove-Item .codex -Recurse', 'iex "echo hello"',
                        'Get-Content README.md; Remove-Item . -Recurse']:
            with self.subTest(command=command):
                args = self.args(command)
                self.tool._authorize(args)
                self.assertIn('REJECTED[level_3]', self.tool.invoke(args))
        self.assertEqual(self.executor.calls, [])

    def test_compound_delete_and_dynamic_commands_require_approval(self):
        for command in ['Get-Content README.md; Remove-Item src/old -Recurse',
                        '$p = "src/old"; Remove-Item $p', 'Write-Output hi > output.txt']:
            with self.subTest(command=command):
                self.assertTrue(classify_shell(self.workspace, self.args(command)).requires_approval)

    def test_legacy_escalation_fields_are_rejected(self):
        with self.assertRaises(ValidationError):
            ShellCommandInput(**self.args('pytest'), sandbox_permissions='danger-full-access')

    def test_edited_approval_cannot_bypass_deny(self):
        middleware = ShellApprovalMiddleware(tools=[self.tool], interrupt_on={'shell_command': True})
        original = dict(name='shell_command', args=self.args('python task.py'), id='test')
        revised, _ = middleware._process_decision(
            {'type': 'edit', 'edited_action': {'name': 'shell_command', 'args': self.args('Remove-Item . -Recurse')}},
            original, {'allowed_decisions': ['approve', 'edit']})
        self.assertIn('REJECTED[level_3]', self.tool.invoke(revised['args']))
        self.assertEqual(self.executor.calls, [])

    def test_timeout_cap(self):
        tool = build_shell_command_tool(self.workspace, executor=self.executor, max_timeout_ms=100)
        tool.invoke({**self.args('Get-Content README.md'), 'timeout_ms': 5000})
        self.assertEqual(self.executor.calls[0].timeout_ms, 100)

    def test_environment_filters_secrets_and_redirects_caches(self):
        with patch.dict('os.environ', {'DEEPSEEK_API_KEY': 'secret', 'SAFE_SETTING': 'visible'}, clear=True):
            env = _sanitized_environment(Path(self.tmp.name))
        self.assertNotIn('DEEPSEEK_API_KEY', env)
        self.assertEqual(env['SAFE_SETTING'], 'visible')
        self.assertTrue(env['UV_CACHE_DIR'].startswith(self.tmp.name))

    def test_real_graph_interrupt_resume_binds_approval(self):
        from langchain.agents import create_agent
        from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
        from langchain_core.messages import AIMessage, HumanMessage
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.types import Command
        from code_agent.graph import _approval_interrupt_config

        class Model(FakeMessagesListChatModel):
            def bind_tools(self, tools, **kwargs):
                return self

        model = Model(responses=[AIMessage(content='', tool_calls=[
            {'name': 'shell_command', 'args': self.args('python task.py'), 'id': 'call-1'}]),
            AIMessage(content='done')])
        middleware = ShellApprovalMiddleware(tools=[self.tool],
            interrupt_on=_approval_interrupt_config(self.workspace, [self.tool]))
        agent = create_agent(model, tools=[self.tool], middleware=[middleware], checkpointer=InMemorySaver())
        config = {'configurable': {'thread_id': 'shell-approval-test'}}
        pending = agent.invoke({'messages': [HumanMessage(content='Run task')]}, config)
        self.assertTrue(pending.get('__interrupt__'))
        self.assertEqual(self.executor.calls, [])
        finished = agent.invoke(Command(resume={'decisions': [{'type': 'approve'}]}), config)
        self.assertEqual(len(self.executor.calls), 1)
        self.assertEqual(self.executor.calls[0].mode, SandboxMode.workspace_write)
        self.assertEqual(finished['messages'][-1].content, 'done')
