from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage

from code_agent.config import AgentConfig
from code_agent.context_middleware import ContextManagementMiddleware, ConversationSummary, ContextBudgetExceeded, estimate_tokens
from code_agent.middleware import ToolErrorMiddleware
from code_agent.services.context_archive import ContextArchive
from code_agent.services.summarizer import truncate, raw_tool_output
from code_agent.services.workspace import Workspace
from code_agent.tools.fs import build_read_file_tool
from code_agent.tools.shell import _truncate_tail


class Summarizer:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def invoke(self, messages):
        self.calls.append(messages)
        if self.fail:
            raise ValueError('summary unavailable')
        return ConversationSummary(goal='fix auth', constraints=['do not change public API'],
                                   decisions=[], completed=[], changed_files=[], validation=[],
                                   unresolved=['auth failure'], next_steps=['inspect auth'], evidence=['archive'])


def manager(tmp_path, **changes):
    config = replace(AgentConfig(), context_keep_recent_turns=1, **changes)
    return ContextManagementMiddleware(tmp_path, config, Summarizer())


def tool_result(mw, content, name='read_file', thread='a', call_id='call'):
    request = SimpleNamespace(state={'thread_id': thread},
                              tool_call={'name': name, 'id': call_id, 'args': {'path': 'src/auth.py'}})
    result = mw.wrap_tool_call(request, lambda _: ToolMessage(content=content, name=name, tool_call_id=call_id))
    return result


def request(messages, **kwargs):
    return ModelRequest(model=object(), messages=messages, state={'thread_id': 'a'}, **kwargs)


def invoke(mw, req):
    seen = []
    def handler(value):
        seen.extend(value.messages)
        return ModelResponse(result=[AIMessage(content='done')])
    mw.wrap_model_call(req, handler)
    return seen


def test_tool_archives_before_head_truncation_and_thread_isolation(tmp_path):
    mw = manager(tmp_path, context_tool_token_limit=512)
    original = '\n'.join(f'{i}: ' + '中' * 100 for i in range(1, 120))
    result = tool_result(mw, original)
    ref = result.artifact['context_archive']['ref']
    assert mw.archive.read('a', ref) == original
    assert result.content.startswith('1:')
    assert '内容过长已截断' in result.content
    assert estimate_tokens(result.content) <= 512
    assert result.artifact['context_archive']['returned_lines'] == 119
    assert result.artifact['context_archive']['returned_range'] == '1-119'
    with pytest.raises(ValueError):
        mw.archive.read('other', ref)
    assert mw.archive.search('other', '中') == []
    assert mw.archive.search('a', '119:', ref)[0]['line'] == 119


def test_single_line_and_shell_status_are_bounded(tmp_path):
    mw = manager(tmp_path, context_tool_token_limit=512)
    result = tool_result(mw, '[sandbox: mode=read-only]\n' + 'x' * 20000 + '\n[timed out]\n[exit code: 1]', name='shell_command')
    assert '[exit code: 1]' in result.content
    assert '[timed out]' in result.content
    assert estimate_tokens(result.content) <= 512


def test_tool_local_truncation_is_bypassed_only_during_execution(tmp_path):
    mw = manager(tmp_path)
    req = SimpleNamespace(state={'thread_id': 'a'}, tool_call={'name': 'read_file', 'id': 'x', 'args': {}})
    def handler(_):
        assert truncate('abc' * 100, 2) == 'abc' * 100
        assert _truncate_tail('abc' * 100, 2) == 'abc' * 100
        raise RuntimeError('failure')
    with pytest.raises(RuntimeError):
        mw.wrap_tool_call(req, handler)
    assert not raw_tool_output.get()
    assert truncate('abcd', 2) != 'abcd'


def history(mw, text):
    result = tool_result(mw, text)
    result.id = 't'
    return [HumanMessage(content='old task', id='h1'),
            AIMessage(content='', id='ai', tool_calls=[{'name': 'read_file', 'args': {'path': 'src/auth.py'}, 'id': 'call'}]),
            result, HumanMessage(content='current task', id='h2')]


def test_masks_old_result_preserves_tool_pair_and_skips_summary(tmp_path):
    mw = manager(tmp_path, context_token_limit=6000, context_tool_token_limit=10000)
    messages = history(mw, 'evidence\n' * 2000)
    seen = invoke(mw, request(messages))
    assert 'Tool result omitted' in seen[2].content
    assert seen[2].tool_call_id == seen[1].tool_calls[0]['id']
    assert seen[-1] is messages[-1]
    assert 'Tool result omitted' not in messages[2].content
    assert mw.summary_model.calls == []
    # The same thread can resume with the original checkpoint messages.
    resumed = ContextManagementMiddleware(tmp_path, mw.config, Summarizer())
    assert 'Tool result omitted' in invoke(resumed, request(messages))[2].content


def test_summary_runs_when_masking_cannot_reach_target_and_survives_resume(tmp_path):
    mw = manager(tmp_path, context_token_limit=6000)
    messages = [HumanMessage(content='old ' * 4000, id='old'),
                AIMessage(content='details ' * 1000, id='reply'),
                HumanMessage(content='current', id='current')]
    seen = invoke(mw, request(messages))
    assert 'Structured summary' in seen[0].content
    assert 'do not change public API' in seen[0].content
    assert seen[-1] is messages[-1]
    assert mw.summary_model.calls
    assert mw.archive.search('a', 'old old')
    resumed = ContextManagementMiddleware(tmp_path, mw.config, Summarizer())
    again = invoke(resumed, request(messages))
    assert again[0].content == seen[0].content
    assert resumed.summary_model.calls == []


def test_recent_rounds_remain_complete_and_overflow_is_explicit(tmp_path):
    mw = manager(tmp_path, context_token_limit=2000, context_tool_token_limit=10000)
    messages = history(mw, 'x' * 10000)[:-1]
    with pytest.raises(ContextBudgetExceeded):
        invoke(mw, request(messages))
    assert mw.summary_model.calls == []


def test_system_context_counts_toward_hard_budget(tmp_path):
    mw = manager(tmp_path, context_token_limit=2000)
    with pytest.raises(ContextBudgetExceeded):
        invoke(mw, request([HumanMessage(content='hello')], system_message=SystemMessage(content='x' * 9000)))


def test_summary_failure_keeps_originals_and_does_not_call_overbudget_model(tmp_path):
    mw = manager(tmp_path, context_token_limit=3000)
    mw.summary_model = Summarizer(fail=True)
    messages = [HumanMessage(content='old ' * 4000, id='old'), HumanMessage(content='now', id='new')]
    with pytest.raises(ContextBudgetExceeded):
        invoke(mw, request(messages))
    assert mw.archive.projection('a').get('summary') == ''
    assert mw.archive.search('a', 'old old')


def test_parallel_archival_and_idempotent_originals(tmp_path):
    mw = manager(tmp_path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda i: tool_result(mw, f'output {i}', call_id=str(i)), range(20)))
    assert len({r.artifact['context_archive']['ref'] for r in results}) == 20
    reopened = ContextArchive(tmp_path)
    assert all(reopened.read('a', r.artifact['context_archive']['ref']) == r.content for r in results)


class ToolModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def test_real_agent_tool_roundtrip_archives_untruncated_read(tmp_path):
    original = 'z' * 20000 + '\nsecond line\n'
    (tmp_path / 'auth.py').write_text(original)
    mw = manager(tmp_path, context_tool_token_limit=512)
    model = ToolModel(responses=[AIMessage(content='', tool_calls=[{'name': 'read_file', 'args': {'path': 'auth.py'}, 'id': 'call'}]), AIMessage(content='finished')])
    graph = create_agent(model, tools=[build_read_file_tool(Workspace(tmp_path), output_limit=10)],
                         middleware=[mw, ToolErrorMiddleware()])
    result = graph.invoke({'messages': [HumanMessage(content='read auth')], 'thread_id': 'a'},
                          config={'configurable': {'thread_id': 'a'}})
    output = next(m for m in result['messages'] if isinstance(m, ToolMessage))
    assert '内容过长已截断' in output.content
    assert 'z' * 20000 in mw.archive.read('a', output.artifact['context_archive']['ref'])
    assert mw.archive.search('a', 'finished')


def test_recovery_tools_work_through_real_agent_runtime(tmp_path):
    mw = manager(tmp_path)
    ref = mw.archive.put('a', 'tool', {'id': 'original'}, 'first\nneedle\nlast')
    model = ToolModel(responses=[
        AIMessage(content='', tool_calls=[{'name': 'read_context_history',
                   'args': {'ref': ref, 'offset_chars': 6, 'limit_chars': 6}, 'id': 'recover'}]),
        AIMessage(content='recovered')])
    graph = create_agent(model, middleware=[mw, ToolErrorMiddleware()])
    result = graph.invoke({'messages': [HumanMessage(content='recover')]},
                          config={'configurable': {'thread_id': 'a'}})
    output = next(m for m in result['messages'] if isinstance(m, ToolMessage))
    assert output.content.endswith('needle')
    assert output.status == 'success'


def test_error_command_preserves_state_updates_and_archives_error(tmp_path):
    from langgraph.types import Command
    mw = manager(tmp_path)
    error_handler = ToolErrorMiddleware()
    req = SimpleNamespace(state={'thread_id': 'a'}, tool_call={'name': 'read_file', 'id': 'err', 'args': {}})
    def fail(_):
        raise ValueError('missing file')
    result = mw.wrap_tool_call(req, lambda r: error_handler.wrap_tool_call(r, fail))
    assert isinstance(result, Command)
    assert result.update['tool_errors'][0]['code'] == 'tool_exception'
    message = result.update['messages'][0]
    assert message.status == 'error'
    assert mw.archive.read('a', message.artifact['context_archive']['ref']) == 'ERROR: missing file'


def test_checkpoint_rollback_restores_recent_tool_result(tmp_path):
    mw = manager(tmp_path, context_token_limit=6000, context_tool_token_limit=10000)
    messages = history(mw, 'evidence\n' * 2000)
    assert 'Tool result omitted' in invoke(mw, request(messages))[2].content
    # Restoring the earlier checkpoint makes that tool part of the protected current turn.
    with pytest.raises(ContextBudgetExceeded):
        invoke(mw, request(messages[:-1]))
    assert mw.summary_model.calls == []


def test_production_graph_wires_context_manager(tmp_path, monkeypatch):
    from code_agent.graph import build_graph
    class FactoryModel(ToolModel):
        def with_structured_output(self, schema, **kwargs):
            assert schema is ConversationSummary
            assert kwargs['method'] == 'json_mode'
            return Summarizer()
    monkeypatch.setattr('code_agent.graph.init_chat_model',
                        lambda *args, **kwargs: FactoryModel(responses=[AIMessage(content='wired')]))
    graph = build_graph(str(tmp_path), AgentConfig())
    result = graph.invoke({'messages': [HumanMessage(content='hello')], 'thread_id': 'wired'},
                         config={'configurable': {'thread_id': 'wired'}})
    assert result['messages'][-1].content == 'wired'
    assert ContextArchive(tmp_path).search('wired', 'wired')
