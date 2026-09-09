"""Bind host HITL decisions to exact Shell calls, including edited calls."""
from langchain.agents.middleware import HumanInTheLoopMiddleware


class ShellApprovalMiddleware(HumanInTheLoopMiddleware):
    def __init__(self, *, tools, **kwargs):
        super().__init__(**kwargs)
        self._tools_by_name = {tool.name: tool for tool in tools}

    def _process_decision(self, decision, tool_call, config):
        revised, message = super()._process_decision(decision, tool_call, config)
        if revised is not None and message is None:
            tool = self._tools_by_name.get(revised['name'])
            authorize = getattr(tool, '_authorize', None)
            if authorize:
                authorize(revised['args'])
        return revised, message
