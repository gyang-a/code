from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from code_agent.models import AgentError
from code_agent.services.errors import (
    ModelRetriesExhaustedError,
    classify_tool_error,
    is_retryable_model_error,
)


class PerModelToolCallLimitMiddleware(AgentMiddleware):
    """Hard-limit the tool calls emitted by one model response."""

    def __init__(self, limit: int) -> None:
        super().__init__()
        if limit < 1:
            raise ValueError("Per-model tool call limit must be at least 1.")
        self.limit = limit

    def wrap_model_call(self, request: ModelRequest, handler) -> ModelResponse | AIMessage:
        return self._limit_response(handler(request))

    def _limit_response(self, response: ModelResponse | AIMessage) -> ModelResponse | AIMessage:
        if isinstance(response, AIMessage):
            return self._limit_message(response)

        limited_result = [
            self._limit_message(message) if isinstance(message, AIMessage) else message
            for message in response.result
        ]
        return ModelResponse(
            result=limited_result,
            structured_response=response.structured_response,
        )

    def _limit_message(self, message: AIMessage) -> AIMessage:
        tool_calls = list(message.tool_calls or [])
        if len(tool_calls) <= self.limit:
            return message

        omitted = len(tool_calls) - self.limit
        notice = (
            f"[Host tool-call limit: executing the first {self.limit} calls; "
            f"{omitted} additional calls were omitted. Request remaining work in a later response.]"
        )
        content: Any = message.content
        if isinstance(content, str):
            content = f"{content}\n\n{notice}" if content else notice
        elif isinstance(content, list):
            content = [*content, {"type": "text", "text": notice}]
        else:
            content = notice

        additional_kwargs = dict(message.additional_kwargs)
        additional_kwargs["tool_call_limit"] = {
            "limit": self.limit,
            "original_count": len(tool_calls),
            "omitted_count": omitted,
        }
        return message.model_copy(
            update={
                "content": content,
                "tool_calls": tool_calls[: self.limit],
                "additional_kwargs": additional_kwargs,
            },
            deep=True,
        )


class ModelRetryMiddleware(AgentMiddleware):
    """Retry transient model failures within one bounded time budget."""

    def __init__(
        self,
        *,
        max_retries: int,
        total_timeout_seconds: float,
        base_delay_seconds: float,
        max_delay_seconds: float,
        fallback_model: Any = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        random_value: Callable[[], float] = random.random,
        on_retry: Callable[[int, int, float, Exception], None] | None = None,
        on_fallback: Callable[[Exception], None] | None = None,
    ) -> None:
        super().__init__()
        self.max_retries = max(0, max_retries)
        self.total_timeout_seconds = max(0.0, total_timeout_seconds)
        self.base_delay_seconds = max(0.0, base_delay_seconds)
        self.max_delay_seconds = max(0.0, max_delay_seconds)
        self.fallback_model = fallback_model
        self.sleep = sleep
        self.monotonic = monotonic
        self.random_value = random_value
        self.on_retry = on_retry
        self.on_fallback = on_fallback

    def wrap_model_call(self, request: ModelRequest, handler):
        started_at = self.monotonic()
        last_error: Exception | None = None
        total_attempts = self.max_retries + 1
        attempts_made = 0

        for attempt in range(1, total_attempts + 1):
            attempts_made = attempt
            try:
                return handler(request)
            except Exception as exc:
                last_error = exc
                if not is_retryable_model_error(exc):
                    raise
                if attempt >= total_attempts:
                    break
                delay = self._retry_delay(attempt)
                if self._would_exceed_budget(started_at, delay):
                    break
                if self.on_retry is not None:
                    self.on_retry(attempt + 1, total_attempts, delay, exc)
                self.sleep(delay)

        if self.fallback_model is not None and last_error is not None and is_retryable_model_error(last_error):
            if not self._would_exceed_budget(started_at, 0):
                if self.on_fallback is not None:
                    self.on_fallback(last_error)
                try:
                    return handler(request.override(model=self.fallback_model))
                except Exception as exc:
                    if not is_retryable_model_error(exc):
                        raise
                    last_error = exc
                    attempts_made += 1

        assert last_error is not None
        raise ModelRetriesExhaustedError(
            attempts=attempts_made,
            last_error=last_error,
        ) from last_error

    def _retry_delay(self, failed_attempt: int) -> float:
        exponential = self.base_delay_seconds * (2 ** (failed_attempt - 1))
        jittered = exponential * (1.0 + max(0.0, min(self.random_value(), 1.0)))
        return min(jittered, self.max_delay_seconds)

    def _would_exceed_budget(self, started_at: float, delay: float) -> bool:
        if self.total_timeout_seconds <= 0:
            return False
        return self.monotonic() - started_at + delay >= self.total_timeout_seconds


class ToolErrorMiddleware(AgentMiddleware):
    """Attach structured errors to state while preserving model-readable results."""

    def wrap_tool_call(self, request, handler):
        tool_name = str(request.tool_call.get("name") or "unknown")
        tool_call_id = str(request.tool_call.get("id") or "") or None
        try:
            result = handler(request)
        except Exception as exc:
            error: AgentError = {
                "source": tool_name,
                "category": "tool_execution",
                "code": "tool_exception",
                "message": str(exc),
                "retryable": False,
                "attempt": 1,
                "tool_call_id": tool_call_id,
                "details": {"exception_type": type(exc).__name__},
            }
            message = ToolMessage(
                content=f"ERROR: {exc}",
                tool_call_id=tool_call_id or "unknown",
                name=tool_name,
                status="error",
                artifact={"agent_error": error},
            )
            return Command(update={"messages": [message], "tool_errors": [error]})

        if not isinstance(result, ToolMessage):
            return result

        error = classify_tool_error(
            str(result.content or ""),
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            status=result.status,
        )
        if error is None:
            return result

        artifact = dict(result.artifact) if isinstance(result.artifact, dict) else {}
        artifact["agent_error"] = error
        error_message = result.model_copy(
            update={"status": "error", "artifact": artifact},
            deep=True,
        )
        return Command(update={"messages": [error_message], "tool_errors": [error]})
