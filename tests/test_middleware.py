from __future__ import annotations

import unittest

from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from code_agent.middleware import (
    ModelRetryMiddleware,
    PerModelToolCallLimitMiddleware,
    ToolErrorMiddleware,
)
from code_agent.services.errors import (
    ModelRetriesExhaustedError,
    classify_tool_error,
    is_retryable_model_error,
)


class PerModelToolCallLimitTests(unittest.TestCase):
    def test_executes_only_first_five_calls_from_one_response(self) -> None:
        middleware = PerModelToolCallLimitMiddleware(5)
        message = AIMessage(
            content="",
            tool_calls=[
                {"name": "read_file", "args": {"path": f"file-{index}.py"}, "id": f"call-{index}"}
                for index in range(8)
            ],
        )
        request = ModelRequest(model=object(), messages=[])

        response = middleware.wrap_model_call(
            request,
            lambda _request: ModelResponse(result=[message]),
        )

        self.assertIsInstance(response, ModelResponse)
        limited = response.result[0]
        self.assertIsInstance(limited, AIMessage)
        self.assertEqual(len(limited.tool_calls), 5)
        self.assertEqual([call["id"] for call in limited.tool_calls], [f"call-{index}" for index in range(5)])
        self.assertEqual(limited.additional_kwargs["tool_call_limit"]["omitted_count"], 3)
        self.assertIn("3 additional calls were omitted", str(limited.content))

    def test_response_within_limit_is_unchanged(self) -> None:
        middleware = PerModelToolCallLimitMiddleware(5)
        message = AIMessage(content="done")

        result = middleware._limit_response(message)

        self.assertIs(result, message)


class ModelRetryMiddlewareTests(unittest.TestCase):
    def test_retries_transient_failure_with_exponential_delay(self) -> None:
        delays: list[float] = []
        attempts = 0
        middleware = ModelRetryMiddleware(
            max_retries=2,
            total_timeout_seconds=120,
            base_delay_seconds=1,
            max_delay_seconds=6,
            sleep=delays.append,
            monotonic=lambda: 0,
            random_value=lambda: 0,
        )

        def handler(_request):
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise TimeoutError("upstream timed out")
            return ModelResponse(result=[AIMessage(content="ok")])

        result = middleware.wrap_model_call(ModelRequest(model=object(), messages=[]), handler)

        self.assertEqual(result.result[0].content, "ok")
        self.assertEqual(attempts, 3)
        self.assertEqual(delays, [1.0, 2.0])

    def test_does_not_retry_non_transient_failure(self) -> None:
        attempts = 0
        middleware = ModelRetryMiddleware(
            max_retries=2,
            total_timeout_seconds=120,
            base_delay_seconds=1,
            max_delay_seconds=6,
            sleep=lambda _delay: None,
        )

        def handler(_request):
            nonlocal attempts
            attempts += 1
            raise ValueError("invalid request")

        with self.assertRaises(ValueError):
            middleware.wrap_model_call(ModelRequest(model=object(), messages=[]), handler)

        self.assertEqual(attempts, 1)

    def test_reports_attempt_count_after_retryable_failures_are_exhausted(self) -> None:
        attempts = 0
        middleware = ModelRetryMiddleware(
            max_retries=2,
            total_timeout_seconds=390,
            base_delay_seconds=0,
            max_delay_seconds=0,
            sleep=lambda _delay: None,
        )

        def handler(_request):
            nonlocal attempts
            attempts += 1
            raise TimeoutError("upstream timed out")

        with self.assertRaises(ModelRetriesExhaustedError) as raised:
            middleware.wrap_model_call(ModelRequest(model=object(), messages=[]), handler)

        self.assertEqual(attempts, 3)
        self.assertEqual(raised.exception.attempts, 3)
        self.assertIsInstance(raised.exception.__cause__, TimeoutError)

    def test_uses_configured_fallback_after_primary_retries(self) -> None:
        primary = object()
        fallback = object()
        seen_models: list[object] = []
        middleware = ModelRetryMiddleware(
            max_retries=1,
            total_timeout_seconds=120,
            base_delay_seconds=0,
            max_delay_seconds=0,
            fallback_model=fallback,
            sleep=lambda _delay: None,
        )

        def handler(request):
            seen_models.append(request.model)
            if request.model is primary:
                raise ConnectionError("connection reset")
            return ModelResponse(result=[AIMessage(content="fallback ok")])

        result = middleware.wrap_model_call(ModelRequest(model=primary, messages=[]), handler)

        self.assertEqual(result.result[0].content, "fallback ok")
        self.assertEqual(seen_models, [primary, primary, fallback])

    def test_retry_classifier_rejects_auth_errors(self) -> None:
        class AuthError(RuntimeError):
            status_code = 401

        self.assertFalse(is_retryable_model_error(AuthError("unauthorized")))


class ToolErrorMiddlewareTests(unittest.TestCase):
    def test_collects_structured_shell_error(self) -> None:
        middleware = ToolErrorMiddleware()
        request = FakeToolRequest("shell_command", "call-1")
        message = ToolMessage(
            content=(
                "[sandbox: mode=read-only enforcement=partial denied=true]\n"
                "[sandbox: file access denied under read-only mode]\n"
                "Retry this exact command with sandbox_permissions='workspace-write'.\n"
                "[exit code: 1]"
            ),
            tool_call_id="call-1",
            name="shell_command",
        )

        result = middleware.wrap_tool_call(request, lambda _request: message)

        self.assertIsInstance(result, Command)
        error = result.update["tool_errors"][0]
        self.assertEqual(error["category"], "sandbox")
        self.assertEqual(error["code"], "sandbox_denied")
        self.assertTrue(error["retryable"])
        self.assertEqual(result.update["messages"][0].status, "error")

    def test_nonzero_exit_is_structured_but_not_retryable(self) -> None:
        error = classify_tool_error(
            "stderr:\nTypeScript error TS2322\n[exit code: 2]",
            tool_name="shell_command",
            tool_call_id="call-2",
        )

        self.assertIsNotNone(error)
        self.assertEqual(error["category"], "tool_execution")
        self.assertEqual(error["details"]["exit_code"], 2)
        self.assertFalse(error["retryable"])


class FakeToolRequest:
    def __init__(self, name: str, call_id: str) -> None:
        self.tool_call = {"name": name, "id": call_id, "args": {}}


if __name__ == "__main__":
    unittest.main()
