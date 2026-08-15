from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

from code_agent.models import AgentError


RETRYABLE_HTTP_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
NON_RETRYABLE_HTTP_STATUS_CODES = {400, 401, 403, 404, 405, 422}
_EXIT_CODE_RE = re.compile(r"\[exit code:\s*(-?\d+)\]", re.IGNORECASE)


def iter_exception_chain(exc: BaseException) -> Iterator[BaseException]:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def is_timeout_error(exc: BaseException) -> bool:
    for current in iter_exception_chain(exc):
        name = type(current).__name__.lower()
        message = str(current).lower()
        if isinstance(current, TimeoutError) or "timeout" in name or "timed out" in message:
            return True
    return False


def is_retryable_model_error(exc: BaseException) -> bool:
    for current in iter_exception_chain(exc):
        status_code = _status_code(current)
        if status_code in NON_RETRYABLE_HTTP_STATUS_CODES:
            return False
        if status_code in RETRYABLE_HTTP_STATUS_CODES or (status_code is not None and status_code >= 500):
            return True

        name = type(current).__name__.lower()
        message = str(current).lower()
        if isinstance(current, (TimeoutError, ConnectionError)):
            return True
        if any(token in name for token in ("timeout", "connection", "ratelimit", "serviceunavailable")):
            return True
        if any(
            token in message
            for token in (
                "timed out",
                "connection reset",
                "connection refused",
                "temporarily unavailable",
                "rate limit",
                "too many requests",
            )
        ):
            return True
    return False


def model_error(exc: BaseException, *, attempt: int) -> AgentError:
    timeout = is_timeout_error(exc)
    return {
        "source": "model",
        "category": "timeout" if timeout else "network" if is_retryable_model_error(exc) else "model",
        "code": "model_timeout" if timeout else "model_request_failed",
        "message": str(exc),
        "retryable": is_retryable_model_error(exc),
        "attempt": attempt,
        "tool_call_id": None,
        "details": {"exception_type": type(exc).__name__},
    }


def classify_tool_error(
    content: str,
    *,
    tool_name: str,
    tool_call_id: str | None,
    status: str | None = None,
) -> AgentError | None:
    text = content.strip()
    lowered = text.lower()
    exit_match = _EXIT_CODE_RE.search(text)
    exit_code = int(exit_match.group(1)) if exit_match else None

    is_error = (
        status == "error"
        or text.startswith("ERROR:")
        or text.startswith("REJECTED[")
        or "denied=true" in lowered
        or "[timed out]" in lowered
        or (exit_code is not None and exit_code != 0)
    )
    if not is_error:
        return None

    if text.startswith("REJECTED["):
        category = "permission"
        code = "permission_rejected"
        retryable = False
    elif "denied=true" in lowered or "sandbox:" in lowered and "access denied" in lowered:
        category = "sandbox"
        code = "sandbox_denied"
        retryable = "retry this exact" in lowered or "sandbox_permissions=" in lowered
    elif "[timed out]" in lowered:
        category = "timeout"
        code = "tool_timeout"
        retryable = True
    elif text.startswith("ERROR:") and any(
        token in lowered for token in ("invalid", "missing", "does not exist", "not found", "appears ")
    ):
        category = "validation"
        code = "tool_validation_error"
        retryable = False
    else:
        category = "tool_execution"
        code = "nonzero_exit" if exit_code is not None else "tool_error"
        retryable = False

    details: dict[str, Any] = {}
    if exit_code is not None:
        details["exit_code"] = exit_code

    return {
        "source": tool_name,
        "category": category,
        "code": code,
        "message": text,
        "retryable": retryable,
        "attempt": 1,
        "tool_call_id": tool_call_id,
        "details": details,
    }


def _status_code(exc: BaseException) -> int | None:
    direct = getattr(exc, "status_code", None)
    if isinstance(direct, int):
        return direct
    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    return response_status if isinstance(response_status, int) else None
