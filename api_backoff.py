import asyncio
import random
import re
import time
from typing import Awaitable, Callable, TypeVar


T = TypeVar("T")

DEFAULT_MAX_ATTEMPTS = 6
DEFAULT_INITIAL_DELAY_SECONDS = 1.0
DEFAULT_MAX_DELAY_SECONDS = 30.0
DEFAULT_BACKOFF_MULTIPLIER = 2.0
DEFAULT_JITTER_RATIO = 0.2
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
TRANSIENT_ERROR_FRAGMENTS = (
    "timeout",
    "timed out",
    "temporarily unavailable",
    "temporary failure",
    "rate limit",
    "resource exhausted",
    "connection reset",
    "connection aborted",
    "connection refused",
    "server error",
    "internal error",
    "service unavailable",
    "bad gateway",
    "gateway timeout",
)


def _extract_status_code(exc: Exception) -> int | None:
    for attr_name in ("status_code", "code", "status", "http_status"):
        raw_value = getattr(exc, attr_name, None)
        if raw_value is None:
            continue

        if isinstance(raw_value, int):
            return raw_value

        nested_value = getattr(raw_value, "value", None)
        if isinstance(nested_value, int):
            return nested_value

        try:
            return int(raw_value)
        except (TypeError, ValueError):
            continue

    match = re.search(r"\b(408|429|500|502|503|504)\b", str(exc))
    if match:
        return int(match.group(1))
    return None


def _is_retryable_exception(exc: Exception) -> bool:
    status_code = _extract_status_code(exc)
    if status_code is not None:
        if status_code in RETRYABLE_STATUS_CODES:
            return True
        if 400 <= status_code < 500:
            return False

    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return True

    class_name = exc.__class__.__name__.lower()
    module_name = exc.__class__.__module__.lower()
    message = str(exc).lower()

    if any(fragment in class_name for fragment in ("timeout", "unavailable", "connection")):
        return True

    if any(fragment in message for fragment in TRANSIENT_ERROR_FRAGMENTS):
        return True

    return module_name.startswith("google") and status_code is None


def _compute_delay_seconds(attempt_number: int) -> float:
    base_delay = min(
        DEFAULT_MAX_DELAY_SECONDS,
        DEFAULT_INITIAL_DELAY_SECONDS * (DEFAULT_BACKOFF_MULTIPLIER ** max(attempt_number - 1, 0)),
    )
    jitter_multiplier = random.uniform(1.0 - DEFAULT_JITTER_RATIO, 1.0 + DEFAULT_JITTER_RATIO)
    return max(0.0, base_delay * jitter_multiplier)


def call_with_exponential_backoff(
    operation: Callable[[], T],
    *,
    description: str = "Gemini API call",
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> T:
    attempt_number = 1
    while True:
        try:
            return operation()
        except Exception as exc:
            should_retry = attempt_number < max_attempts and _is_retryable_exception(exc)
            if not should_retry:
                raise

            delay_seconds = _compute_delay_seconds(attempt_number)
            print(
                f"{description} failed on attempt {attempt_number}/{max_attempts}: {exc}. "
                f"Retrying in {delay_seconds:.1f}s..."
            )
            time.sleep(delay_seconds)
            attempt_number += 1


async def async_call_with_exponential_backoff(
    operation: Callable[[], Awaitable[T]],
    *,
    description: str = "Gemini API call",
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> T:
    attempt_number = 1
    while True:
        try:
            return await operation()
        except Exception as exc:
            should_retry = attempt_number < max_attempts and _is_retryable_exception(exc)
            if not should_retry:
                raise

            delay_seconds = _compute_delay_seconds(attempt_number)
            print(
                f"{description} failed on attempt {attempt_number}/{max_attempts}: {exc}. "
                f"Retrying in {delay_seconds:.1f}s..."
            )
            await asyncio.sleep(delay_seconds)
            attempt_number += 1