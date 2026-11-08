"""RESILIENCE: wrap every LLM or embedding call so a bad moment does not become a crash.

Two tools:
- retry: try again after a pause that doubles each time (exponential backoff).
  Only for errors that may go away on their own (TransientError).
- CircuitBreaker: after several failures in a row, stop calling for a while.
  This protects the service from a flood of requests and fails fast for us.
"""
import functools
import logging
import random
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from rag.infra.telemetry import log_event


class TransientError(Exception):
    """Might work if tried again: network trouble, rate limit (429), server error (5xx).
    retry_after is how many seconds the server asked us to wait, when it said (None otherwise)."""

    def __init__(self, message: str = "", retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class PermanentError(Exception):
    """Will not work if tried again: bad request, bad key. Never retried."""


class CircuitOpenError(Exception):
    """The breaker is open: the call was not even attempted."""


class DeadlineExceeded(Exception):
    """The question's remaining time budget is exhausted."""


_deadline: ContextVar[float | None] = ContextVar("rag_deadline", default=None)


@contextmanager
def deadline_budget(seconds: float) -> Iterator[None]:
    deadline = time.monotonic() + max(0.0, seconds)
    parent = _deadline.get()
    token = _deadline.set(min(deadline, parent) if parent is not None else deadline)
    try:
        yield
    finally:
        _deadline.reset(token)


def remaining_timeout(limit: float) -> float:
    deadline = _deadline.get()
    if deadline is None:
        return limit
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise DeadlineExceeded("Question time budget exhausted")
    return min(limit, remaining)


def retry(
    max_attempts: int = 3,
    base_delay: float = 0.5,
    jitter: float = 0.2,
    max_delay: float = 30.0,
    retry_on: tuple[type[Exception], ...] = (TransientError,),
    sleep: Callable[[float], None] = time.sleep,
) -> Callable:
    """A decorator is a function that takes a function and returns a wrapped
    version of it. Use it as @retry(...) on the line above a def."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for attempt in range(max_attempts):
                remaining_timeout(float("inf"))
                try:
                    return func(*args, **kwargs)
                except retry_on as error:
                    if attempt == max_attempts - 1:
                        raise
                    server_wait = getattr(error, "retry_after", None) or 0.0
                    if server_wait > max_delay:
                        # The server asked for longer than we are willing to wait (for example an
                        # exhausted quota: "come back in an hour"). Retrying sooner would only fail
                        # again, so stop here and let the caller deal with the error.
                        log_event("retry_abandoned", level=logging.WARNING, attempt=attempt + 1,
                                  error_type=type(error).__name__, server_retry_after=server_wait,
                                  max_delay_seconds=max_delay)
                        raise
                    delay = base_delay * (2**attempt)  # 0.5, 1.0, 2.0 ...
                    delay = max(delay, server_wait)  # never come back sooner than the server asked
                    # jitter spreads out clients that were all told to wait the same time
                    delay *= 1 + random.uniform(0, jitter)
                    delay = min(delay, max_delay)  # caps the growth of the backoff itself
                    if remaining_timeout(float("inf")) <= delay:
                        raise DeadlineExceeded("Not enough question time remains for another retry") from error
                    log_event("retry", level=logging.WARNING, attempt=attempt + 1, max_attempts=max_attempts,
                              delay_seconds=round(delay, 3), error_type=type(error).__name__,
                              server_retry_after=server_wait or None)
                    sleep(delay)

        return wrapper

    return decorator


class CircuitBreaker:
    """closed: calls go through.  open: calls are refused.
    half_open: after reset_after seconds one trial call is allowed."""

    def __init__(
        self,
        failure_threshold: int = 3,
        reset_after: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.reset_after = reset_after
        self.clock = clock
        self.failures = 0
        self.opened_at: float | None = None

    @property
    def state(self) -> str:
        if self.opened_at is None:
            return "closed"
        if self.clock() - self.opened_at >= self.reset_after:
            return "half_open"
        return "open"

    def call(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        if self.state == "open":
            raise CircuitOpenError("Circuit is open; skipping the call")
        try:
            result = func(*args, **kwargs)
        except TransientError:
            # PermanentError is the caller's bug, not an outage, so it does not trip the breaker
            self.failures += 1
            if self.state == "half_open" or self.failures >= self.failure_threshold:
                self.opened_at = self.clock()
                log_event("circuit_opened", level=logging.WARNING, failures=self.failures,
                          reset_after_seconds=self.reset_after)
            raise
        self.failures = 0
        self.opened_at = None
        return result
