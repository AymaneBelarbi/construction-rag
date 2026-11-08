"""OBSERVABILITY: measure what each question costs and where the time goes.

Two parts:
- Events: one JSON line per thing that happened (an LLM call, a tool call, a retry, a search,
  a finished question). Lines go to the standard `logging` system, so the operator decides
  where they land (stderr, a file, a log shipper). JSON means they can be searched and graphed.
- RunStats: running totals for ONE question (calls, tokens, seconds, estimated cost).

Privacy rule: events carry numbers and names, never the question, the retrieved records or the
model's answer. Those can hold business data, and logs are usually kept longer and shared wider.
"""
import json
import logging
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

# Same shape as llm.Chat. Defined here again because llm.py imports resilience.py, which imports
# this file; importing llm.py from here would make a circle.
Chat = Callable[[str, list[dict], list[dict]], dict]

EVENT_LOGGER = "rag.events"
_logger = logging.getLogger(EVENT_LOGGER)
# A library should not print log lines on its own. The program (main.py) decides where logs go.
_logger.addHandler(logging.NullHandler())

# USD per million tokens (input, output). Anthropic's published rates, checked 2026-10-06.
# Haiku 5.5 costs more for prompts over 100K tokens; that is not modelled because our prompts are
# a few thousand tokens. Update this table when prices change. Unknown models report no cost
# instead of a wrong one.
PRICE_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-opus-5-5": (4.00, 20.00),
}


# The request id of the question being answered right now. A ContextVar is a variable whose value
# belongs to the current thread/task, so events from retries, searches and tool calls get the same
# id without every function having to pass it along.
_current_request_id: ContextVar[str | None] = ContextVar("rag_request_id", default=None)


@contextmanager
def request_context(request_id: str | None) -> Iterator[None]:
    """Inside this block, every event gets request_id unless it sets its own."""
    token = _current_request_id.set(request_id)
    try:
        yield
    finally:
        _current_request_id.reset(token)  # restore whatever was set before, even after an error


def log_event(event: str, *, level: int = logging.INFO, **fields: Any) -> None:
    """Write one JSON line. `event` and `level` are reserved names, so they cannot be field names.
    default=str keeps odd values (sets, paths) from crashing the logger."""
    if _logger.isEnabledFor(level):  # skip the JSON work when nobody is listening
        request_id = _current_request_id.get()
        if request_id is not None:
            fields.setdefault("request_id", request_id)
        _logger.log(level, json.dumps({"event": event, **fields}, default=str, sort_keys=True))


@dataclass
class RunStats:
    """Totals for ONE question. Not thread-safe: the agent loop runs one step at a time. If tools
    ever run in parallel, add a lock around the += updates.

    llm_calls counts calls to the chat function, not HTTP attempts: a call that needed two retries
    counts once, and its seconds include the pauses between attempts (the retry events show them)."""

    model: str = ""
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])  # ties one question's events together
    llm_calls: int = 0
    llm_errors: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    llm_seconds: float = 0.0
    call_seconds: list[float] = field(default_factory=list)  # one entry per successful LLM call
    tool_calls: int = 0
    tool_errors: int = 0
    tool_seconds: float = 0.0
    wall_seconds: float = 0.0  # set by run_agent: the whole question, start to finish

    def add_llm_call(self, seconds: float, usage: dict[str, Any] | None) -> None:
        usage = usage or {}
        self.llm_calls += 1
        self.llm_seconds += seconds
        self.call_seconds.append(seconds)
        self.input_tokens += usage.get("input_tokens") or 0
        self.output_tokens += usage.get("output_tokens") or 0
        self.cache_read_tokens += usage.get("cache_read_input_tokens") or 0
        self.cache_write_tokens += usage.get("cache_creation_input_tokens") or 0

    def add_llm_error(self, seconds: float) -> None:
        self.llm_errors += 1
        self.llm_seconds += seconds

    def add_tool_call(self, seconds: float, is_error: bool) -> None:
        self.tool_calls += 1
        self.tool_seconds += seconds
        if is_error:
            self.tool_errors += 1

    def cost_usd(self) -> float | None:
        """Estimated from token counts, or None when it cannot be estimated honestly: an unknown
        model, or any cached tokens (they are billed at other rates that the table does not hold,
        and `input_tokens` leaves them out, so the sum would be too low). We do not use caching."""
        if self.model not in PRICE_PER_MTOK or self.cache_read_tokens or self.cache_write_tokens:
            return None
        price_in, price_out = PRICE_PER_MTOK[self.model]
        return (self.input_tokens * price_in + self.output_tokens * price_out) / 1_000_000

    def to_dict(self) -> dict[str, Any]:
        cost = self.cost_usd()
        return {
            "request_id": self.request_id,
            "model": self.model,
            "llm_calls": self.llm_calls,
            "llm_errors": self.llm_errors,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "llm_seconds": round(self.llm_seconds, 6),
            "tool_calls": self.tool_calls,
            "tool_errors": self.tool_errors,
            "tool_seconds": round(self.tool_seconds, 6),
            "wall_seconds": round(self.wall_seconds, 6),
            "cost_usd": None if cost is None else round(cost, 6),
        }


def format_summary(stats: RunStats) -> str:
    """One human-readable line for the terminal."""
    cost = stats.cost_usd()
    cost_text = "cost n/a" if cost is None else f"${cost:.4f}"
    return (
        f"[request {stats.request_id}: {stats.llm_calls} LLM calls, {stats.tool_calls} tool calls, "
        f"{stats.input_tokens:,} in / {stats.output_tokens:,} out tokens, {cost_text}, "
        f"{stats.wall_seconds:.1f}s]"
    )


def metered_chat(chat: Chat, stats: RunStats, clock: Callable[[], float] = time.perf_counter) -> Chat:
    """Wrap a chat function so every call is timed, counted and logged. The response passes through
    untouched. clock is a parameter so tests can supply fake times."""

    def metered(system: str, messages: list[dict], tools: list[dict]) -> dict:
        started = clock()
        try:
            response = chat(system, messages, tools)
        except Exception:
            seconds = clock() - started
            stats.add_llm_error(seconds)
            log_event("llm_call", level=logging.WARNING, request_id=stats.request_id, model=stats.model,
                      ok=False, seconds=round(seconds, 6))
            raise
        seconds = clock() - started
        usage = response.get("usage")
        stats.add_llm_call(seconds, usage)
        usage = usage or {}
        log_event("llm_call", request_id=stats.request_id, model=stats.model, ok=True,
                  seconds=round(seconds, 6), input_tokens=usage.get("input_tokens") or 0,
                  output_tokens=usage.get("output_tokens") or 0, stop_reason=response.get("stop_reason"))
        return response

    return metered
