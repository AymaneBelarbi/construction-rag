"""LANGSMITH: send each question to LangSmith as one trace: the question, its model calls, its tool calls.

Off unless LANGSMITH_API_KEY is set. Runs go to LangSmith's batch endpoint with urllib, like the model
providers, so there is still no SDK to install.

Privacy follows telemetry.py: by default a run carries names, timings, token counts and outcome flags,
never the question, the records or the model's text. LANGSMITH_INCLUDE_CONTENT=true adds them, for a
private project on test data.

The runs of one trace are kept until the question ends, then sent in one request. A failed upload is
logged as an event and never changes the answer.
"""
import json
import logging
import os
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from rag.infra.telemetry import log_event

DEFAULT_ENDPOINT = "https://api.smith.langchain.com"
DEFAULT_PROJECT = "construction-rag"
UPLOAD_TIMEOUT_SECONDS = 10.0

Send = Callable[[str, dict[str, str], bytes], None]


@dataclass(frozen=True)
class LangSmithConfig:
    api_key: str
    project: str = DEFAULT_PROJECT
    endpoint: str = DEFAULT_ENDPOINT
    include_content: bool = False

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> LangSmithConfig | None:
        api_key = env.get("LANGSMITH_API_KEY", "").strip()
        if not api_key:
            return None
        return cls(
            api_key=api_key,
            project=env.get("LANGSMITH_PROJECT", "").strip() or DEFAULT_PROJECT,
            endpoint=(env.get("LANGSMITH_ENDPOINT", "").strip() or DEFAULT_ENDPOINT).rstrip("/"),
            include_content=env.get("LANGSMITH_INCLUDE_CONTENT", "").strip().lower() in ("1", "true", "yes"),
        )


@dataclass
class Run:
    """One step of a trace. Callers add to `outputs` (always sent) and `content` (sent only with
    include_content) while the step runs; `metadata` ends up under extra.metadata."""

    name: str
    run_type: str
    inputs: dict[str, Any]
    trace_id: uuid.UUID
    parent: Run | None
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    end_time: datetime | None = None
    outputs: dict[str, Any] = field(default_factory=dict)
    content: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def dotted_order(self) -> str:
        own = f"{self.start_time:%Y%m%dT%H%M%S%fZ}{self.id}"
        return own if self.parent is None else f"{self.parent.dotted_order}.{own}"

    def payload(self, project: str, include_content: bool, input_content: dict[str, Any]) -> dict[str, Any]:
        outputs = {**self.outputs, **self.content} if include_content else dict(self.outputs)
        return {
            "id": str(self.id),
            "trace_id": str(self.trace_id),
            "parent_run_id": None if self.parent is None else str(self.parent.id),
            "dotted_order": self.dotted_order,
            "name": self.name,
            "run_type": self.run_type,
            "inputs": {**self.inputs, **input_content} if include_content else dict(self.inputs),
            "outputs": outputs,
            "error": self.error,
            "start_time": self.start_time.isoformat(),
            "end_time": (self.end_time or datetime.now(UTC)).isoformat(),
            "session_name": project,
            "extra": {"metadata": self.metadata},
        }


def _post(url: str, headers: dict[str, str], body: bytes) -> None:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=UPLOAD_TIMEOUT_SECONDS):
        pass


class Tracer:
    def __init__(self, config: LangSmithConfig, send: Send = _post) -> None:
        self.config = config
        self._send = send
        self._finished: dict[uuid.UUID, list[tuple[Run, dict[str, Any]]]] = {}

    def finish(self, run: Run, input_content: dict[str, Any]) -> None:
        self._finished.setdefault(run.trace_id, []).append((run, input_content))
        if run.parent is None:
            self._upload(self._finished.pop(run.trace_id))

    def _upload(self, runs: list[tuple[Run, dict[str, Any]]]) -> None:
        body = json.dumps(
            {"post": [run.payload(self.config.project, self.config.include_content, content) for run, content in runs]},
            default=str,
        ).encode()
        headers = {"Content-Type": "application/json", "x-api-key": self.config.api_key}
        try:
            self._send(f"{self.config.endpoint}/runs/batch", headers, body)
        except (urllib.error.URLError, OSError, ValueError) as problem:
            log_event("langsmith_upload", level=logging.WARNING, ok=False, runs=len(runs), error_type=type(problem).__name__)
        else:
            log_event("langsmith_upload", level=logging.DEBUG, ok=True, runs=len(runs))


_UNSET = object()
_tracer: Tracer | None | object = _UNSET
_current: ContextVar[Run | None] = ContextVar("rag_langsmith_run", default=None)


def get_tracer() -> Tracer | None:
    """The tracer for this process, built from the environment on first use (after load_env)."""
    global _tracer
    if _tracer is _UNSET:
        config = LangSmithConfig.from_env()
        _tracer = None if config is None else Tracer(config)
    return _tracer if isinstance(_tracer, Tracer) else None


def set_tracer(tracer: Tracer | None) -> None:
    global _tracer
    _tracer = tracer


@contextmanager
def traced(
    name: str,
    run_type: str,
    inputs: dict[str, Any] | None = None,
    *,
    content: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Iterator[Run | None]:
    """Record the block as a run, nested under the run that is open. Yields None when tracing is off,
    so callers write `if run:` before adding outputs. `content` holds inputs that may carry project
    data; it is sent only with include_content. An exception marks the run with its type name only."""
    tracer = get_tracer()
    if tracer is None:
        yield None
        return
    parent = _current.get()
    run = Run(name=name, run_type=run_type, inputs=dict(inputs or {}), parent=parent,
              trace_id=uuid.uuid4() if parent is None else parent.trace_id, metadata=dict(metadata or {}))
    if parent is None:
        run.trace_id = run.id
    token = _current.set(run)
    try:
        yield run
    except BaseException as problem:
        run.error = type(problem).__name__
        raise
    finally:
        _current.reset(token)
        run.end_time = datetime.now(UTC)
        tracer.finish(run, dict(content or {}))


ChatFn = Callable[[str, list[dict], list[dict]], dict]


def traced_chat(chat: ChatFn, name: str, model: str = "") -> ChatFn:
    """Wrap a chat function so each call is an LLM run with its token usage. The response is untouched."""

    def call(system: str, messages: list[dict], tools: list[dict]) -> dict:
        metadata = {"ls_provider": "anthropic", **({"ls_model_name": model} if model else {})}
        with traced(name, "llm", {"messages": len(messages), "tools": len(tools)},
                    content={"system": system, "messages": messages}, metadata=metadata) as run:
            response = chat(system, messages, tools)
            if run:
                usage = response.get("usage") or {}
                input_tokens = usage.get("input_tokens") or 0
                output_tokens = usage.get("output_tokens") or 0
                run.outputs["stop_reason"] = response.get("stop_reason")
                run.metadata["usage_metadata"] = {"input_tokens": input_tokens, "output_tokens": output_tokens,
                                                  "total_tokens": input_tokens + output_tokens}
                run.content["content"] = response.get("content")
            return response

    return call
