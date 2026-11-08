"""All network code lives here, so every other file can be tested offline.

Uses only the standard library (urllib), so there is no SDK to install.
Each real call is wrapped with retry + circuit breaker from resilience.py.
API keys are read by main.py from the environment (.env), never written in code.
"""
import http.client
import json
import math
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rag.infra.resilience import CircuitBreaker, PermanentError, TransientError, remaining_timeout, retry
from rag.providers.embed import HASHING_ID, Embedder, hashing_embed

# A chat function takes (system prompt, messages, tool definitions) and returns the
# provider's response as a dict with "content" (a list of blocks) and "stop_reason".
Chat = Callable[[str, list[dict], list[dict]], dict]

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"

# Retry sits OUTSIDE the circuit breaker, so every failed attempt counts against the breaker.
# The breaker must therefore tolerate more failures than one retry sequence makes, otherwise a
# single question that fails 3 times would open it and fail every following call for 30 seconds.
#
# Worst case for ONE call that never succeeds: 3 attempts x 60 s timeout + 2 pauses of up to 30 s
# = about 4 minutes before the error reaches the user. Lower `timeout` if that is too long.
RETRY_ATTEMPTS = 3
BREAKER_FAILURES = 5
BREAKER_RESET_SECONDS = 30.0


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: urllib.request.Request, fp: Any, code: int, msg: str,
                         headers: Any, newurl: str) -> urllib.request.Request | None:
        # API credentials must never be sent to a redirect destination.
        return None


_OPENER = urllib.request.build_opener(_NoRedirect())


def load_env(path: str = ".env") -> None:
    """Copy KEY=value lines from a .env file into os.environ. Real environment wins."""
    file = Path(path)
    if not file.exists():
        return
    for line in file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _retry_after(error: urllib.error.HTTPError) -> float | None:
    """Seconds from a Retry-After header. The header can also hold a date, which we ignore."""
    value = error.headers.get("Retry-After") if error.headers else None
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None  # not a number, for example a date
    # float() accepts "nan" and "inf"; neither is a usable number of seconds.
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def post_json(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float = 30.0) -> dict:
    """POST JSON and sort failures into 'worth retrying' and 'not worth retrying'."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with _OPENER.open(request, timeout=remaining_timeout(timeout)) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:  # must come before URLError, which it extends
        body = error.read().decode("utf-8", errors="replace")[:300]
        error.close()  # release the connection
        message = f"HTTP {error.code} from {url}: {body}"
        if error.code == 429 or error.code >= 500:
            raise TransientError(message, retry_after=_retry_after(error)) from error
        raise PermanentError(message) from error
    except http.client.InvalidURL as error:  # our configuration is wrong; retrying cannot fix it
        raise PermanentError(f"Invalid URL {url!r}: {error}") from error
    except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException) as error:
        raise TransientError(f"Network problem calling {url}: {error}") from error
    except json.JSONDecodeError as error:
        raise PermanentError(f"{url} returned something that is not JSON") from error


def make_anthropic_chat(
    api_key: str,
    model: str = "claude-sonnet-5-5",
    max_tokens: int = 2048,
    timeout: float = 60.0,
    sleep: Callable[[float], None] = time.sleep,
) -> Chat:
    headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
    breaker = CircuitBreaker(failure_threshold=BREAKER_FAILURES, reset_after=BREAKER_RESET_SECONDS)

    # retry sits outside the breaker: an open breaker raises CircuitOpenError, which is not retried
    @retry(max_attempts=RETRY_ATTEMPTS, base_delay=1.0, sleep=sleep)
    def send(payload: dict) -> dict:
        return breaker.call(post_json, ANTHROPIC_URL, headers, payload, timeout)

    def chat(system: str, messages: list[dict], tools: list[dict]) -> dict:
        payload: dict[str, Any] = {
            "model": model, "max_tokens": max_tokens, "system": system, "messages": messages,
        }
        if tools:
            payload["tools"] = tools
        return send(payload)

    return chat


def make_api_embedder(
    api_key: str,
    model: str = "text-embedding-3-small",
    base_url: str = "https://api.openai.com/v1",
    sleep: Callable[[float], None] = time.sleep,
) -> Embedder:
    """Works with OpenAI and with any service that copies its /embeddings format."""
    headers = {"Authorization": f"Bearer {api_key}"}
    breaker = CircuitBreaker(failure_threshold=BREAKER_FAILURES, reset_after=BREAKER_RESET_SECONDS)

    @retry(max_attempts=RETRY_ATTEMPTS, base_delay=1.0, sleep=sleep)
    def send(payload: dict) -> dict:
        return breaker.call(post_json, f"{base_url}/embeddings", headers, payload, 60.0)

    def embed(texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 64):  # send in batches of 64
            data = send({"model": model, "input": texts[start : start + 64]})["data"]
            for item in sorted(data, key=lambda d: d["index"]):
                vector = item["embedding"]
                length = math.sqrt(sum(x * x for x in vector)) or 1.0
                vectors.append([x / length for x in vector])  # unit length, as retrieve.py expects
        return vectors

    return embed


def embedder_from_env() -> tuple[Embedder, str]:
    """The embedder to use and a name for it (stored in the cache, so a change of model rebuilds it).
    EMBED_API_KEY set: an OpenAI-compatible endpoint. Otherwise the offline hashing embedder."""
    api_key = os.environ.get("EMBED_API_KEY")
    if not api_key:
        return hashing_embed, HASHING_ID
    model = os.environ.get("EMBED_MODEL", "text-embedding-3-small")
    base_url = os.environ.get("EMBED_BASE_URL", "https://api.openai.com/v1")
    return make_api_embedder(api_key, model, base_url), f"api:{base_url}:{model}"
