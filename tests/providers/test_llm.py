import http.client
import io
import os
import tempfile
import unittest
import urllib.error
import urllib.request
from unittest import mock

from rag.infra.resilience import CircuitOpenError, PermanentError, TransientError, deadline_budget
from rag.providers.llm import (
    BREAKER_FAILURES,
    RETRY_ATTEMPTS,
    _NoRedirect,
    load_env,
    make_anthropic_chat,
    make_api_embedder,
    post_json,
)


def http_error(code: int, headers: dict[str, str] | None = None) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("http://x", code, "msg", headers or {}, io.BytesIO(b"body"))  # type: ignore[arg-type]


class LlmTests(unittest.TestCase):
    def test_api_timeout_is_bounded_by_the_question_budget(self) -> None:
        with mock.patch("rag.infra.resilience.time.monotonic", return_value=100.0), deadline_budget(1):
            with mock.patch("rag.providers.llm._OPENER.open") as opener:
                opener.return_value.__enter__.return_value.read.return_value = b"{}"
                post_json("https://trusted.example", {}, {}, timeout=60)
        self.assertEqual(opener.call_args.kwargs["timeout"], 1.0)

    def test_api_requests_do_not_follow_redirects_with_credentials(self) -> None:
        request = urllib.request.Request("https://trusted.example", headers={"Authorization": "Bearer secret"})
        handler = _NoRedirect()
        for code in (301, 302, 303, 307, 308):
            with self.subTest(code=code):
                self.assertIsNone(handler.redirect_request(request, None, code, "redirect", {}, "https://other.example"))

    def test_load_env_reads_pairs_skips_comments_and_keeps_existing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, ".env")
            with open(path, "w", encoding="utf-8") as f:
                f.write("# comment\nTEST_KEY_A=abc\nTEST_KEY_B='quoted'\n\nTEST_KEY_C=fromfile\n")
            with mock.patch.dict(os.environ, {"TEST_KEY_C": "already"}, clear=False):
                os.environ.pop("TEST_KEY_A", None)
                os.environ.pop("TEST_KEY_B", None)
                load_env(path)
                self.assertEqual(os.environ["TEST_KEY_A"], "abc")
                self.assertEqual(os.environ["TEST_KEY_B"], "quoted")
                self.assertEqual(os.environ["TEST_KEY_C"], "already")
                os.environ.pop("TEST_KEY_A", None)
                os.environ.pop("TEST_KEY_B", None)

    def test_load_env_missing_file_is_fine(self) -> None:
        load_env("/nonexistent/.env")

    def test_rate_limit_and_server_errors_are_transient(self) -> None:
        for code in (429, 500, 503):
            with mock.patch("rag.providers.llm._OPENER.open", side_effect=http_error(code)):
                with self.assertRaises(TransientError):
                    post_json("http://x", {}, {})

    def test_client_errors_are_permanent(self) -> None:
        with mock.patch("rag.providers.llm._OPENER.open", side_effect=http_error(400)):
            with self.assertRaises(PermanentError):
                post_json("http://x", {}, {})

    def test_retry_after_header_is_passed_on(self) -> None:
        with mock.patch("rag.providers.llm._OPENER.open", side_effect=http_error(429, {"Retry-After": "5"})):
            with self.assertRaises(TransientError) as caught:
                post_json("http://x", {}, {})
        self.assertEqual(caught.exception.retry_after, 5.0)

    def test_unreadable_retry_after_header_is_ignored(self) -> None:
        with mock.patch("rag.providers.llm._OPENER.open", side_effect=http_error(429, {"Retry-After": "soon"})):
            with self.assertRaises(TransientError) as caught:
                post_json("http://x", {}, {})
        self.assertIsNone(caught.exception.retry_after)

    def test_odd_retry_after_values_become_none(self) -> None:
        for value in ("inf", "-1", "nan", "Wed, 21 Oct 2026 07:28:00 GMT", ""):
            with self.subTest(value=value):
                with mock.patch("rag.providers.llm._OPENER.open", side_effect=http_error(429, {"Retry-After": value})):
                    with self.assertRaises(TransientError) as caught:
                        post_json("http://x", {}, {})
                self.assertIsNone(caught.exception.retry_after)

    def test_zero_retry_after_is_kept_as_zero(self) -> None:
        with mock.patch("rag.providers.llm._OPENER.open", side_effect=http_error(429, {"Retry-After": "0"})):
            with self.assertRaises(TransientError) as caught:
                post_json("http://x", {}, {})
        self.assertEqual(caught.exception.retry_after, 0.0)

    def test_a_malformed_url_is_a_configuration_bug_not_a_network_blip(self) -> None:
        with mock.patch("rag.providers.llm._OPENER.open", side_effect=http.client.InvalidURL("nonnumeric port")):
            with self.assertRaises(PermanentError):
                post_json("http://x", {}, {})

    def test_embedder_waits_between_retries_using_the_sleep_it_was_given(self) -> None:
        waits: list[float] = []
        embed = make_api_embedder("key", sleep=waits.append)
        replies = [TransientError("busy", retry_after=2.0), {"data": [{"index": 0, "embedding": [3.0, 4.0]}]}]
        with mock.patch("rag.providers.llm.post_json", side_effect=replies):
            vectors = embed(["hello"])
        self.assertEqual(len(waits), 1)
        self.assertGreaterEqual(waits[0], 2.0)
        self.assertEqual(vectors, [[0.6, 0.8]])  # scaled to unit length

    def test_cut_off_response_is_transient(self) -> None:
        with mock.patch("rag.providers.llm._OPENER.open", side_effect=http.client.IncompleteRead(b"")):
            with self.assertRaises(TransientError):
                post_json("http://x", {}, {})

    def test_breaker_needs_more_failures_than_one_retry_sequence_makes(self) -> None:
        self.assertGreater(BREAKER_FAILURES, RETRY_ATTEMPTS)

    def test_one_exhausted_retry_sequence_does_not_open_the_breaker(self) -> None:
        chat = make_anthropic_chat("key", sleep=lambda seconds: None)
        with mock.patch("rag.providers.llm.post_json", side_effect=TransientError("down")) as post:
            with self.assertRaises(TransientError):
                chat("system", [], [])
            self.assertEqual(post.call_count, RETRY_ATTEMPTS)
            # Second question: attempts 4 and 5 still reach the API (the breaker opens on failure 5),
            # then attempt 6 is refused by the open breaker without calling the API.
            with self.assertRaises(CircuitOpenError):
                chat("system", [], [])
            self.assertEqual(post.call_count, BREAKER_FAILURES)

    def test_network_failure_is_transient(self) -> None:
        with mock.patch("rag.providers.llm._OPENER.open", side_effect=urllib.error.URLError("offline")):
            with self.assertRaises(TransientError):
                post_json("http://x", {}, {})


if __name__ == "__main__":
    unittest.main()
