import json
import unittest
from unittest import mock

from rag.infra.resilience import CircuitBreaker, CircuitOpenError, DeadlineExceeded, PermanentError, TransientError, deadline_budget, retry
from rag.infra.telemetry import EVENT_LOGGER


class RetryTests(unittest.TestCase):
    def test_retry_does_not_sleep_past_the_remaining_budget(self) -> None:
        attempts: list[int] = []
        delays: list[float] = []

        @retry(max_attempts=3, base_delay=5.0, jitter=0.0, sleep=delays.append)
        def down() -> None:
            attempts.append(1)
            raise TransientError("busy")

        with mock.patch("rag.infra.resilience.time.monotonic", return_value=100.0), deadline_budget(1):
            with self.assertRaises(DeadlineExceeded):
                down()
        self.assertEqual(len(attempts), 1)
        self.assertEqual(delays, [])

    def test_succeeds_after_transient_failures_with_growing_delays(self) -> None:
        delays: list[float] = []
        attempts: list[int] = []

        @retry(max_attempts=3, base_delay=0.5, jitter=0.0, sleep=delays.append)
        def flaky() -> str:
            attempts.append(1)
            if len(attempts) < 3:
                raise TransientError("try again")
            return "ok"

        self.assertEqual(flaky(), "ok")
        self.assertEqual(delays, [0.5, 1.0])

    def test_gives_up_after_max_attempts_and_raises_last_error(self) -> None:
        attempts: list[int] = []

        @retry(max_attempts=3, sleep=lambda s: None)
        def always_fails() -> None:
            attempts.append(1)
            raise TransientError("down")

        with self.assertRaises(TransientError):
            always_fails()
        self.assertEqual(len(attempts), 3)

    def test_permanent_error_is_not_retried(self) -> None:
        attempts: list[int] = []

        @retry(max_attempts=3, sleep=lambda s: None)
        def bad_request() -> None:
            attempts.append(1)
            raise PermanentError("400")

        with self.assertRaises(PermanentError):
            bad_request()
        self.assertEqual(len(attempts), 1)

    def test_waits_at_least_as_long_as_the_servers_retry_after(self) -> None:
        delays: list[float] = []
        attempts: list[int] = []

        @retry(max_attempts=2, base_delay=0.5, jitter=0.0, sleep=delays.append)
        def rate_limited() -> str:
            attempts.append(1)
            if len(attempts) < 2:
                raise TransientError("429", retry_after=7.0)
            return "ok"

        self.assertEqual(rate_limited(), "ok")
        self.assertEqual(delays, [7.0])

    def test_gives_up_at_once_when_the_server_asks_for_more_than_max_delay(self) -> None:
        delays: list[float] = []
        attempts: list[int] = []

        @retry(max_attempts=3, jitter=0.0, max_delay=30.0, sleep=delays.append)
        def quota_exhausted() -> None:
            attempts.append(1)
            raise TransientError("429", retry_after=3600.0)

        with self.assertLogs(EVENT_LOGGER, level="WARNING") as logs:
            with self.assertRaises(TransientError):
                quota_exhausted()
        self.assertEqual(len(attempts), 1)  # retrying early would fail again and use up the breaker
        self.assertEqual(delays, [])
        self.assertEqual(json.loads(logs.records[0].getMessage())["event"], "retry_abandoned")

    def test_backoff_growth_is_capped_by_max_delay(self) -> None:
        delays: list[float] = []

        @retry(max_attempts=4, base_delay=10.0, jitter=0.0, max_delay=15.0, sleep=delays.append)
        def always_fails() -> None:
            raise TransientError("down")

        with self.assertRaises(TransientError):
            always_fails()
        self.assertEqual(delays, [10.0, 15.0, 15.0])

    def test_jitter_is_added_even_when_the_server_chose_the_wait(self) -> None:
        delays: list[float] = []
        attempts: list[int] = []

        @retry(max_attempts=2, jitter=0.2, sleep=delays.append)
        def rate_limited() -> str:
            attempts.append(1)
            if len(attempts) < 2:
                raise TransientError("429", retry_after=7.0)
            return "ok"

        with mock.patch("rag.infra.resilience.random.uniform", return_value=0.1):
            rate_limited()
        self.assertAlmostEqual(delays[0], 7.7)  # never less than 7, spread out so clients do not retry in lockstep

    def test_every_retry_is_logged_with_attempt_delay_and_error_type(self) -> None:
        attempts: list[int] = []

        @retry(max_attempts=3, base_delay=0.5, jitter=0.0, sleep=lambda s: None)
        def flaky() -> str:
            attempts.append(1)
            if len(attempts) < 3:
                raise TransientError("503")
            return "ok"

        with self.assertLogs(EVENT_LOGGER, level="WARNING") as logs:
            flaky()
        events = [json.loads(r.getMessage()) for r in logs.records]
        self.assertEqual([e["event"] for e in events], ["retry", "retry"])
        self.assertEqual([e["attempt"] for e in events], [1, 2])
        self.assertEqual([e["delay_seconds"] for e in events], [0.5, 1.0])
        self.assertEqual(events[0]["error_type"], "TransientError")

    def test_transient_error_without_retry_after_has_none(self) -> None:
        self.assertIsNone(TransientError("down").retry_after)


class CircuitBreakerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = 0.0
        self.breaker = CircuitBreaker(failure_threshold=2, reset_after=30.0, clock=lambda: self.now)

    @staticmethod
    def down() -> None:
        raise TransientError("down")

    def test_opens_after_threshold_and_stops_calling(self) -> None:
        for _ in range(2):
            with self.assertRaises(TransientError):
                self.breaker.call(self.down)
        calls: list[int] = []
        with self.assertRaises(CircuitOpenError):
            self.breaker.call(lambda: calls.append(1))
        self.assertEqual(calls, [])

    def test_opening_the_breaker_is_logged(self) -> None:
        with self.assertLogs(EVENT_LOGGER, level="WARNING") as logs:
            for _ in range(2):
                with self.assertRaises(TransientError):
                    self.breaker.call(self.down)
        events = [json.loads(r.getMessage()) for r in logs.records]
        self.assertEqual([e["event"] for e in events], ["circuit_opened"])
        self.assertEqual(events[0]["failures"], 2)

    def test_half_open_after_timeout_and_success_closes(self) -> None:
        for _ in range(2):
            with self.assertRaises(TransientError):
                self.breaker.call(self.down)
        self.now = 31.0
        self.assertEqual(self.breaker.call(lambda: "ok"), "ok")
        self.assertEqual(self.breaker.state, "closed")

    def test_failure_in_half_open_reopens(self) -> None:
        for _ in range(2):
            with self.assertRaises(TransientError):
                self.breaker.call(self.down)
        self.now = 31.0
        with self.assertRaises(TransientError):
            self.breaker.call(self.down)
        self.assertEqual(self.breaker.state, "open")

    def test_success_resets_failure_count(self) -> None:
        with self.assertRaises(TransientError):
            self.breaker.call(self.down)
        self.breaker.call(lambda: "ok")
        with self.assertRaises(TransientError):
            self.breaker.call(self.down)
        self.assertEqual(self.breaker.state, "closed")


if __name__ == "__main__":
    unittest.main()
