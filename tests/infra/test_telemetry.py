import json
import logging
import unittest

from rag.infra.resilience import TransientError
from rag.infra.telemetry import EVENT_LOGGER, RunStats, format_summary, log_event, metered_chat, request_context
from tests.support.fakes import ScriptedChat, text_response


def events_from(records: list[logging.LogRecord]) -> list[dict]:
    return [json.loads(record.getMessage()) for record in records]


class RunStatsTests(unittest.TestCase):
    def test_records_usage_time_and_cost(self) -> None:
        stats = RunStats(model="claude-sonnet-5-5")
        stats.add_llm_call(2.0, {"input_tokens": 1000, "output_tokens": 200})
        stats.add_llm_call(1.0, {"input_tokens": 500, "output_tokens": 100})
        self.assertEqual(stats.llm_calls, 2)
        self.assertEqual((stats.input_tokens, stats.output_tokens), (1500, 300))
        self.assertEqual(stats.llm_seconds, 3.0)
        self.assertEqual(stats.call_seconds, [2.0, 1.0])
        cost = stats.cost_usd()
        assert cost is not None
        self.assertAlmostEqual(cost, (1500 * 2.00 + 300 * 10.00) / 1_000_000)

    def test_cost_is_unknown_once_cache_tokens_appear_instead_of_being_too_low(self) -> None:
        # Cached tokens are billed at other rates that the price table does not hold.
        for key in ("cache_read_input_tokens", "cache_creation_input_tokens"):
            with self.subTest(key=key):
                stats = RunStats(model="claude-sonnet-5-5")
                stats.add_llm_call(1.0, {"input_tokens": 100, "output_tokens": 10, key: 50})
                self.assertIsNone(stats.cost_usd())

    def test_unknown_model_has_no_cost_instead_of_a_wrong_one(self) -> None:
        stats = RunStats(model="some-future-model")
        stats.add_llm_call(1.0, {"input_tokens": 10, "output_tokens": 10})
        self.assertIsNone(stats.cost_usd())

    def test_missing_or_null_usage_counts_as_zero(self) -> None:
        stats = RunStats(model="claude-haiku-5-5")
        stats.add_llm_call(1.0, None)
        stats.add_llm_call(1.0, {"input_tokens": None})
        self.assertEqual((stats.input_tokens, stats.output_tokens), (0, 0))
        self.assertEqual(stats.llm_calls, 2)

    def test_tool_calls_are_counted_with_errors_separately(self) -> None:
        stats = RunStats()
        stats.add_tool_call(0.5, is_error=False)
        stats.add_tool_call(0.25, is_error=True)
        self.assertEqual((stats.tool_calls, stats.tool_errors, stats.tool_seconds), (2, 1, 0.75))

    def test_summary_is_json_serializable_and_has_a_request_id(self) -> None:
        summary = RunStats(model="claude-sonnet-5-5").to_dict()
        json.dumps(summary)
        self.assertEqual(len(summary["request_id"]), 12)

    def test_each_run_gets_its_own_request_id(self) -> None:
        self.assertNotEqual(RunStats().request_id, RunStats().request_id)

    def test_summary_line_shows_calls_tokens_cost_and_time(self) -> None:
        stats = RunStats(model="claude-sonnet-5-5")
        stats.add_llm_call(1.0, {"input_tokens": 6800, "output_tokens": 650})
        stats.wall_seconds = 7.44
        line = format_summary(stats)
        for expected in (stats.request_id, "1 LLM calls", "6,800 in", "650 out", "$0.0201", "7.4s"):
            self.assertIn(expected, line)
        self.assertIn("cost n/a", format_summary(RunStats(model="unknown")))


class MeteredChatTests(unittest.TestCase):
    def test_times_the_call_and_records_usage_without_changing_the_response(self) -> None:
        response = {**text_response("hi"), "usage": {"input_tokens": 11, "output_tokens": 7}}
        ticks = iter([10.0, 12.5])
        stats = RunStats(model="claude-sonnet-5-5")
        chat = metered_chat(ScriptedChat([response]), stats, clock=lambda: next(ticks))
        self.assertIs(chat("system", [], []), response)
        self.assertEqual(stats.call_seconds, [2.5])
        self.assertEqual((stats.input_tokens, stats.output_tokens), (11, 7))

    def test_failed_call_is_counted_timed_and_still_raises(self) -> None:
        def failing(system: str, messages: list[dict], tools: list[dict]) -> dict:
            raise TransientError("down")

        ticks = iter([0.0, 4.0])
        stats = RunStats()
        chat = metered_chat(failing, stats, clock=lambda: next(ticks))
        with self.assertLogs(EVENT_LOGGER, level="WARNING"), self.assertRaises(TransientError):
            chat("system", [], [])
        self.assertEqual(stats.llm_errors, 1)
        self.assertEqual(stats.llm_seconds, 4.0)

    def test_emits_an_llm_call_event_with_numbers_but_no_message_content(self) -> None:
        response = {**text_response("TOP-SECRET-ANSWER"), "usage": {"input_tokens": 5, "output_tokens": 2}}
        stats = RunStats(model="claude-sonnet-5-5")
        chat = metered_chat(ScriptedChat([response]), stats)
        with self.assertLogs(EVENT_LOGGER, level="INFO") as logs:
            chat("SECRET-SYSTEM", [{"role": "user", "content": "SECRET-QUESTION"}], [])
        (event,) = events_from(logs.records)
        self.assertEqual(event["event"], "llm_call")
        self.assertEqual(event["input_tokens"], 5)
        self.assertEqual(event["request_id"], stats.request_id)
        joined = " ".join(r.getMessage() for r in logs.records)
        for secret in ("TOP-SECRET-ANSWER", "SECRET-SYSTEM", "SECRET-QUESTION"):
            self.assertNotIn(secret, joined)


class LogEventTests(unittest.TestCase):
    def test_writes_one_json_object_per_event(self) -> None:
        with self.assertLogs(EVENT_LOGGER, level="INFO") as logs:
            log_event("something", count=3, tags={"b", "a"})  # a set is not JSON; it must not crash
        (event,) = events_from(logs.records)
        self.assertEqual(event["event"], "something")
        self.assertEqual(event["count"], 3)

    def test_level_can_be_chosen(self) -> None:
        with self.assertLogs(EVENT_LOGGER, level="WARNING") as logs:
            log_event("trouble", level=logging.WARNING)
        self.assertEqual(logs.records[0].levelname, "WARNING")

    def test_events_inside_a_request_context_carry_its_request_id(self) -> None:
        with self.assertLogs(EVENT_LOGGER, level="INFO") as logs:
            log_event("outside")
            with request_context("abc123"):
                log_event("inside")
                log_event("explicit", request_id="other")
            log_event("after")
        by_name = {e["event"]: e for e in events_from(logs.records)}
        self.assertNotIn("request_id", by_name["outside"])
        self.assertEqual(by_name["inside"]["request_id"], "abc123")
        self.assertEqual(by_name["explicit"]["request_id"], "other")  # an explicit value wins
        self.assertNotIn("request_id", by_name["after"])  # the context is restored on exit


if __name__ == "__main__":
    unittest.main()
