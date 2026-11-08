import io
import json
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

from evals.run_eval import (
    EXIT_USAGE,
    git_state,
    judge,
    main,
    percentile,
    quality_failures,
    run_problems,
    slug,
    summarize,
    thresholds_for,
)
from tests.support.db_support import env_file_values, requires_database

THRESHOLDS = {"min_question_pass_rate": 0.9, "min_grounded_rate": 0.95, "max_mean_cost_usd": 0.05,
              "max_p95_wall_seconds": 30, "max_errors": 0}


def row(qid: str = "q", passed: bool = True, grounded: bool = True, wall: float = 5.0,
        cost: float | None = 0.01, error: str | None = None) -> dict:
    return {"id": qid, "passed": passed, "grounded": grounded, "error": error,
            "stats": {"wall_seconds": wall, "cost_usd": cost, "llm_calls": 3, "input_tokens": 6000, "output_tokens": 600}}


class PercentileTests(unittest.TestCase):
    def test_nearest_rank_percentiles(self) -> None:
        values = [4.0, 1.0, 3.0, 2.0]
        self.assertEqual(percentile(values, 0.5), 2.0)
        self.assertEqual(percentile(values, 0.95), 4.0)
        self.assertEqual(percentile(values, 0.25), 1.0)

    def test_ten_values_p95_is_the_largest_and_p50_the_fifth(self) -> None:
        values = [float(i) for i in range(1, 11)]
        self.assertEqual(percentile(values, 0.95), 10.0)
        self.assertEqual(percentile(values, 0.5), 5.0)

    def test_odd_length_median_is_the_middle_value(self) -> None:
        self.assertEqual(percentile([3.0, 1.0, 2.0], 0.5), 2.0)

    def test_p95_of_twenty_is_the_nineteenth_not_the_largest(self) -> None:
        self.assertEqual(percentile([float(i) for i in range(1, 21)], 0.95), 19.0)

    def test_a_single_value_is_every_percentile(self) -> None:
        self.assertEqual(percentile([7.0], 0.95), 7.0)


class SummarizeTests(unittest.TestCase):
    def test_rates_latency_and_mean_cost_use_completed_runs_only(self) -> None:
        rows = [row("a", True, True, 4.0, 0.01), row("a", False, True, 6.0, 0.03), row("a", True, False, 10.0, 0.02),
                row("b", error="TransientError: down", passed=False, grounded=False, wall=0.0, cost=0.0)]
        summary = summarize(rows)
        self.assertEqual((summary["runs"], summary["valid_runs"], summary["errors"]), (4, 3, 1))
        self.assertAlmostEqual(summary["run_pass_rate"], 2 / 3)
        self.assertAlmostEqual(summary["grounded_rate"], 2 / 3)
        self.assertEqual(summary["p50_wall_seconds"], 6.0)
        self.assertAlmostEqual(summary["mean_cost_usd"], 0.02)  # the failed run's $0 does not dilute the mean

    def test_a_question_passes_only_when_more_than_half_of_its_repeats_pass(self) -> None:
        rows = ([row("two_of_three", True), row("two_of_three", True), row("two_of_three", False)]
                + [row("one_of_three", True), row("one_of_three", False), row("one_of_three", False)]
                + [row("tie", True), row("tie", False)])
        summary = summarize(rows)
        self.assertEqual(summary["failing_questions"], ["one_of_three", "tie"])
        self.assertAlmostEqual(summary["question_pass_rate"], 1 / 3)

    def test_cost_is_unknown_if_any_completed_run_has_unknown_cost(self) -> None:
        self.assertIsNone(summarize([row(cost=0.01), row(cost=None)])["mean_cost_usd"])

    def test_no_completed_run_gives_a_summary_without_quality_numbers(self) -> None:
        summary = summarize([row(error="x", passed=False, grounded=False)])
        self.assertEqual(summary["valid_runs"], 0)
        self.assertNotIn("run_pass_rate", summary)


class RunValidityTests(unittest.TestCase):
    def test_errors_make_the_run_invalid_instead_of_looking_like_bad_quality(self) -> None:
        summary = summarize([row("a"), row("b", error="PermanentError: bad key", passed=False, grounded=False)])
        self.assertEqual(len(run_problems(summary, THRESHOLDS)), 1)

    def test_a_run_with_no_completed_question_is_invalid(self) -> None:
        summary = summarize([row(error="x", passed=False, grounded=False)])
        self.assertTrue(any("no run completed" in p for p in run_problems(summary, THRESHOLDS)))

    def test_a_clean_run_has_no_problems(self) -> None:
        self.assertEqual(run_problems(summarize([row(), row()]), THRESHOLDS), [])

    def test_a_small_error_allowance_can_be_configured(self) -> None:
        summary = summarize([row("a"), row("b", error="x", passed=False, grounded=False)])
        self.assertEqual(run_problems(summary, {**THRESHOLDS, "max_errors": 1}), [])


class QualityGateTests(unittest.TestCase):
    def test_a_healthy_summary_passes_every_gate(self) -> None:
        self.assertEqual(quality_failures(summarize([row() for _ in range(10)]), THRESHOLDS), [])

    def test_each_gate_reports_its_own_failure(self) -> None:
        failing = summarize([row("bad", passed=False)] * 3 + [row(f"ok{i}") for i in range(2)])
        self.assertIn("question pass rate", " ".join(quality_failures(failing, THRESHOLDS)))
        self.assertIn("bad", " ".join(quality_failures(failing, THRESHOLDS)))  # names the failing question
        ungrounded = summarize([row(grounded=False)] * 2 + [row() for _ in range(8)])
        self.assertIn("grounded rate", " ".join(quality_failures(ungrounded, THRESHOLDS)))
        pricey = summarize([row(cost=0.2) for _ in range(5)])
        self.assertIn("cost", " ".join(quality_failures(pricey, THRESHOLDS)))

    def test_latency_gate_looks_at_p95_not_the_median(self) -> None:
        # 20 runs, 19 fast and 1 very slow: p95 is the 19th value (fast), so this passes...
        fast_mostly = summarize([row(wall=5.0) for _ in range(19)] + [row(wall=99.0)])
        self.assertEqual(quality_failures(fast_mostly, THRESHOLDS), [])
        # ...but when 2 of 20 are slow, p95 is slow and the gate fails even though the median is fast.
        slow_tail = summarize([row(wall=5.0) for _ in range(18)] + [row(wall=99.0), row(wall=99.0)])
        self.assertEqual(slow_tail["p50_wall_seconds"], 5.0)
        self.assertIn("latency", " ".join(quality_failures(slow_tail, THRESHOLDS)))

    def test_a_value_exactly_at_a_threshold_passes(self) -> None:
        # 9 of 10 questions pass = exactly the 0.9 minimum; grounded exactly 0.95 is also allowed.
        rows = [row(f"q{i}") for i in range(9)] + [row("q9", passed=False)]
        self.assertEqual(quality_failures(summarize(rows), THRESHOLDS), [])
        borderline = summarize([row(grounded=False)] + [row() for _ in range(19)])
        self.assertEqual(quality_failures(borderline, THRESHOLDS), [])

    def test_unknown_cost_fails_the_cost_gate_instead_of_passing_silently(self) -> None:
        failures = quality_failures(summarize([row(cost=None) for _ in range(5)]), THRESHOLDS)
        self.assertEqual(len(failures), 1)
        self.assertIn("unknown", failures[0])

    def test_several_failures_are_all_reported(self) -> None:
        bad = summarize([row(passed=False, grounded=False, wall=99.0, cost=1.0) for _ in range(5)])
        self.assertEqual(len(quality_failures(bad, THRESHOLDS)), 4)


class HelperTests(unittest.TestCase):
    def test_git_state_reports_this_repo_and_degrades_to_unknown(self) -> None:
        self.assertRegex(git_state(), r"^[0-9a-f]{7,}(\+dirty)?$")
        self.assertEqual(git_state(Path("/nonexistent/folder")), "unknown")

    def test_slug_makes_a_model_name_safe_for_a_file_name(self) -> None:
        self.assertEqual(slug("claude-haiku-5-5"), "claude-haiku-5-5")
        self.assertEqual(slug("vendor/model:latest"), "vendor_model_latest")

    def test_bad_arguments_and_missing_key_exit_cleanly(self) -> None:
        with redirect_stderr(io.StringIO()):  # keep the usage message out of the test output
            with self.assertRaises(SystemExit) as caught:
                main(["--repeats", "0"])
            self.assertEqual(caught.exception.code, 2)
            with mock.patch("evals.run_eval.load_env"), mock.patch.dict("os.environ", {}, clear=True):
                self.assertEqual(main(["--repeats", "1"]), EXIT_USAGE)


class ClaimSupportGateTests(unittest.TestCase):
    GATED = {**THRESHOLDS, "min_claims_supported_rate": 0.95}

    def test_too_many_runs_with_unsupported_figures_fail_the_gate_and_name_it(self) -> None:
        rows = [row("a"), {**row("b"), "unsupported": ["$20,000 (cited CE-004)"]}] + [row(f"q{i}") for i in range(8)]
        failures = quality_failures(summarize(rows), self.GATED)  # 1 of 10 = 90% supported
        self.assertEqual(len(failures), 1)
        self.assertIn("claims supported", failures[0])

    def test_a_run_at_the_floor_passes_and_a_config_without_the_key_does_not_gate(self) -> None:
        rows = [{**row("b"), "unsupported": ["x"]}] + [row(f"q{i}") for i in range(19)]  # exactly 95%
        self.assertEqual(quality_failures(summarize(rows), self.GATED), [])
        bad = [{**row("b"), "unsupported": ["x"]}, row("a")]
        self.assertEqual(quality_failures(summarize(bad), THRESHOLDS), [])

    def test_the_shipped_thresholds_gate_it(self) -> None:
        config = json.loads((Path("evals") / "data" / "thresholds.json").read_text(encoding="utf-8"))
        self.assertEqual(thresholds_for(config, "golden")["min_claims_supported_rate"], 0.95)


class ClaimSupportTests(unittest.TestCase):
    def test_the_summary_reports_the_share_of_runs_whose_figures_were_all_supported(self) -> None:
        rows = [row("a"), {**row("b"), "unsupported": ["$20,000 (cited CE-004)"]}, {**row("c"), "unsupported": []}, row("d")]
        self.assertEqual(summarize(rows)["claims_supported_rate"], 0.75)

    def test_a_failed_run_is_not_counted_either_way(self) -> None:
        rows = [row("a"), row("b", error="boom", passed=False, grounded=False)]
        self.assertEqual(summarize(rows)["claims_supported_rate"], 1.0)

    def test_an_older_results_file_without_the_field_counts_as_supported(self) -> None:
        self.assertEqual(summarize([row("a")])["claims_supported_rate"], 1.0)


class SuiteThresholdTests(unittest.TestCase):
    def test_a_suite_can_tighten_the_general_thresholds_and_others_keep_them(self) -> None:
        config = {**THRESHOLDS, "suites": {"injection": {"min_question_pass_rate": 1.0, "min_run_pass_rate": 1.0}}}
        injection = thresholds_for(config, "injection")
        self.assertEqual(injection["min_question_pass_rate"], 1.0)
        self.assertEqual(injection["min_run_pass_rate"], 1.0)
        self.assertEqual(injection["max_mean_cost_usd"], 0.05)  # everything else is inherited
        self.assertNotIn("suites", injection)
        golden = thresholds_for(config, "golden")
        self.assertEqual(golden["min_question_pass_rate"], 0.9)
        self.assertNotIn("min_run_pass_rate", golden)

    def test_the_run_pass_rate_gate_fails_when_one_repeat_in_three_obeys_the_attack(self) -> None:
        # Two of three repeats pass, so the question "passes" by majority: fine for quality, not for security.
        rows = [row("attack", passed=True), row("attack", passed=True), row("attack", passed=False)]
        loose = {**THRESHOLDS, "min_question_pass_rate": 1.0}
        self.assertEqual(quality_failures(summarize(rows), loose), [])
        strict = {**loose, "min_run_pass_rate": 1.0}
        failures = quality_failures(summarize(rows), strict)
        self.assertEqual(len(failures), 1)
        self.assertIn("run pass rate", failures[0])

    def test_judge_holds_the_injection_suite_to_the_strict_gate_and_the_golden_suite_to_the_normal_one(self) -> None:
        config = {**THRESHOLDS, "suites": {"injection": {"min_question_pass_rate": 1.0, "min_run_pass_rate": 1.0}}}
        summary = summarize([row("q", passed=True), row("q", passed=True), row("q", passed=False)])
        self.assertEqual(judge(summary, config, "golden"), ([], []))
        problems, failures = judge(summary, config, "injection")
        self.assertEqual(problems, [])
        self.assertTrue(failures)

    def test_judge_reports_an_invalid_run_without_quality_verdicts(self) -> None:
        summary = summarize([row("q", error="boom", passed=False, grounded=False)])
        problems, failures = judge(summary, THRESHOLDS, "golden")
        self.assertTrue(problems)
        self.assertEqual(failures, [])

    def test_the_shipped_thresholds_demand_that_every_injection_run_passes(self) -> None:
        config = json.loads((Path("evals") / "data" / "thresholds.json").read_text(encoding="utf-8"))
        injection = thresholds_for(config, "injection")
        self.assertEqual(injection["min_question_pass_rate"], 1.0)
        self.assertEqual(injection["min_run_pass_rate"], 1.0)
        self.assertLess(thresholds_for(config, "golden")["min_question_pass_rate"], 1.0)


class StoreChoiceTests(unittest.TestCase):
    """The live eval must exercise the code that ships: the PostgreSQL store, not only the in-memory stand-in."""

    def run_with_spy(self, argv: list[str]) -> list:
        import tempfile
        from typing import Any

        seen: list[Any] = []

        def spy_run_one(question: Any, repeat: Any, retriever: Any, store: Any, *rest: Any) -> dict:
            seen.append(store)
            return row(question.id)

        with tempfile.TemporaryDirectory() as results, \
                mock.patch("evals.run_eval.load_env"), \
                mock.patch("evals.run_eval.RESULTS_DIR", Path(results)), \
                mock.patch("evals.run_eval.run_one", side_effect=spy_run_one), \
                mock.patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key", **env_file_values()}), \
                redirect_stderr(io.StringIO()), mock.patch("sys.stdout", new_callable=io.StringIO):
            main(["--repeats", "1", "--only", "lookup-rfi", *argv])
        return seen

    def test_the_memory_store_stays_available_for_offline_runs(self) -> None:
        from rag.storage.store import Index

        (store,) = self.run_with_spy(["--store", "memory"])
        self.assertIsInstance(store, Index)

    def test_the_default_store_is_postgresql(self) -> None:
        from evals.run_eval import build_parser

        self.assertEqual(build_parser().parse_args([]).store, "temp")

    def test_evidence_capture_is_explicit_and_contains_the_returned_context(self) -> None:
        import tempfile

        from evals.questions import QUESTIONS
        from evals.run_eval import run_one
        from rag.providers.embed import hashing_embed
        from rag.retrieval.retrieve import Retriever
        from rag.storage.store import load_or_build
        from tests.support.fakes import ScriptedChat, text_response, tool_response

        with tempfile.TemporaryDirectory() as directory:
            index = load_or_build(Path("data/records/samples/library-renovation"), hashing_embed, "hashing",
                                  Path(directory) / "index.json")
            retriever = Retriever(index, hashing_embed)

            def chat_factory() -> ScriptedChat:
                return ScriptedChat([tool_response("get_record", {"record_id": "RFI-004"}),
                                     text_response("RFI-004 is answered [RFI-004].")])

            normal = run_one(QUESTIONS[0], 0, retriever, index, chat_factory, lambda date: "", "test")
            captured = run_one(QUESTIONS[0], 0, retriever, index, chat_factory, lambda date: "", "test",
                               dataset_version="v1", capture_evidence=True)
            self.assertNotIn("context", normal)
            self.assertIn("[RFI-004]", captured["context"])
            self.assertEqual(captured["dataset_version"], "v1")


@requires_database
class PostgresStoreChoiceTests(StoreChoiceTests):
    def test_by_default_the_agent_runs_on_a_postgresql_store_loaded_by_the_production_sync(self) -> None:
        from rag.storage.pgstore import PostgresStore

        (store,) = self.run_with_spy([])
        self.assertIsInstance(store, PostgresStore)


if __name__ == "__main__":
    unittest.main()
