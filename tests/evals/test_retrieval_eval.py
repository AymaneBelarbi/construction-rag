import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

from evals.retrieval_cases import CASES, RetrievalCase
from evals.retrieval_eval import (
    CaseResult,
    check_thresholds,
    main,
    recall_at_k,
    reciprocal_rank,
    record_ranking,
    run_case,
    summarize,
    thresholds_for_embedder,
)
from rag.db.migrate import apply_migrations
from rag.providers.embed import HASHING_ID, hashing_embed, min_vector_score_for
from rag.retrieval.retrieve import Retriever
from rag.storage.pgstore import PostgresStore
from rag.storage.store import load_chunks, make_index
from rag.storage.store_api import Hit, Related, RelatedRecords
from tests.support.db_support import requires_database, temporary_schema

SAMPLE = Path("data/records/samples/library-renovation")
CONFIG = json.loads(Path("evals/data/retrieval_thresholds.json").read_text(encoding="utf-8"))
THRESHOLDS = thresholds_for_embedder(CONFIG, HASHING_ID)  # the offline stand-in: regression floors


class MetricTests(unittest.TestCase):
    def test_recall_is_the_share_of_relevant_records_in_the_first_k(self) -> None:
        ranked = ["A", "X", "B", "Y", "C"]
        self.assertEqual(recall_at_k(ranked, {"A", "B", "C"}, 3), 2 / 3)
        self.assertEqual(recall_at_k(ranked, {"A", "B", "C"}, 5), 1.0)
        self.assertEqual(recall_at_k(ranked, {"Z"}, 5), 0.0)
        self.assertEqual(recall_at_k(ranked, {"A"}, 1), 1.0)

    def test_reciprocal_rank_is_one_over_the_position_of_the_first_hit(self) -> None:
        self.assertEqual(reciprocal_rank(["A", "X"], {"A"}), 1.0)
        self.assertEqual(reciprocal_rank(["X", "Y", "B"], {"A", "B"}), 1 / 3)
        self.assertEqual(reciprocal_rank(["X"], {"A"}), 0.0)

    def test_a_record_with_several_revisions_counts_once_at_its_best_position(self) -> None:
        self.assertEqual(record_ranking(["S-1", "RFI-2", "S-1", "PL-3"]), ["S-1", "RFI-2", "PL-3"])


class FakeStore:
    def __init__(self, related: RelatedRecords | None = None) -> None:
        self._related = related or RelatedRecords()
        self.related_calls: list[tuple] = []

    def vector_search(self, vector: Any, limit: int, filters: Any = None) -> list[Hit]:
        return []

    def related(self, record_id: str, limit: int = 15, per_entity: int = 5) -> RelatedRecords:
        self.related_calls.append((record_id, limit, per_entity))
        return self._related


class FakeRetriever:
    def __init__(self, ids: list[str]) -> None:
        self.ids = ids
        self.embed = hashing_embed
        self.filters_seen: list[Any] = []

    def search(self, query: str, top_k: int = 5, filters: Any = None) -> list[Hit]:
        from rag.records.chunker import Chunk

        self.filters_seen.append(filters)

        return [Hit(Chunk("rfi", record_id, "t", {}), 1.0) for record_id in self.ids][:top_k]


class ThresholdSelectionTests(unittest.TestCase):
    CONFIG = {"default": {"min_mrr": 0.9, "min_recall_at_5": 0.9}, "embedders": {"stand-in": {"min_mrr": 0.5}}}

    def test_an_embedder_with_its_own_floors_overrides_the_default_ones_it_names(self) -> None:
        chosen = thresholds_for_embedder(self.CONFIG, "stand-in")
        self.assertEqual(chosen, {"min_mrr": 0.5, "min_recall_at_5": 0.9})

    def test_any_other_embedder_gets_the_default_targets(self) -> None:
        self.assertEqual(thresholds_for_embedder(self.CONFIG, "api:https://x:text-embedding-3-small"), self.CONFIG["default"])

    def test_a_real_model_is_held_to_a_higher_bar_than_the_stand_in(self) -> None:
        # The stand-in cannot do semantics, so its floors are regression guards; the default is the quality target.
        real = thresholds_for_embedder(CONFIG, "api:https://api.openai.com/v1:text-embedding-3-small")
        for metric in ("min_recall_at_5", "min_recall_at_10", "min_mrr"):
            with self.subTest(metric=metric):
                self.assertGreater(real[metric], THRESHOLDS[metric])
        self.assertEqual(real["min_negative_pass_rate"], 1.0)
        self.assertEqual(THRESHOLDS["min_negative_pass_rate"], 1.0)  # nonsense must return nothing for every embedder


class RunCaseTests(unittest.TestCase):
    def test_a_search_case_measures_recall_at_5_and_10_and_the_first_hit(self) -> None:
        case = RetrievalCase("c", "topic", "q", ("RFI-3", "RFI-9"))
        ids = [f"RFI-{n}" for n in (1, 2, 3, 4, 5, 6, 7, 8, 9)]
        result = run_case(case, FakeRetriever(ids), FakeStore())  # type: ignore[arg-type]
        self.assertEqual(result.recall, {5: 0.5, 10: 1.0})
        self.assertEqual(result.rr, 1 / 3)

    def test_searches_ask_for_current_revisions_only_as_the_agent_does(self) -> None:
        retriever = FakeRetriever(["RFI-1"])
        run_case(RetrievalCase("c", "topic", "q", ("RFI-1",)), retriever, FakeStore())  # type: ignore[arg-type]
        run_case(RetrievalCase("f", "filtered", "q", ("RFI-1",), filters=(("status", "open"),)), retriever, FakeStore())  # type: ignore[arg-type]
        self.assertEqual(retriever.filters_seen, [{"is_latest": "yes"}, {"status": "open", "is_latest": "yes"}])

    def test_graph_cases_use_the_limits_the_agent_uses(self) -> None:
        store = FakeStore()
        run_case(RetrievalCase("g", "graph", "q", ("PL-6",), seed="PL-1"), FakeRetriever([]), store)  # type: ignore[arg-type]
        self.assertEqual(store.related_calls, [("PL-1", 15, 5)])

    def test_an_adjacent_case_is_judged_like_a_negative_one(self) -> None:
        case = RetrievalCase("a", "adjacent", "what did the owner say about the parking lot budget", ())
        self.assertTrue(run_case(case, FakeRetriever([]), FakeStore()).passed_negative)  # type: ignore[arg-type]
        self.assertFalse(run_case(case, FakeRetriever(["CE-2"]), FakeStore()).passed_negative)  # type: ignore[arg-type]

    def test_a_negative_case_passes_only_when_nothing_comes_back(self) -> None:
        case = RetrievalCase("n", "negative", "q", ())
        self.assertTrue(run_case(case, FakeRetriever([]), FakeStore()).passed_negative)  # type: ignore[arg-type]
        self.assertFalse(run_case(case, FakeRetriever(["RFI-1"]), FakeStore()).passed_negative)  # type: ignore[arg-type]

    def test_a_graph_case_measures_the_neighbours_the_store_reaches(self) -> None:
        from rag.records.chunker import Chunk

        found = RelatedRecords([Related(Chunk("punch_item", "PL-6", "t", {}), "shares trade: X")])
        case = RetrievalCase("g", "graph", "q", ("PL-6", "PL-7"), seed="PL-1")
        result = run_case(case, FakeRetriever([]), FakeStore(found))  # type: ignore[arg-type]
        self.assertEqual(result.recall[0], 0.5)


def result(kind: str, recall10: float = 1.0, recall5: float | None = None, rr: float | None = 1.0, neg: bool | None = None) -> CaseResult:
    no_answer = kind in ("negative", "adjacent")
    case = RetrievalCase("x", kind, "q", () if no_answer else ("A",))
    recall = {} if no_answer else ({0: recall10} if kind == "graph" else {5: recall10 if recall5 is None else recall5, 10: recall10})
    return CaseResult(case, [], recall, None if kind in ("graph", "negative", "adjacent") else rr, neg)


class AdjacentReportingTests(unittest.TestCase):
    def test_adjacent_cases_are_reported_but_never_enter_the_gated_negative_rate_or_the_gates(self) -> None:
        results = [result("topic"), result("graph"), result("negative", neg=True),
                   result("adjacent", neg=False), result("adjacent", neg=True)]
        summary = summarize(results)
        self.assertEqual(summary["negative_pass_rate"], 1.0)  # the adjacent failure is not in it
        self.assertEqual(summary["adjacent_pass_rate"], 0.5)
        self.assertEqual(summary["by_kind"]["adjacent"], {"cases": 2, "score": 0.5})
        self.assertEqual(check_thresholds(summary, THRESHOLDS), [])  # a known gap, reported and not gated

    def test_the_rate_is_absent_when_there_are_no_adjacent_cases(self) -> None:
        self.assertNotIn("adjacent_pass_rate", summarize([result("topic")]))

    def test_the_shipped_cases_include_adjacent_questions_that_expect_no_records(self) -> None:
        adjacent = [c for c in CASES if c.kind == "adjacent"]
        self.assertGreaterEqual(len(adjacent), 3)
        self.assertTrue(all(c.relevant == () for c in adjacent))


class SummaryTests(unittest.TestCase):
    def test_each_kind_of_case_feeds_only_its_own_metric(self) -> None:
        summary = summarize([result("topic", 1.0, 0.5, 1.0), result("semantic", 0.0, 0.0, 0.0),
                             result("graph", 0.5), result("negative", neg=True), result("negative", neg=False)])
        self.assertEqual(summary["recall_at_10"], 0.5)
        self.assertEqual(summary["recall_at_5"], 0.25)
        self.assertEqual(summary["mrr"], 0.5)
        self.assertEqual(summary["graph_recall"], 0.5)
        self.assertEqual(summary["negative_pass_rate"], 0.5)

    def test_each_gate_reports_its_own_failure_and_a_value_at_the_floor_passes(self) -> None:
        good = summarize([result("topic"), result("graph"), result("negative", neg=True)])
        self.assertEqual(check_thresholds(good, THRESHOLDS), [])
        for kind, name in (("topic", "recall_at_10"), ("graph", "graph_recall"), ("negative", "negative_pass_rate")):
            bad = summarize([result("topic", 0.1, 0.1, 0.1) if kind == "topic" else result("topic"),
                             result("graph", 0.0) if kind == "graph" else result("graph"),
                             result("negative", neg=False) if kind == "negative" else result("negative", neg=True)])
            with self.subTest(gate=name):
                self.assertIn(name, " ".join(check_thresholds(bad, THRESHOLDS)))
        at_the_floors = {name: THRESHOLDS[f"min_{name}"] for name in
                         ("recall_at_5", "recall_at_10", "mrr", "graph_recall", "negative_pass_rate")}
        self.assertEqual(check_thresholds(at_the_floors, THRESHOLDS), [])


def evaluate_all(store: Any, retriever: Any) -> list[CaseResult]:
    return [run_case(case, retriever, store) for case in CASES]


class OfflineBaselineTests(unittest.TestCase):
    """The regression test: with the offline embedder, every case a no-semantics embedder CAN get right must be
    right, and the gates in retrieval_thresholds.json must hold. A change that hurts retrieval fails here."""

    def check(self, results: list[CaseResult]) -> None:
        self.assertEqual(check_thresholds(summarize(results), THRESHOLDS), [])
        for r in results:
            if r.case.kind in ("exact", "topic", "filtered", "graph"):
                key = 0 if r.case.kind == "graph" else 10
                with self.subTest(case=r.case.id):
                    self.assertEqual(r.recall[key], 1.0, r.ranked)
            if r.case.kind == "negative":
                with self.subTest(case=r.case.id):
                    self.assertTrue(r.passed_negative, r.ranked)

    def test_the_in_memory_store(self) -> None:
        index = make_index(load_chunks(SAMPLE), hashing_embed, HASHING_ID)
        self.check(evaluate_all(index, Retriever(index, hashing_embed, min_vector_score_for(HASHING_ID, {}))))

    @requires_database
    def test_the_postgresql_store(self) -> None:
        with temporary_schema() as (conn, _):
            apply_migrations(conn, 256)
            store = PostgresStore(conn, "library-renovation")
            store.sync(load_chunks(SAMPLE), hashing_embed, HASHING_ID)
            self.check(evaluate_all(store, Retriever(store, hashing_embed, min_vector_score_for(HASHING_ID, {}))))


class CliTests(unittest.TestCase):
    def run_main(self, argv: list[str]) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_the_memory_store_run_passes_its_gates(self) -> None:
        code, out, err = self.run_main(["--store", "memory"])
        self.assertEqual(code, 0, err)
        self.assertIn("summary:", out)

    def test_a_gate_that_cannot_be_met_exits_with_1_and_names_the_metric(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            strict = Path(tmp) / "t.json"
            strict.write_text(json.dumps({"default": {"min_recall_at_10": 1.01}}), encoding="utf-8")
            code, out, err = self.run_main(["--store", "memory", "--thresholds", str(strict)])
        self.assertEqual(code, 1)
        self.assertIn("recall_at_10", err)

    def test_an_embedding_service_failure_is_a_clear_error_not_a_gate_failure(self) -> None:
        from unittest import mock

        from rag.infra.resilience import CircuitOpenError, TransientError

        for problem in (TransientError("service down"), CircuitOpenError("circuit open")):
            def failing(texts: list[str], problem: Exception = problem) -> list[list[float]]:
                raise problem

            with self.subTest(problem=type(problem).__name__), \
                    mock.patch("evals.retrieval_eval.embedder_from_env", return_value=(failing, "api:test:model")):
                code, out, err = self.run_main(["--store", "memory", "--real-embeddings"])
            self.assertEqual(code, 2)  # 1 is reserved for "the quality gates failed"
            self.assertIn("Cannot run the evaluation", err)

    def test_a_missing_records_folder_is_a_clear_error(self) -> None:
        code, out, err = self.run_main(["--store", "memory", "--records", "/nonexistent"])
        self.assertEqual(code, 2)
        self.assertIn("Cannot run the evaluation", err)

    def test_verbose_lists_every_case_with_its_best_similarity(self) -> None:
        code, out, err = self.run_main(["--store", "memory", "--verbose"])
        self.assertIn("best_similarity=", out)
        self.assertIn("id-rfi", out)  # a passing case is listed too


if __name__ == "__main__":
    unittest.main()
