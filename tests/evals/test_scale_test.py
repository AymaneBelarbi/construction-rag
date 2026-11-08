import io
import math
import unittest
from collections import Counter
from contextlib import redirect_stderr
from unittest import mock

from evals.scale_test import NEEDLES, main, padded_embedder, percentile, run_scale_test, synthetic_chunks
from rag.db.migrate import apply_migrations
from rag.providers.embed import HASHING_ID, hashing_embed
from tests.support.db_support import requires_database, temporary_schema


class SyntheticDataTests(unittest.TestCase):
    def test_it_makes_exactly_the_requested_number_of_distinct_records(self) -> None:
        chunks, _ = synthetic_chunks(500)
        self.assertEqual(len(chunks), 500)
        self.assertEqual(len({chunk.key for chunk in chunks}), 500)

    def test_the_same_seed_gives_the_same_records_and_another_seed_does_not(self) -> None:
        first, _ = synthetic_chunks(300, seed=1)
        again, _ = synthetic_chunks(300, seed=1)
        other, _ = synthetic_chunks(300, seed=2)
        self.assertEqual(first, again)
        self.assertNotEqual(first, other)

    def test_every_record_type_is_present(self) -> None:
        chunks, _ = synthetic_chunks(500)
        self.assertEqual(set(Counter(chunk.record_type for chunk in chunks)),
                         {"rfi", "submittal", "change_event", "daily_report", "punch_item"})

    def test_each_needle_word_appears_in_exactly_one_record_and_that_record_is_named(self) -> None:
        chunks, needles = synthetic_chunks(800)
        self.assertEqual(len(needles), NEEDLES)
        for word, record_id in needles.items():
            with self.subTest(word=word):
                holders = [chunk.record_id for chunk in chunks if word in chunk.text.lower()]
                self.assertEqual(holders, [record_id])

    def test_a_few_trades_are_shared_by_many_records_so_the_graph_has_hubs(self) -> None:
        chunks, _ = synthetic_chunks(2000)
        members = Counter(name for chunk in chunks for relation, kind, name in chunk.relations if kind == "trade")
        self.assertGreaterEqual(max(members.values()), 40)

    def test_too_few_records_for_the_needles_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            synthetic_chunks(NEEDLES - 1)


class PercentileTests(unittest.TestCase):
    def test_nearest_rank_percentiles(self) -> None:
        values = [float(n) for n in range(1, 101)]
        self.assertEqual(percentile(values, 50), 50.0)
        self.assertEqual(percentile(values, 95), 95.0)
        self.assertEqual(percentile(values, 100), 100.0)

    def test_the_input_order_does_not_matter(self) -> None:
        self.assertEqual(percentile([9.0, 1.0, 5.0], 50), 5.0)

    def test_one_value_is_every_percentile_and_nothing_is_an_error(self) -> None:
        self.assertEqual(percentile([7.0], 95), 7.0)
        with self.assertRaises(ValueError):
            percentile([], 50)


class PaddedEmbedderTests(unittest.TestCase):
    def test_it_makes_vectors_of_the_requested_size_with_length_one(self) -> None:
        for dim in (256, 512, 1536):
            with self.subTest(dim=dim):
                (vector,) = padded_embedder(dim)(["abandoned conduit"])
                self.assertEqual(len(vector), dim)
                self.assertAlmostEqual(math.sqrt(sum(x * x for x in vector)), 1.0, places=6)

    def test_similarity_between_two_texts_is_the_same_as_with_the_plain_embedder(self) -> None:
        a, b = padded_embedder(1536)(["abandoned conduit", "conduit in the ceiling"])
        (plain_a, plain_b) = hashing_embed(["abandoned conduit", "conduit in the ceiling"])
        self.assertAlmostEqual(sum(x * y for x, y in zip(a, b, strict=True)), sum(x * y for x, y in zip(plain_a, plain_b, strict=True)), places=6)

    def test_a_size_that_is_not_a_multiple_of_the_base_size_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            padded_embedder(300)


class CliTests(unittest.TestCase):
    def test_a_missing_database_password_is_a_clear_error_not_a_traceback(self) -> None:
        err = io.StringIO()
        with mock.patch("evals.scale_test.load_env"), mock.patch.dict("os.environ", {}, clear=True), redirect_stderr(err):
            code = main(["--records", "30"])
        self.assertEqual(code, 2)
        self.assertIn("RAG_DB_PASSWORD", err.getvalue())


@requires_database
class RunScaleTestTests(unittest.TestCase):
    def setUp(self) -> None:
        self._schema = temporary_schema()
        self.conn, _ = self._schema.__enter__()
        self.addCleanup(self._schema.__exit__, None, None, None)
        apply_migrations(self.conn, embedding_dim=len(hashing_embed(["probe"])[0]))

    def test_a_small_run_reports_every_measurement_and_finds_every_needle(self) -> None:
        chunks, needles = synthetic_chunks(300)
        report = run_scale_test(self.conn, chunks, needles, hashing_embed, HASHING_ID, queries=5, bulk_threshold=100)
        self.assertEqual(report["records"], 300)
        self.assertEqual(report["needle_recall_at_5"], 1.0)
        for operation in ("keyword", "vector", "hybrid", "hybrid_filtered", "get", "related"):
            with self.subTest(operation=operation):
                self.assertEqual(set(report["latency_ms"][operation]), {"p50", "p95", "max"})
        self.assertGreater(report["size_mb"]["chunks_table"], 0)

    def test_the_first_load_embeds_everything_and_a_second_run_embeds_nothing(self) -> None:
        chunks, needles = synthetic_chunks(300)
        report = run_scale_test(self.conn, chunks, needles, hashing_embed, HASHING_ID, queries=3, bulk_threshold=100)
        self.assertEqual(report["ingest"]["embedded"], 300)
        self.assertEqual(report["resync_unchanged"]["embedded"], 0)

    def test_changing_one_record_embeds_exactly_one_text(self) -> None:
        chunks, needles = synthetic_chunks(300)
        report = run_scale_test(self.conn, chunks, needles, hashing_embed, HASHING_ID, queries=3, bulk_threshold=100)
        self.assertEqual(report["resync_one_change"]["embedded"], 1)

    def test_the_bulk_path_is_used_for_a_big_load_and_not_for_a_small_one(self) -> None:
        chunks, needles = synthetic_chunks(300)
        big = run_scale_test(self.conn, chunks, needles, hashing_embed, HASHING_ID, queries=2, bulk_threshold=100,
                             slug="big")
        small = run_scale_test(self.conn, chunks, needles, hashing_embed, HASHING_ID, queries=2, bulk_threshold=10_000,
                               slug="small")
        self.assertTrue(big["ingest"]["bulk_path"])
        self.assertFalse(small["ingest"]["bulk_path"])


if __name__ == "__main__":
    unittest.main()
