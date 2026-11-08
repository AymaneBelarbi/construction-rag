"""Retrieval logic (fusion, ID pinning, score cutoff) tested once and run against both stores."""
import json
import math
import unittest
from pathlib import Path
from typing import Any

from rag.db.migrate import apply_migrations
from rag.infra.telemetry import EVENT_LOGGER
from rag.providers.embed import hashing_embed
from rag.records.chunker import Chunk
from rag.retrieval.retrieve import Retriever, find_amounts, rrf
from rag.storage.bm25 import Bm25
from rag.storage.pgstore import PostgresStore
from rag.storage.store import build_index, load_chunks, make_index
from tests.support.db_support import requires_database, temporary_schema
from tests.support.fakes import MixinBase

SAMPLE = Path("data/records/samples/library-renovation")


class RankingMathTests(unittest.TestCase):
    def test_bm25_scores_matching_document_higher(self) -> None:
        bm25 = Bm25([["beam", "size"], ["paint", "colour"]])
        scores = bm25.scores(["beam"])
        self.assertGreater(scores[0], scores[1])
        self.assertEqual(scores[1], 0.0)

    def test_rrf_puts_item_ranked_well_in_both_lists_first(self) -> None:
        fused = rrf([[2, 0, 1], [2, 1, 0]])
        best = max(fused, key=lambda i: fused[i])
        self.assertEqual(best, 2)

    def test_rrf_works_with_any_hashable_item(self) -> None:
        fused = rrf([[("RFI-1", "0"), ("RFI-2", "0")], [("RFI-2", "0")]])
        self.assertGreater(fused[("RFI-2", "0")], fused[("RFI-1", "0")])


class FindAmountsTests(unittest.TestCase):
    def test_amounts_are_numbers_of_three_or_more_digits_with_thousands_commas_removed(self) -> None:
        self.assertEqual(find_amounts("4850.00"), ["4850"])
        self.assertEqual(find_amounts("What does $18,500 cover, and $1,250,000 or 22300?"), ["18500", "1250000", "22300"])
        self.assertEqual(find_amounts("500 500 500"), ["500"])  # once

    def test_short_numbers_and_figures_with_units_of_two_digits_are_not_amounts(self) -> None:
        self.assertEqual(find_amounts("30 days, 28 workers, level 2, 99"), [])

    def test_digits_inside_ids_dates_and_words_are_not_amounts(self) -> None:
        self.assertEqual(find_amounts("RFI-013 and CE-007 on 2026-08-03 in DR-2026-08-05"), [])
        self.assertEqual(find_amounts("room102 A201 version2024b"), [])


class RetrieverContract(MixinBase):
    """Run against every store: swapping the store must not change what a question finds."""

    retriever: Retriever

    def test_exact_id_query_finds_that_record_first(self) -> None:
        hits = self.retriever.search("RFI-003", top_k=3)
        self.assertEqual(hits[0].chunk.record_id, "RFI-003")

    def test_topic_query_finds_relevant_record(self) -> None:
        hits = self.retriever.search("abandoned conduit in the ceiling", top_k=3)
        self.assertIn("RFI-003", [h.chunk.record_id for h in hits])

    def test_filters_limit_results(self) -> None:
        hits = self.retriever.search("paint", top_k=10, filters={"record_type": "punch_item", "status": "open"})
        self.assertTrue(hits)
        for hit in hits:
            self.assertEqual(hit.chunk.record_type, "punch_item")
            self.assertEqual(hit.chunk.metadata["status"], "open")

    def test_an_id_in_the_question_is_pinned_but_still_respects_filters(self) -> None:
        hits = self.retriever.search("RFI-010", top_k=3, filters={"status": "open"})
        self.assertEqual(hits[0].chunk.record_id, "RFI-010")  # open, so it passes the filter and is pinned
        hits = self.retriever.search("RFI-003", top_k=3, filters={"status": "open"})
        self.assertNotIn("RFI-003", [h.chunk.record_id for h in hits])  # closed, so the filter excludes it

    def test_a_rare_exact_amount_in_the_question_is_pinned_like_an_id(self) -> None:
        # An embedding of a bare number means nothing, so without the pin the fusion put CE-001 sixth for "4850.00".
        for query, expected in (("4850.00", "CE-001"), ("which change event is 18500", "CE-004"), ("What does $18,500 cover?", "CE-004"),
                                ("22300", "CE-005")):
            with self.subTest(query=query):
                self.assertEqual(self.retriever.search(query, top_k=3)[0].chunk.record_id, expected)

    def test_a_number_that_many_records_contain_or_a_short_one_is_not_pinned(self) -> None:
        for query in ("2026", "00", "28", "2041"):
            with self.subTest(query=query):
                self.assertTrue(all(hit.score < 1.0 for hit in self.retriever.search(query, top_k=10)))  # 1.0 is the pin bonus

    def test_a_pinned_amount_still_has_to_pass_the_filters(self) -> None:
        self.assertEqual(self.retriever.search("4850.00", filters={"status": "nonsense"}), [])
        hits = self.retriever.search("4850.00", top_k=5, filters={"record_type": "daily_report"})
        self.assertNotIn("CE-001", [h.chunk.record_id for h in hits])

    def test_filter_with_no_matches_returns_empty_list(self) -> None:
        self.assertEqual(self.retriever.search("beam", filters={"status": "nonsense"}), [])

    def test_nonsense_query_returns_nothing(self) -> None:
        self.assertEqual(self.retriever.search("xyzzy plugh frobnicate"), [])

    def test_empty_query_returns_nothing(self) -> None:
        self.assertEqual(self.retriever.search("   "), [])

    def test_top_k_is_respected(self) -> None:
        self.assertEqual(len(self.retriever.search("rfi", top_k=2)), 2)

    def test_results_are_best_first_and_not_repeated(self) -> None:
        hits = self.retriever.search("reading room ceiling", top_k=8)
        scores = [h.score for h in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))
        keys = [h.chunk.key for h in hits]
        self.assertEqual(len(keys), len(set(keys)))

    def test_each_search_logs_a_retrieval_event_without_the_query_text(self) -> None:
        with self.assertLogs(EVENT_LOGGER, level="INFO") as logs:
            self.retriever.search("SECRET-QUERY-WORDS RFI-003", top_k=2)
        (event,) = [json.loads(r.getMessage()) for r in logs.records]
        self.assertEqual(event["event"], "retrieval")
        self.assertGreater(event["candidates"], 0)
        self.assertIn("vector_hits", event)
        self.assertIn("keyword_hits", event)
        self.assertGreaterEqual(event["retrieval_ms"], 0.0)
        self.assertNotIn("SECRET-QUERY-WORDS", logs.records[0].getMessage())


# Meaning, spelled out by hand: each text maps to a direction. The offline hashing embedder only counts words, so with it
# a test can never tell whether the vector half of hybrid search works. Here a question and a record share NO word and
# are close only because their vectors point the same way, which is what a real embedding model provides.
MEANING = {
    "Abandoned conduit found above the ceiling.": [1.0, 0.0, 0.0, 0.0],
    "Schedule delay waiting for an answer.": [0.0, 1.0, 0.0, 0.0],
    "Door hardware set was not scheduled.": [0.0, 0.0, 1.0, 0.0],
    "old wiring left behind in the roof space": [0.9, 0.1, 0.0, 0.0],  # a paraphrase of the first record
    "something about lunch": [0.0, 0.0, 0.0, 1.0],  # near nothing
}


def meaning_embed(texts: list[str]) -> list[list[float]]:
    """Unit-length vectors, as every embedder must return (the memory store compares them with a plain dot product)."""
    vectors = [MEANING[text] for text in texts]
    return [[x / math.sqrt(sum(y * y for y in v)) for x in v] for v in vectors]


def meaning_chunks() -> list[Chunk]:
    return [Chunk("rfi", f"RFI-{n}", text, {"is_latest": "yes", "status": "open"})
            for n, text in enumerate(list(MEANING)[:3], start=1)]


class VectorLegContract(MixinBase):
    """The vector half of the hybrid search, shown to matter: delete it and these fail."""

    def make_retriever(self) -> Retriever:
        raise NotImplementedError

    def test_a_paraphrase_with_no_shared_word_is_found_only_by_meaning(self) -> None:
        hits = self.make_retriever().search("old wiring left behind in the roof space", top_k=3)
        self.assertEqual([h.chunk.record_id for h in hits][:1], ["RFI-1"])

    def test_a_question_near_nothing_returns_nothing(self) -> None:
        self.assertEqual(self.make_retriever().search("something about lunch"), [])

    def test_the_similarity_cut_off_decides_what_counts_as_near(self) -> None:
        strict = Retriever(self.make_retriever().store, meaning_embed, min_vector_score=0.999)
        self.assertEqual(strict.search("old wiring left behind in the roof space"), [])  # the similarity is about 0.994
        lax = Retriever(self.make_retriever().store, meaning_embed, min_vector_score=0.5)
        self.assertEqual([h.chunk.record_id for h in lax.search("old wiring left behind in the roof space")], ["RFI-1"])


class MemoryVectorLegTests(VectorLegContract, unittest.TestCase):
    def make_retriever(self) -> Retriever:
        return Retriever(make_index(meaning_chunks(), meaning_embed, "meaning"), meaning_embed, min_vector_score=0.5)


@requires_database
class PostgresVectorLegTests(VectorLegContract, unittest.TestCase):
    def make_retriever(self) -> Retriever:
        schema = temporary_schema()
        conn, _ = schema.__enter__()
        self.addCleanup(schema.__exit__, None, None, None)
        apply_migrations(conn, embedding_dim=4)
        store = PostgresStore(conn, "meaning")
        store.sync(meaning_chunks(), meaning_embed, "meaning")
        return Retriever(store, meaning_embed, min_vector_score=0.5)


class MemoryRetrieverTests(RetrieverContract, unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.retriever = Retriever(build_index(SAMPLE, hashing_embed, "hashing"), hashing_embed)


@requires_database
class PostgresRetrieverTests(RetrieverContract, unittest.TestCase):
    _schema: Any  # the open temporary_schema() context, closed in tearDownClass

    @classmethod
    def setUpClass(cls) -> None:
        cls._schema = temporary_schema()
        conn, _ = cls._schema.__enter__()
        apply_migrations(conn, embedding_dim=256)
        store = PostgresStore(conn, "library-renovation")
        store.sync(load_chunks(SAMPLE), hashing_embed, "hashing")
        cls.retriever = Retriever(store, hashing_embed)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._schema.__exit__(None, None, None)


if __name__ == "__main__":
    unittest.main()
