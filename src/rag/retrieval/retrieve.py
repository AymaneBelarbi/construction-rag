"""RETRIEVE: question -> best chunks, using hybrid search.

Two searches run side by side and their rankings are merged:
- keyword: great at exact tokens such as "RFI-003"
- vector (embedding similarity): great at meaning, such as "ceiling pipe problem"
Merging uses Reciprocal Rank Fusion (RRF): each list gives an item
1 / (k + rank) points and the points are added up. It does not require tuning
the score scales of the two searches.

The searches themselves are done by a Store (in memory or PostgreSQL). This file holds the logic that
must behave the same for every store: merging, pinning exact IDs, and the minimum-similarity cut-off.
"""
import re
import time
from collections.abc import Hashable

from rag.infra.telemetry import log_event
from rag.providers.embed import Embedder
from rag.records.chunker import Chunk, find_record_ids
from rag.storage.store_api import Filters, Hit, Store, chunk_matches

__all__ = ["Hit", "Retriever", "rrf"]

CANDIDATES = 100  # how many results to take from each search before merging


PINNED_AMOUNT_MAX_RECORDS = 3  # a number in more records than this is not specific enough to be an exact request
_AMOUNT = re.compile(r"(?<![\w-])\d{3,}(?![\w-])")


def find_amounts(query: str) -> list[str]:
    """Numbers of 3 or more digits in a question, thousands commas removed, in order, without repeats: "$18,500" gives
    18500 and "4850.00" gives 4850. Digits inside an ID or a date (RFI-003, 2026-08-03) are not amounts."""
    plain = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", query)
    return list(dict.fromkeys(_AMOUNT.findall(plain)))


def rrf[Item: Hashable](rankings: list[list[Item]], k: int = 60) -> dict[Item, float]:
    """Each ranking is a list of items, best first."""
    fused: dict[Item, float] = {}
    for ranking in rankings:
        for position, item in enumerate(ranking):
            fused[item] = fused.get(item, 0.0) + 1.0 / (k + position + 1)
    return fused


class Retriever:
    def __init__(self, store: Store, embed: Embedder, min_vector_score: float = 0.2) -> None:
        self.store = store
        self.embed = embed
        # Vector matches weaker than this are ignored, otherwise a nonsense query still "finds"
        # records by chance. The right value depends on the embedder: tune it with real questions.
        self.min_vector_score = min_vector_score

    def search(self, query: str, top_k: int = 5, filters: Filters | None = None) -> list[Hit]:
        started = time.perf_counter()
        if not query.strip():
            return []

        query_vector = self.embed([query])[0]
        by_vector = [
            hit for hit in self.store.vector_search(query_vector, CANDIDATES, filters)
            if hit.score > 0 and hit.score >= self.min_vector_score
        ]
        by_keyword = self.store.keyword_search(query, CANDIDATES, filters)

        chunks: dict[tuple[str, str], Chunk] = {}
        for hit in (*by_keyword, *by_vector):
            chunks.setdefault(hit.chunk.key, hit.chunk)
        fused = rrf([[h.chunk.key for h in by_keyword], [h.chunk.key for h in by_vector]])

        # An ID in the question is an exact request, not a hint. Records with that ID are pinned
        # to the top (a score bonus bigger than any RRF score) so a weak embedder cannot bury them.
        # They still have to pass the filters.
        for record_id in find_record_ids(query):
            for chunk in self.store.get(record_id):
                if chunk_matches(chunk, filters):
                    chunks.setdefault(chunk.key, chunk)
                    fused[chunk.key] = fused.get(chunk.key, 0.0) + 1.0

        # The same goes for a rare exact amount ("4850.00", "$18,500"): a number says nothing to an embedding model, so
        # the fusion can bury the one record that states it. A number of 3 or more digits that at most
        # PINNED_AMOUNT_MAX_RECORDS records contain is pinned too; a common one (a year) is left to the ranking.
        for amount in find_amounts(query):
            found = self.store.keyword_search(amount, PINNED_AMOUNT_MAX_RECORDS + 1, filters)
            if 0 < len(found) <= PINNED_AMOUNT_MAX_RECORDS:
                for hit in found:
                    chunks.setdefault(hit.chunk.key, hit.chunk)
                    fused[hit.chunk.key] = fused.get(hit.chunk.key, 0.0) + 1.0

        best = sorted(fused, key=lambda key: (-fused[key], key))[:top_k]
        elapsed_ms = (time.perf_counter() - started) * 1000
        # The query text is deliberately not logged: it may contain business data.
        log_event("retrieval", retrieval_ms=round(elapsed_ms, 3), vector_hits=len(by_vector),
                  keyword_hits=len(by_keyword), candidates=len(chunks), returned=len(best),
                  filtered=bool(filters))
        return [Hit(chunks[key], fused[key]) for key in best]
