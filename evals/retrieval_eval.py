"""Retrieval quality: do the right records come back? No LLM, no cost (embedding a short query is a fraction
of a cent with real embeddings, free with the offline hashing embedder).

    python -m evals.retrieval_eval                          offline: a throw-away PostgreSQL schema, hashing embedder
    python -m evals.retrieval_eval --store postgres         your ingested project (use the same embedder as the ingest)
    python -m evals.retrieval_eval --store memory           the in-memory store (a stand-in; production is PostgreSQL)
    python -m evals.retrieval_eval --real-embeddings --store postgres     OpenAI embeddings from .env

Measures, over the search cases: recall@5, recall@10 (what share of the relevant records are in the top k) and
MRR (how high the first relevant record is). Graph cases measure how many expected neighbours `related_records`
reaches. Negative cases check that nonsense questions return NOTHING (which is what makes the "not in the
records" refusal work). With --verbose it also prints the best similarity of every query, which is how the
similarity cut-off for a new embedding model is chosen. Exit code 1 when a gate in retrieval_thresholds.json fails.
"""
import argparse
import json
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg

from evals.retrieval_cases import CASES, RetrievalCase
from evals.tempdb import temporary_database_schema
from rag.db.connection import ConfigError, connect, database_url_from_env, schema_from_env
from rag.infra.resilience import CircuitOpenError, PermanentError, TransientError
from rag.providers.embed import HASHING_ID, Embedder, hashing_embed, min_vector_score_for
from rag.providers.llm import embedder_from_env, load_env
from rag.retrieval.retrieve import Retriever
from rag.storage.pgstore import PostgresStore
from rag.storage.store import load_chunks, make_index
from rag.storage.store_api import Store, StoreError

HERE = Path(__file__).parent
SAMPLE = Path("data/records/samples/library-renovation")
NO_ANSWER_KINDS = ("negative", "adjacent")  # nothing in the records answers these, so nothing should come back
DEPTHS = (5, 10)


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """The share of the relevant records found in the first k results."""
    return len(set(ranked[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    """1 / position of the first relevant result (1.0 if it is first, 0.0 if it is missing)."""
    for position, record_id in enumerate(ranked, start=1):
        if record_id in relevant:
            return 1.0 / position
    return 0.0


def record_ranking(hit_ids: list[str]) -> list[str]:
    """Search returns one entry per stored revision; a record counts once, at its best position."""
    return list(dict.fromkeys(hit_ids))


@dataclass(frozen=True)
class CaseResult:
    case: RetrievalCase
    ranked: list[str]  # record ids, best first (graph cases: the neighbours found)
    recall: dict[int, float]  # recall@k for k in DEPTHS (graph cases: one entry under k=0; negative: empty)
    rr: float | None  # reciprocal rank; None for graph and negative cases
    passed_negative: bool | None = None  # negative cases only: nothing came back
    best_similarity: float | None = None  # the best vector similarity of the query (for choosing the cut-off)


def best_similarity(store: Store, embed: Embedder, query: str, filters: dict[str, str] | None) -> float | None:
    hits = store.vector_search(embed([query])[0], 1, filters)
    return hits[0].score if hits else None


def run_case(case: RetrievalCase, retriever: Retriever, store: Store) -> CaseResult:
    relevant = set(case.relevant)
    if case.kind == "graph":
        assert case.seed is not None
        found = [item.chunk.record_id for item in store.related(case.seed)]  # the limits the agent's tool uses
        return CaseResult(case, found, {0: recall_at_k(found, relevant, len(found) or 1)}, None)
    filters = {**case.filter_dict, "is_latest": "yes"}  # the agent always asks for current revisions only
    similarity = best_similarity(store, retriever.embed, case.query, filters)
    hits = retriever.search(case.query, top_k=max(DEPTHS) * 2, filters=filters)
    ranked = record_ranking([hit.chunk.record_id for hit in hits])
    if case.kind in NO_ANSWER_KINDS:
        return CaseResult(case, ranked, {}, None, passed_negative=not ranked, best_similarity=similarity)
    return CaseResult(case, ranked, {k: recall_at_k(ranked, relevant, k) for k in DEPTHS},
                      reciprocal_rank(ranked, relevant), best_similarity=similarity)


def summarize(results: list[CaseResult]) -> dict[str, Any]:
    searched = [r for r in results if r.case.kind not in ("graph", *NO_ANSWER_KINDS)]
    graphs = [r for r in results if r.case.kind == "graph"]
    negatives = [r for r in results if r.case.kind == "negative"]
    adjacent = [r for r in results if r.case.kind == "adjacent"]

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    summary: dict[str, Any] = {
        "cases": len(results),
        "recall_at_5": mean([r.recall[5] for r in searched]),
        "recall_at_10": mean([r.recall[10] for r in searched]),
        "mrr": mean([r.rr or 0.0 for r in searched]),
        "graph_recall": mean([r.recall[0] for r in graphs]),
        "negative_pass_rate": mean([1.0 if r.passed_negative else 0.0 for r in negatives]),
        "by_kind": {},
    }
    if adjacent:  # reported, never gated: questions near the topic that the records cannot answer (a known gap)
        summary["adjacent_pass_rate"] = mean([1.0 if r.passed_negative else 0.0 for r in adjacent])
    for kind in dict.fromkeys(r.case.kind for r in results):
        group = [r for r in results if r.case.kind == kind]
        if kind in NO_ANSWER_KINDS:
            score = mean([1.0 if r.passed_negative else 0.0 for r in group])
        else:
            score = mean([r.recall[0 if kind == "graph" else 10] for r in group])
        summary["by_kind"][kind] = {"cases": len(group), "score": round(score, 4)}
    return summary


def thresholds_for_embedder(config: dict[str, Any], embedder_id: str) -> dict[str, float]:
    """The gates for this embedding model: the default targets, with the floors the file names for this exact model
    laid over them. The offline stand-in has its own (low) floors because it cannot do semantics; they only
    guard against regressions. Every other model is held to the default targets."""
    return {**config["default"], **config.get("embedders", {}).get(embedder_id, {})}


def check_thresholds(summary: dict[str, Any], thresholds: dict[str, float]) -> list[str]:
    failures: list[str] = []
    for metric in ("recall_at_5", "recall_at_10", "mrr", "graph_recall", "negative_pass_rate"):
        floor = thresholds.get(f"min_{metric}")
        if floor is not None and summary[metric] < floor:
            failures.append(f"{metric} {summary[metric]:.3f} is below {floor}")
    return failures


@contextmanager
def temporary_postgres_store(embed: Embedder, embedder_id: str, dim: int, records: Path) -> Iterator[Store]:
    """A throw-away schema filled with the sample project (see tempdb.py)."""
    with temporary_database_schema(dim) as conn:
        store = PostgresStore(conn, "library-renovation")
        store.sync(load_chunks(records / SAMPLE.relative_to("data/records")), embed, embedder_id)
        yield store


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure retrieval quality without calling an LLM.")
    parser.add_argument("--store", choices=["temp", "postgres", "memory"], default="temp")
    parser.add_argument("--slug", default="library-renovation", help="project name in the database (--store postgres)")
    parser.add_argument("--records", default="data/records")
    parser.add_argument("--real-embeddings", action="store_true", help="use EMBED_* from .env instead of the offline hashing embedder")
    parser.add_argument("--thresholds", default=str(HERE / "data" / "retrieval_thresholds.json"))
    parser.add_argument("--verbose", action="store_true", help="list every case with its best similarity, not only the misses")
    args = parser.parse_args(argv)

    load_env()
    embed, embedder_id = embedder_from_env() if args.real_embeddings else (hashing_embed, HASHING_ID)
    try:
        dim = len(embed(["probe"])[0])  # inside the try: with a real embedder this is a network call
        min_score = min_vector_score_for(embedder_id, os.environ)
        if args.store == "temp":
            with temporary_postgres_store(embed, embedder_id, dim, Path(args.records)) as store:
                results = [run_case(case, Retriever(store, embed, min_score), store) for case in CASES]
        elif args.store == "memory":
            index = make_index(load_chunks(Path(args.records) / SAMPLE.relative_to("data/records")), embed, embedder_id)
            results = [run_case(case, Retriever(index, embed, min_score), index) for case in CASES]
        else:
            conn = connect(database_url_from_env(), schema=schema_from_env(), read_only=True)
            try:
                store = PostgresStore(conn, args.slug)
                if store.count() == 0:
                    raise StoreError(f"Project {args.slug!r} holds no records. Ingest it first: python -m rag.cli.ingest")
                stored = store.embedder_ids()
                if stored and embedder_id not in stored:
                    raise StoreError(f"The records were embedded with {sorted(stored)}, not {embedder_id!r}; use the same embedder.")
                results = [run_case(case, Retriever(store, embed, min_score), store) for case in CASES]
            finally:
                conn.close()
    except (ConfigError, StoreError, FileNotFoundError, ValueError, TransientError, PermanentError, CircuitOpenError) as problem:
        print(f"Cannot run the evaluation: {problem}", file=sys.stderr)
        return 2
    except psycopg.Error as problem:
        print(f"Database problem: {problem}. Is it running? Try: docker compose up -d --wait db", file=sys.stderr)
        return 2

    summary = summarize(results)
    print(f"store={args.store} embedder={embedder_id} min_vector_score={min_score}")
    for result in results:
        if result.case.kind in NO_ANSWER_KINDS:
            ok = bool(result.passed_negative)  # printed when it fails; an adjacent miss is labelled a known gap and is not gated
            verdict = "FAIL" if result.case.kind == "negative" else "known gap"
            outcome = "ok" if result.passed_negative else f"{verdict} returned {result.ranked[:5]}"
            line = f"  {result.case.id:<18} {result.case.kind:<9} {outcome}"
        else:
            key = 0 if result.case.kind == "graph" else 10
            ok = result.recall[key] >= 1.0
            missing = sorted(set(result.case.relevant) - set(result.ranked[: (len(result.ranked) if key == 0 else key)]))
            line = f"  {result.case.id:<18} {result.case.kind:<9} recall={result.recall[key]:.2f}" + (f"  missing={missing}" if missing else "")
        if args.verbose or not ok:
            sim = "" if result.best_similarity is None else f"  best_similarity={result.best_similarity:.3f}"
            print(line + sim)
    print("summary: " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in summary.items()}))

    config = json.loads(Path(args.thresholds).read_text(encoding="utf-8"))
    failures = check_thresholds(summary, thresholds_for_embedder(config, embedder_id))
    for failure in failures:
        print(f"GATE FAILED: {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
