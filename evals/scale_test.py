"""Scale test: how does the store behave with tens of thousands of records instead of 88?

    python -m evals.scale_test                       10,000 synthetic records, offline hashing embedder
    python -m evals.scale_test --records 100000 --dim 1536     a bigger run with OpenAI-sized vectors (a few minutes)

It fills a throw-away PostgreSQL schema (see tempdb.py) with made-up records of all five types, then measures:
- ingest time (the first load, an unchanged re-run, and a re-run after one record changed),
- latency (p50 / p95 / max, in milliseconds) of keyword, vector and hybrid search, a filtered search, get, related,
- whether "needle" records (each holds one invented word that appears nowhere else) are found: recall@5,
- how big the tables and the vector index are.

Not modelled: the delay of a real embedding service (about 100 to 300 ms per question, on top of the numbers here).
Exit code 1 when a gate fails (a needle is missed, or hybrid p95 is above --max-hybrid-p95-ms), 2 when it cannot run.
"""
import argparse
import json
import logging
import math
import os
import random
import re
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from datetime import date, timedelta
from functools import partial
from typing import Any

import psycopg

from evals.tempdb import temporary_database_schema
from rag.db.connection import ConfigError, fetch_one
from rag.infra.telemetry import EVENT_LOGGER
from rag.providers.embed import HASHING_ID, Embedder, hashing_embed, min_vector_score_for
from rag.providers.llm import load_env
from rag.records.chunker import Chunk, row_to_chunk
from rag.retrieval.retrieve import Retriever
from rag.storage.pgstore import BULK_LOAD_THRESHOLD, VECTOR_INDEX, PostgresStore
from rag.storage.store_api import StoreError

NEEDLES = 20  # records that hold one invented word each
TYPE_SHARES = (("punch_item", 0.50), ("rfi", 0.15), ("submittal", 0.15), ("daily_report", 0.15), ("change_event", 0.05))
CRAFTS = ("Painting", "Electrical", "Plumbing", "Mechanical", "Flooring", "Roofing", "Drywall", "Masonry", "Glazing", "Steel")
FIRMS = ("Sample", "Apex", "Delta", "Summit")
WORDS = (
    "conduit ceiling beam duct grille door frame hardware sealant paint crack stain gypsum board stud track insulation "
    "flashing curb skylight membrane drain valve pipe sprinkler panel breaker cable tray light fixture outlet switch "
    "carpet tile grout threshold stair rail bracket anchor slab joint patch scratch dent gap loose missing damaged "
    "misaligned leaking uneven touch-up replace repair verify inspect clean adjust".split()
)
SPEC_SECTIONS = tuple(f"{a:02d} {b:02d} {c:02d}" for a, b, c in
                      ((5, 50, 0), (7, 52, 0), (8, 11, 13), (8, 71, 0), (9, 21, 16), (9, 51, 13), (9, 65, 0), (9, 91, 23),
                       (21, 13, 13), (22, 11, 16), (23, 31, 13), (26, 5, 19), (26, 51, 0), (27, 10, 0), (28, 31, 0)))
PARTIES = ("Architect", "General contractor", "Owner", "Structural engineer", "MEP engineer", "Superintendent")


def padded_embedder(dim: int) -> Embedder:
    """The offline hashing embedder with its vector repeated up to `dim` numbers (and scaled back to length one), so
    the database sees realistic vector sizes such as OpenAI's 1536. Similarity between two texts is unchanged."""
    base = len(hashing_embed(["probe"])[0])
    if dim <= 0 or dim % base != 0:
        raise ValueError(f"--dim must be a multiple of {base}, got {dim}")
    copies = dim // base
    scale = 1 / math.sqrt(copies)

    def embed(texts: list[str]) -> list[list[float]]:
        return [[x * scale for x in vector] * copies for vector in hashing_embed(texts)]

    return embed


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile: the smallest value that at least q percent of the values are at or below."""
    if not values:
        raise ValueError("percentile of no values")
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[rank - 1]


def synthetic_chunks(count: int, seed: int = 7) -> tuple[list[Chunk], dict[str, str]]:
    """`count` made-up records of every type, built with the real chunker so the text and relations look like real
    ones. A few trades and rooms are shared by very many records (the graph's hubs). Returns the chunks and the
    needles: {invented word: the one record that contains it}."""
    if count < NEEDLES:
        raise ValueError(f"At least {NEEDLES} records are needed (one per needle), got {count}")
    rng = random.Random(seed)
    trades = [f"{firm} {craft}" for firm in FIRMS for craft in CRAFTS]
    trade_weights = [1 / (rank + 1) for rank in range(len(trades))]  # rank 1 is shared far more than rank 40
    rooms = [f"Room {100 + n}" for n in range(max(50, count // 20))]
    needle_positions = {(2 * n + 1) * count // (2 * NEEDLES): f"zorquat{n:03d}" for n in range(NEEDLES)}

    types: list[str] = []
    for kind, share in TYPE_SHARES:
        types += [kind] * round(count * share)
    types = (types + ["punch_item"] * count)[:count]  # rounding can leave the list a little short or long
    rng.shuffle(types)
    for position in needle_positions:
        types[position] = "punch_item"  # needles live in punch item descriptions
    rfi_numbers = [f"RFI-{i:06d}" for i, kind in enumerate(types) if kind == "rfi"]

    def phrase() -> str:
        return " ".join(rng.choices(WORDS, k=rng.randint(4, 7)))

    chunks: list[Chunk] = []
    needles: dict[str, str] = {}
    for i, kind in enumerate(types):
        base = {"project_id": "P-SCALE"}
        when = date(2020, 1, 1) + timedelta(days=i % 3000)
        if kind == "punch_item":
            word = needle_positions.get(i)
            description = phrase() + (f" {word}" if word else "")
            row = {**base, "item_number": f"PL-{i:06d}", "status": rng.choice(("open", "closed", "ready_for_review")),
                   "location": rng.choice(rooms), "description": description,
                   "responsible_party": rng.choices(trades, trade_weights)[0], "created_date": str(when)}
            if word:
                needles[word] = row["item_number"]
        elif kind == "rfi":
            row = {**base, "rfi_number": f"RFI-{i:06d}", "status": rng.choice(("open", "closed")), "subject": phrase(),
                   "question": phrase(), "references": f"A-{rng.randint(100, 399)}, spec {rng.choice(SPEC_SECTIONS)}",
                   "ball_in_court": rng.choice(PARTIES), "created_date": str(when)}
        elif kind == "submittal":
            row = {**base, "submittal_number": f"S-{i:06d}", "revision": "0", "status": rng.choice(("approved", "submitted")),
                   "title": phrase(), "spec_section": rng.choice(SPEC_SECTIONS),
                   "responsible_party": rng.choices(trades, trade_weights)[0], "ball_in_court": rng.choice(PARTIES)}
        elif kind == "change_event":
            row = {**base, "event_number": f"CE-{i:06d}", "status": rng.choice(("pending", "approved")), "title": phrase(),
                   "related_rfi": rng.choice(rfi_numbers) if rfi_numbers else "", "change_order_number": f"CO-{i:06d}"}
        else:
            row = {**base, "report_date": str(date(2000, 1, 1) + timedelta(days=i)), "weather": rng.choice(("Dry", "Rain")),
                   "work_performed": phrase(), "manpower_total": str(rng.randint(5, 40)), "author": "Superintendent"}
        chunks.append(row_to_chunk(kind, row))
    return chunks, needles


class _EventCatcher(logging.Handler):
    """Collects the names of the JSON events the store logs while it runs."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.events.append(json.loads(record.getMessage())["event"])


@contextmanager
def caught_events() -> Iterator[_EventCatcher]:
    logger = logging.getLogger(EVENT_LOGGER)
    catcher, old_level = _EventCatcher(), logger.level
    logger.addHandler(catcher)
    logger.setLevel(logging.INFO)
    try:
        yield catcher
    finally:
        logger.removeHandler(catcher)
        logger.setLevel(old_level)


def timed(label_calls: list[Callable[[], Any]]) -> dict[str, float]:
    """Run each call once (after three untimed warm-up calls) and summarise how long they took, in milliseconds."""
    for call in label_calls[:3]:
        call()
    durations: list[float] = []
    for call in label_calls:
        started = time.perf_counter()
        call()
        durations.append((time.perf_counter() - started) * 1000)
    return {"p50": round(percentile(durations, 50), 2), "p95": round(percentile(durations, 95), 2),
            "max": round(max(durations), 2)}


def seconds_and_report(store: PostgresStore, chunks: list[Chunk], embed: Embedder, embedder_id: str) -> dict[str, Any]:
    started = time.perf_counter()
    report = store.sync(chunks, embed, embedder_id)
    return {"seconds": round(time.perf_counter() - started, 3), "inserted": report.inserted, "updated": report.updated,
            "unchanged": report.unchanged, "embedded": report.embedded}


def megabytes(conn: psycopg.Connection, sql: str) -> float:
    return round(fetch_one(conn, sql)[0] / 1_048_576, 3)


def run_scale_test(conn: psycopg.Connection, chunks: list[Chunk], needles: dict[str, str], embed: Embedder,
                   embedder_id: str, queries: int = 100, bulk_threshold: int = BULK_LOAD_THRESHOLD,
                   slug: str = "scale", seed: int = 7) -> dict[str, Any]:
    store = PostgresStore(conn, slug, bulk_threshold=bulk_threshold)
    retriever = Retriever(store, embed, min_vector_score_for(embedder_id, os.environ))

    with caught_events() as caught:
        ingest = seconds_and_report(store, chunks, embed, embedder_id)
    ingest["bulk_path"] = "bulk_load" in caught.events
    conn.execute("ANALYZE chunks, nodes, edges")  # fresh planner statistics, which autovacuum would do within a minute

    resync_unchanged = seconds_and_report(store, chunks, embed, embedder_id)
    changed = [replace(chunks[0], text=chunks[0].text + " Revised note.")] + chunks[1:]
    resync_one_change = seconds_and_report(store, changed, embed, embedder_id)

    rng = random.Random(seed)
    sampled = rng.sample(changed, min(queries, len(changed)))
    texts = [" ".join(rng.sample(words, min(3, len(words)))) for words in
             (re.findall(r"[a-z]{4,}", chunk.text.lower()) for chunk in sampled)]
    vectors = embed(texts)
    filters = {"record_type": "punch_item", "status": "open"}
    latency = {
        "keyword": timed([partial(store.keyword_search, q, 100) for q in texts]),
        "vector": timed([partial(store.vector_search, v, 100) for v in vectors]),
        "hybrid": timed([partial(retriever.search, q, 10) for q in texts]),
        "hybrid_filtered": timed([partial(retriever.search, q, 10, filters) for q in texts]),
        "get": timed([partial(store.get, c.record_id) for c in sampled]),
        "related": timed([partial(store.related, c.record_id) for c in sampled]),
    }

    found = sum(1 for word, record_id in needles.items()
                if record_id in [hit.chunk.record_id for hit in retriever.search(word, 5)])
    sizes = {
        "chunks_table": megabytes(conn, "SELECT pg_relation_size('chunks')"),
        "chunks_with_indexes": megabytes(conn, "SELECT pg_total_relation_size('chunks')"),
        "vector_index": megabytes(conn, f"SELECT pg_relation_size('{VECTOR_INDEX}')"),
        "graph": megabytes(conn, "SELECT pg_total_relation_size('nodes') + pg_total_relation_size('edges')"),
    }
    return {"records": len(chunks), "queries": len(texts), "ingest": ingest, "resync_unchanged": resync_unchanged,
            "resync_one_change": resync_one_change, "latency_ms": latency,
            "needle_recall_at_5": round(found / len(needles), 4) if needles else 1.0, "size_mb": sizes}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure the store with many synthetic records.")
    parser.add_argument("--records", type=int, default=10_000)
    parser.add_argument("--queries", type=int, default=100, help="how many questions to time per operation")
    parser.add_argument("--dim", type=int, default=256, help="vector size, a multiple of 256 (1536 matches text-embedding-3-small)")
    parser.add_argument("--bulk-threshold", type=int, default=BULK_LOAD_THRESHOLD)
    parser.add_argument("--max-hybrid-p95-ms", type=float, default=250.0, help="gate: the hybrid search p95 must stay below this")
    args = parser.parse_args(argv)

    load_env()
    try:
        chunks, needles = synthetic_chunks(args.records)
        embed = padded_embedder(args.dim)
        with temporary_database_schema(args.dim) as conn:
            report = run_scale_test(conn, chunks, needles, embed, f"{HASHING_ID}-x{args.dim}", args.queries, args.bulk_threshold)
    except (ConfigError, StoreError, ValueError) as problem:
        print(f"Cannot run the scale test: {problem}", file=sys.stderr)
        return 2
    except psycopg.Error as problem:
        print(f"Database problem: {problem}. Is it running? Try: docker compose up -d --wait db", file=sys.stderr)
        return 2

    print(json.dumps({**report, "vector_size": args.dim}, indent=2))
    failures = []
    if report["needle_recall_at_5"] < 1.0:
        failures.append(f"needle recall@5 is {report['needle_recall_at_5']}, expected 1.0")
    if report["latency_ms"]["hybrid"]["p95"] > args.max_hybrid_p95_ms:
        failures.append(f"hybrid p95 {report['latency_ms']['hybrid']['p95']} ms is above {args.max_hybrid_p95_ms}")
    for failure in failures:
        print(f"GATE FAILED: {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
