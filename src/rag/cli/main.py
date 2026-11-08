"""Command line entry point. Run from the project root:

    python -m rag.cli.main "Which RFIs are overdue?"
    python -m rag.cli.main --search-only "abandoned conduit"     (no LLM, no Anthropic key needed)
    python -m rag.cli.main --store memory "..."                  (offline: no database, no embedding key)

Default store: PostgreSQL. Load the records first with `python -m rag.cli.ingest`.
Keys come from .env: ANTHROPIC_API_KEY (answers) and EMBED_API_KEY (embeddings, OpenAI by default).
"""
import argparse
import datetime
import logging
import os
import sys
from pathlib import Path

import psycopg

from rag.agent import Tools, make_cpr_runner, run_agent
from rag.db.connection import ConfigError, connect, database_url_from_env, redact_url, schema_from_env
from rag.infra.resilience import CircuitOpenError, PermanentError, TransientError
from rag.infra.telemetry import RunStats, format_summary, metered_chat
from rag.providers.embed import Embedder, min_vector_score_for
from rag.providers.llm import embedder_from_env, load_env, make_anthropic_chat
from rag.records.snapshot import version
from rag.retrieval.retrieve import Retriever
from rag.storage.pgstore import PostgresStore
from rag.storage.store import Index, load_or_build
from rag.storage.store_api import Store, StoreError


def open_store(args: argparse.Namespace, embed: Embedder, embedder_id: str) -> tuple[Store, psycopg.Connection | None]:
    """The store to search, and the database connection to close afterwards (None for the in-memory store)."""
    if args.store == "memory":
        return load_or_build(Path(args.records) / args.project, embed, embedder_id, Path("data/index.json")), None
    conn = connect(database_url_from_env(), schema=schema_from_env(), read_only=True)  # asking questions never writes
    try:
        store = PostgresStore(conn, args.slug or Path(args.project).name)
    except Exception:
        conn.close()
        raise
    return store, conn


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask questions about a construction project's logs.")
    parser.add_argument("question")
    parser.add_argument("--store", choices=["postgres", "memory"], default="postgres")
    parser.add_argument("--records", default="data/records", help="path to the records repo clone")
    parser.add_argument("--project", default="samples/library-renovation", help="project folder inside --records")
    parser.add_argument("--slug", default=None, help="project name in the database (default: the folder name)")
    parser.add_argument("--as-of", default=datetime.date.today().isoformat(), help="date for overdue checks")
    parser.add_argument("--search-only", action="store_true", help="show retrieved records, skip the LLM")
    parser.add_argument("--model", default="claude-sonnet-5-5", help="Anthropic model id")
    parser.add_argument("--verifier-mode", choices=["off", "shadow", "enforce"], default="enforce",
                        help="experimental model verifier: off, observe, or block (default)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="also print the routine JSON events (calls, searches). Warnings such as retries always print.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")
    load_env()

    embed, embedder_id = embedder_from_env()
    try:
        store, conn = open_store(args, embed, embedder_id)
    except (FileNotFoundError, ValueError, ConfigError) as problem:
        print(f"Cannot load the project: {problem}", file=sys.stderr)
        return 2
    except psycopg.OperationalError as problem:
        print(f"Cannot reach the database ({redact_url(database_url_from_env())}). Is it running? "
              f"Try: docker compose up -d --wait db, or use --store memory.\n  {problem}", file=sys.stderr)
        return 2
    except StoreError as problem:
        print(f"Cannot use the database: {problem}", file=sys.stderr)
        return 2
    try:
        return _answer(args, store, embed, embedder_id)
    finally:
        if conn is not None:
            conn.close()


def _answer(args: argparse.Namespace, store: Store, embed: Embedder, embedder_id: str) -> int:
    generation: int | None = None
    try:
        if isinstance(store, PostgresStore):
            generation = store.source_generation()
            if generation is None:
                raise StoreError("The project has no saved source snapshot. Re-ingest it before querying.")
            stored_models = store.embedder_ids()
            if stored_models and embedder_id not in stored_models:
                raise StoreError(f"The records were embedded with {sorted(stored_models)} but this run would embed the question "
                                 f"with {embedder_id!r}. Use the same EMBED_* settings, or re-ingest.")
        retriever = Retriever(store, embed, min_vector_score=min_vector_score_for(embedder_id, os.environ))
    except (StoreError, psycopg.Error) as problem:
        print(f"Cannot use the database: {problem}", file=sys.stderr)
        return 2
    except ValueError as problem:
        print(f"Configuration problem: {problem}", file=sys.stderr)
        return 2

    if args.search_only:
        try:
            hits = retriever.search(args.question, top_k=5)
        except (StoreError, psycopg.Error, TransientError, PermanentError, CircuitOpenError) as problem:
            print(f"Search failed ({type(problem).__name__}): {problem}", file=sys.stderr)
            return 1
        try:
            if isinstance(store, PostgresStore) and store.source_generation() != generation:
                print("The project changed during the search. Please retry.", file=sys.stderr)
                return 1
        except (StoreError, psycopg.Error):
            print("Could not confirm the project version. Please retry.", file=sys.stderr)
            return 1
        for hit in hits:
            print(f"{hit.score:.3f}  [{hit.chunk.record_id}] {hit.chunk.text[:110]}")
        return 0

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY is not set. Add it to .env, or use --search-only.", file=sys.stderr)
        return 2

    records_root, project_folder = Path(args.records), Path(args.project)
    try:
        if isinstance(store, PostgresStore):
            snapshot = store.source_snapshot()
            if snapshot is None:
                print("The project has no saved source snapshot. Re-ingest it before asking questions.", file=sys.stderr)
                return 2
            dataset_version, files = snapshot
        elif isinstance(store, Index):
            memory_files = store.source_files
            if memory_files is None:
                print("The project has no saved source snapshot.", file=sys.stderr)
                return 2
            files = memory_files
            dataset_version = version(files)
        else:
            raise StoreError("This store cannot provide a project snapshot.")
        stats = RunStats(model=args.model)
        tools = Tools(retriever, store, make_cpr_runner(records_root, project_folder, files), args.as_of, stats=stats,
                      dataset_version=dataset_version, source_files=files)
    except StoreError as problem:
        print(f"Cannot use the dataset snapshot: {problem}", file=sys.stderr)
        return 2
    except ValueError:
        print(f"--as-of must be a date written YYYY-MM-DD, got {args.as_of!r}", file=sys.stderr)
        return 2
    question = f"{args.question}\n\n(Today's date for this question is {args.as_of}.)"
    chat = metered_chat(make_anthropic_chat(api_key, model=args.model), stats)
    try:
        answer = run_agent(question, tools, chat, verifier_mode=args.verifier_mode)
    except (TransientError, PermanentError, CircuitOpenError) as problem:
        # Tokens may already have been spent, so the summary line is printed on failure too.
        print(f"The request failed ({type(problem).__name__}): {problem}", file=sys.stderr)
        print(format_summary(stats), file=sys.stderr)
        return 1

    try:
        if isinstance(store, PostgresStore) and store.source_generation() != generation:
            print("The project changed during the question. Please retry.", file=sys.stderr)
            print(format_summary(stats), file=sys.stderr)
            return 1
    except (StoreError, psycopg.Error):
        print("Could not confirm the project version. Please retry.", file=sys.stderr)
        print(format_summary(stats), file=sys.stderr)
        return 1

    if answer.verification_failed:
        print(answer.text)
        print(f"\nDataset: {dataset_version} (as of {args.as_of})", file=sys.stderr)
        if args.verifier_mode != "off":
            print(f"Verifier: {answer.verification_status}", file=sys.stderr)
        print(f"Withheld because: {'; '.join(answer.withheld_because)}", file=sys.stderr)
        print(f"\n{format_summary(stats)}", file=sys.stderr)
        return 1
    print(answer.text)
    shown_calculations: set[str] = set()
    for item in answer.evidence:
        if item.computed_id and item.computed_id in answer.cited_ids and item.computed_id not in shown_calculations:
            shown_calculations.add(item.computed_id)
            print(f"\n[{item.computed_id}] {item.computed}")
    print(f"\nDataset: {dataset_version} (as of {args.as_of})", file=sys.stderr)
    if args.verifier_mode != "off":
        print(f"Verifier: {answer.verification_status}", file=sys.stderr)
    if answer.uncited:
        print("\nWARNING: the answer cites no record IDs.", file=sys.stderr)
    print(f"\n{format_summary(stats)}", file=sys.stderr)
    return 1 if answer.status in ("retrieval_failed", "budget_exhausted") else 0


if __name__ == "__main__":
    sys.exit(main())
