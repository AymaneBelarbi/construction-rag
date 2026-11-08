"""INGEST: load a project's CSV files into PostgreSQL, embedding what is new or changed.

    python -m rag.cli.ingest                                  the sample project
    python -m rag.cli.ingest --project samples/other --slug other

Safe to run again and again: records whose text did not change are not embedded a second time, records
removed from the CSVs are removed from the database, and the whole update is one transaction.
Needs the tables (python -m rag.db.migrate) and real embeddings (EMBED_API_KEY in .env).
"""
import argparse
import sys
from pathlib import Path

import psycopg

from rag.db.connection import ConfigError, connect, database_url_from_env, redact_url, schema_from_env
from rag.infra.resilience import CircuitOpenError, PermanentError, TransientError
from rag.providers.llm import embedder_from_env, load_env
from rag.records.snapshot import capture, materialize, version
from rag.storage.pgstore import PostgresStore
from rag.storage.store import load_chunks
from rag.storage.store_api import StoreError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load a project's records into the database.")
    parser.add_argument("--records", default="data/records", help="path to the records repo clone")
    parser.add_argument("--project", default="samples/library-renovation", help="project folder inside --records")
    parser.add_argument("--slug", default=None, help="name of the project in the database (default: the folder name)")
    parser.add_argument("--allow-missing-types", action="store_true",
                        help="delete the stored records of a record type whose CSV is missing or empty (refused by default)")
    args = parser.parse_args(argv)
    slug = args.slug or Path(args.project).name

    load_env()
    try:
        files = capture(Path(args.records) / args.project)
        with materialize(files) as folder:
            chunks = load_chunks(folder)
    except (FileNotFoundError, ValueError) as problem:  # a missing folder, no CSV files, or a row that cannot be read
        print(f"Cannot load the project records: {problem}", file=sys.stderr)
        return 2
    try:
        embed, embedder_id = embedder_from_env()
        url = database_url_from_env()
        with connect(url, schema=schema_from_env()) as conn:
            report = PostgresStore(conn, slug).sync(chunks, embed, embedder_id, allow_missing_types=args.allow_missing_types,
                                                   source_files=files)
    except ConfigError as problem:
        print(f"Configuration problem: {problem}", file=sys.stderr)
        return 2
    except psycopg.OperationalError as problem:
        print(f"Cannot reach the database ({redact_url(database_url_from_env())}). Is it running? "
              f"Try: docker compose up -d --wait db\n  {problem}", file=sys.stderr)
        return 2
    except (StoreError, psycopg.Error, ValueError) as problem:
        print(f"Ingest failed ({type(problem).__name__}): {problem}", file=sys.stderr)
        return 1
    except (TransientError, PermanentError, CircuitOpenError) as problem:
        print(f"The embedding service failed ({type(problem).__name__}): {problem}. Nothing was written.", file=sys.stderr)
        return 1

    print(f"project {slug!r}: {len(chunks)} records. inserted {report.inserted}, updated {report.updated}, "
          f"unchanged {report.unchanged}, deleted {report.deleted}; {report.embedded} texts embedded with {embedder_id}; "
          f"dataset {version(files)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
