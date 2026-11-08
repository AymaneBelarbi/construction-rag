"""Helpers for tests that talk to a real Postgres. Unit tests never import this.

Integration tests are skipped (not failed) when the database is not running, with a message saying how to
start it. Each test gets its own throw-away schema inside the real database, so tests cannot see each
other's data and nothing is left behind."""
import os
import unittest
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import psycopg

from rag.db.connection import ConfigError, connect, database_url_from_env

_available: bool | None = None


def env_file_values(path: Path = Path(".env"), prefix: str = "RAG_DB_") -> dict[str, str]:
    """Read only the RAG_DB_* settings from .env WITHOUT touching os.environ (loading the whole file would
    put the real API keys into the test process)."""
    values: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip().startswith(prefix):
                values[key.strip()] = value.strip().strip('"').strip("'")
    return values


TEST_DATABASE = "rag_test"
_LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def _server_url() -> str | None:
    """URL of the local server's main database, built from RAG_DB_* (environment first, then .env)."""
    settings = {**env_file_values(), **{k: v for k, v in os.environ.items() if k.startswith("RAG_DB_")}}
    try:
        return database_url_from_env(settings)
    except ConfigError:
        return None


def database_url_for_tests() -> str | None:
    """The database the integration tests use: RAG_TEST_DATABASE_URL if you set one, else a separate
    `rag_test` database on the LOCAL server, so the tests can never touch the real data. A non-local
    host is refused unless you give RAG_TEST_DATABASE_URL on purpose. None means 'no database configured'."""
    explicit = os.environ.get("RAG_TEST_DATABASE_URL")
    if explicit:
        return explicit
    server = _server_url()
    if server is None:
        return None
    parts = urlsplit(server)
    if parts.hostname not in _LOOPBACK:
        return None
    return urlunsplit(parts._replace(path="/" + TEST_DATABASE))


def database_available() -> bool:
    """True when the test database accepts connections and has pgvector. Checked once per test run."""
    global _available
    if _available is None:
        _available = False
        url = database_url_for_tests()
        if url:
            try:
                if not os.environ.get("RAG_TEST_DATABASE_URL"):
                    # Create the throw-away database on first use (the server's main database stays untouched).
                    with psycopg.connect(str(_server_url()), connect_timeout=3, autocommit=True) as admin:
                        if not admin.execute("SELECT 1 FROM pg_database WHERE datname = %s", (TEST_DATABASE,)).fetchone():
                            admin.execute(f"CREATE DATABASE {TEST_DATABASE}")
                with psycopg.connect(url, connect_timeout=3, autocommit=True) as conn:
                    conn.execute("CREATE EXTENSION IF NOT EXISTS vector SCHEMA public")
                _available = True
            except psycopg.Error:
                _available = False
    return _available


def skip_unless(condition: bool, reason: str, strict: bool | None = None) -> Callable[[Any], Any]:
    """unittest.skipUnless, except in strict mode (RAG_REQUIRE_FULL=1, which CI sets): there a missing database or
    missing sample data is not skipped but left to fail, so a broken CI service cannot turn the run green."""
    if strict is None:
        strict = os.environ.get("RAG_REQUIRE_FULL") == "1"
    if strict:
        return lambda item: item
    return unittest.skipUnless(condition, reason)


requires_database = skip_unless(
    database_available(),
    "Postgres is not available: run `docker compose up -d db` and set RAG_DB_PASSWORD in .env",
)


@contextmanager
def temporary_schema() -> Iterator[tuple[psycopg.Connection, str]]:
    """A connection whose search_path is a brand-new schema; the schema is dropped afterwards."""
    url = database_url_for_tests()
    assert url is not None
    schema = "test_" + uuid.uuid4().hex[:12]
    with psycopg.connect(url, autocommit=True) as admin:
        admin.execute(f'CREATE SCHEMA "{schema}"')
    conn = connect(url, schema=schema)
    try:
        yield conn, schema
    finally:
        conn.close()
        with psycopg.connect(url, autocommit=True) as admin:
            admin.execute(f'DROP SCHEMA "{schema}" CASCADE')
