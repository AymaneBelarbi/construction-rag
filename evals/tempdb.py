"""A throw-away, migrated PostgreSQL schema for evaluations that need the real store.

It lives in a separate `rag_test` database on the LOCAL server, so nothing of your real data is touched, and it
is dropped afterwards. Refuses a non-local host."""
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from urllib.parse import urlsplit, urlunsplit

import psycopg

from rag.db.connection import connect, database_url_from_env
from rag.db.migrate import apply_migrations
from rag.storage.store_api import StoreError

LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")


@contextmanager
def temporary_database_schema(embedding_dim: int) -> Iterator[psycopg.Connection]:
    server = urlsplit(database_url_from_env())
    if server.hostname not in LOCAL_HOSTS:
        raise StoreError("This only runs against a local database. Use an ingested project on a real one.")
    test_url = urlunsplit(server._replace(path="/rag_test"))
    with psycopg.connect(database_url_from_env(), autocommit=True, connect_timeout=5) as admin:
        if not admin.execute("SELECT 1 FROM pg_database WHERE datname = 'rag_test'").fetchone():
            admin.execute("CREATE DATABASE rag_test")
    with psycopg.connect(test_url, autocommit=True, connect_timeout=5) as admin:
        admin.execute("CREATE EXTENSION IF NOT EXISTS vector SCHEMA public")
        schema = "eval_" + uuid.uuid4().hex[:12]
        admin.execute(f'CREATE SCHEMA "{schema}"')
        try:
            with connect(test_url, schema=schema) as conn:
                apply_migrations(conn, embedding_dim)
                yield conn
        finally:
            admin.execute(f'DROP SCHEMA "{schema}" CASCADE')
