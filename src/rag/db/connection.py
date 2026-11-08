"""DATABASE CONNECTION: where the database is and how we open a connection to it.

Settings come from RAG_DB_* environment variables (usually from .env). Only RAG_DB_PASSWORD is required;
the other parts default to the local Docker Compose database. The password never appears in an error
message or a log line: use redact_url() before printing a URL.
"""
import os
import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import quote, urlsplit

import psycopg


class ConfigError(Exception):
    """A required setting is missing or malformed."""


def database_url_from_env(env: Mapping[str, str] = os.environ) -> str:
    def setting(name: str, default: str) -> str:
        return env.get(name, "").strip() or default  # an empty value counts as "not set"

    password = env.get("RAG_DB_PASSWORD", "")
    if not password:
        raise ConfigError("RAG_DB_PASSWORD is not set. Add it to .env (see .env.example).")
    host = setting("RAG_DB_HOST", "127.0.0.1")
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"  # an IPv6 address such as ::1 must be written in brackets inside a URL
    port_text = setting("RAG_DB_PORT", "5433")
    try:
        port = int(port_text)
    except ValueError:
        raise ConfigError(f"RAG_DB_PORT must be a number, got {port_text!r}.") from None
    if not 1 <= port <= 65535:
        raise ConfigError(f"RAG_DB_PORT must be between 1 and 65535, got {port}.")
    user = setting("RAG_DB_USER", "rag")
    name = setting("RAG_DB_NAME", "rag")
    # quote() turns characters such as @ / : # in a password into %XX so they cannot break the URL.
    return f"postgresql://{quote(user, safe='')}:{quote(password, safe='')}@{host}:{port}/{quote(name, safe='')}"


def redact_url(url: str) -> str:
    """The URL with the password replaced by ***, safe to print. Meant for the URLs built above; a free-form
    key=value connection string is not understood and would be returned as it is."""
    parts = urlsplit(url)
    host = parts.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    port = f":{parts.port}" if parts.port else ""
    user = f"{parts.username}:***@" if parts.username else ""
    return f"{parts.scheme}://{user}{host}{port}{parts.path}"


def schema_from_env(env: Mapping[str, str] = os.environ) -> str | None:
    """RAG_DB_SCHEMA, if set: the schema to use instead of the database's default (the tests use it)."""
    name = env.get("RAG_DB_SCHEMA", "").strip()
    if not name:
        return None
    try:
        return validate_schema_name(name)
    except ValueError:
        raise ConfigError(f"RAG_DB_SCHEMA must be a lower-case identifier, got {name!r}.") from None


_SCHEMA_NAME = re.compile(r"[a-z_][a-z0-9_]{0,62}")


def validate_schema_name(name: str) -> str:
    """Only plain lower-case identifiers are accepted, so a schema name can never smuggle SQL."""
    if not _SCHEMA_NAME.fullmatch(name):
        raise ValueError(f"Invalid schema name: {name!r}")
    return name


def fetch_one(conn: psycopg.Connection, sql: str, params: Any = None) -> tuple[Any, ...]:
    """Run a query that always returns one row (SELECT count(*), INSERT ... RETURNING) and give back that row.
    psycopg's fetchone() may return None, which type checkers rightly refuse to unpack."""
    row = conn.execute(sql, params).fetchone()
    if row is None:
        raise RuntimeError("The database returned no row for a query that should always return one")
    return row


# What a connection that only reads (the agent, the evaluations) is held to.
QUERY_STATEMENT_TIMEOUT_MS = 30_000  # a query that runs longer is stopped
QUERY_LOCK_TIMEOUT_MS = 10_000  # waiting for a lock (for example behind a bulk load that rebuilds the vector index) longer than this fails


def connect(url: str, schema: str | None = None, connect_timeout: int = 5, read_only: bool = False,
            lock_timeout_ms: int = QUERY_LOCK_TIMEOUT_MS) -> psycopg.Connection:
    """Open a connection in autocommit mode: every statement is saved at once, and code that needs several
    statements to succeed or fail together wraps them in `with conn.transaction():`.

    schema: use this schema first (tests use a throw-away one). pgvector stays reachable through `public`.
    read_only: for code that only reads. The database itself refuses any write, and a statement or a lock wait that
    takes too long fails with an error instead of hanging; ingest and migrations leave this off."""
    settings = []
    if schema:
        settings.append(f"-c search_path={validate_schema_name(schema)},public")
    if read_only:
        settings += ["-c default_transaction_read_only=on", f"-c statement_timeout={QUERY_STATEMENT_TIMEOUT_MS}",
                     f"-c lock_timeout={int(lock_timeout_ms)}"]
    options: dict[str, Any] = {"options": " ".join(settings)} if settings else {}
    return psycopg.connect(url, autocommit=True, connect_timeout=connect_timeout, **options)
