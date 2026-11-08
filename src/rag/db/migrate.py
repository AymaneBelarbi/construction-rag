"""MIGRATIONS: the database schema, as numbered SQL files applied in order, each exactly once.

    python -m rag.db.migrate            create or update the tables
    python -m rag.db.migrate --help

How it works:
- Files in migrations/ are named NNNN_description.sql, numbered 0001, 0002 ... without gaps.
- Applied versions are recorded in the schema_migrations table together with a checksum of the file.
  Editing a migration after it was applied is an error: add a new file instead.
- Each migration runs in one transaction. If any statement fails, nothing from that file is kept.
- An advisory lock makes two processes starting at the same moment take turns instead of colliding.
- The embedding size is fixed when the vector column is created, so asking for a different size on an
  existing database is refused (it needs a new migration and a re-embed of every chunk).
"""
import argparse
import hashlib
import os
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import psycopg

from rag.db.connection import ConfigError, connect, database_url_from_env, fetch_one, redact_url, schema_from_env
from rag.providers.llm import load_env

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
DEFAULT_EMBEDDING_DIM = 1536  # OpenAI text-embedding-3-small
MAX_INDEXABLE_DIM = 2000  # pgvector cannot put an HNSW index on a plain vector column wider than this

_FILE_NAME = re.compile(r"(\d{4})_([a-z0-9_]+)\.sql")
_PLACEHOLDER = re.compile(r"\{\{[^}]*\}\}")

_BOOTSTRAP_SQL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version     text PRIMARY KEY,
    name        text NOT NULL,
    checksum    text NOT NULL,
    applied_at  timestamptz NOT NULL DEFAULT now()
)
"""
# One lock per schema, so unrelated schemas (the tests use many) never wait for each other.
_LOCK_KEY = "hashtext('rag_migrations:' || current_schema())"


class MigrationError(Exception):
    """The migrations cannot be applied safely."""


@dataclass(frozen=True)
class Migration:
    version: str
    name: str
    sql: str  # the file as written, with {{placeholders}}
    checksum: str


def checksum(sql: str) -> str:
    # Line endings are normalised first, so a checkout that turned \n into \r\n (Windows) does not look
    # like an edited migration.
    normalised = sql.replace("\r\n", "\n")
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def discover_migrations(directory: Path = MIGRATIONS_DIR) -> list[Migration]:
    found: list[Migration] = []
    for path in sorted(Path(directory).glob("*.sql")):
        match = _FILE_NAME.fullmatch(path.name)
        if not match:
            raise MigrationError(f"Bad migration file name {path.name!r}: expected NNNN_description.sql")
        sql = path.read_text(encoding="utf-8")
        found.append(Migration(match.group(1), match.group(2), sql, checksum(sql)))
    versions = [m.version for m in found]
    if len(versions) != len(set(versions)):
        raise MigrationError(f"Two migration files share a version number: {versions}")
    for position, version in enumerate(versions, start=1):
        if int(version) != position:
            raise MigrationError(f"Migration numbers must run 0001, 0002 ... without gaps; found {versions}")
    return found


def render(sql: str, embedding_dim: int) -> str:
    """Fill in {{EMBEDDING_DIM}}. Any other {{placeholder}} is an error, never silently left in the SQL."""
    if type(embedding_dim) is not int or not 1 <= embedding_dim <= MAX_INDEXABLE_DIM:
        raise ValueError(f"embedding_dim must be a whole number from 1 to {MAX_INDEXABLE_DIM}, got {embedding_dim!r}")
    rendered = sql.replace("{{EMBEDDING_DIM}}", str(embedding_dim))
    leftover = _PLACEHOLDER.search(rendered)
    if leftover:
        raise MigrationError(f"Unknown placeholder {leftover.group(0)} in a migration")
    return rendered


def _check_history(applied: dict[str, str], migrations: list[Migration]) -> None:
    on_disk = {m.version: m for m in migrations}
    for version, stored_checksum in sorted(applied.items()):
        if version not in on_disk:
            raise MigrationError(f"Migration {version} is recorded as applied but its file is missing")
        if on_disk[version].checksum != stored_checksum:
            raise MigrationError(f"Migration {version} changed after it was applied; add a new migration instead")


def _check_embedding_dim(conn: psycopg.Connection, requested: int) -> None:
    # Look in THIS schema only, so a table of the same name in another schema cannot confuse the check.
    (exists,) = fetch_one(conn, "SELECT to_regclass(format('%I.rag_settings', current_schema())) IS NOT NULL")
    if not exists:
        return
    row = conn.execute("SELECT value FROM rag_settings WHERE key = 'embedding_dim'").fetchone()
    if row and int(row[0]) != requested:
        raise MigrationError(
            f"embedding_dim is {row[0]} in this database but {requested} was requested. Changing it needs a "
            "new migration and a re-embed of every chunk."
        )


def apply_migrations(conn: psycopg.Connection, embedding_dim: int, directory: Path = MIGRATIONS_DIR) -> list[str]:
    """Apply every migration not yet applied. Returns the versions applied by this call (often none).

    The connection must be in autocommit mode (connection.connect() does that). Otherwise the per-file
    transactions would only be savepoints inside an outer transaction that nobody commits, and this function
    would report success for changes that are then lost."""
    if not conn.autocommit:
        raise ValueError("apply_migrations needs an autocommit connection (use rag.db.connection.connect)")
    migrations = discover_migrations(directory)
    render("", embedding_dim)  # validate the dimension before touching the database
    conn.execute(f"SELECT pg_advisory_lock({_LOCK_KEY})")
    try:
        with conn.transaction():
            conn.execute(_BOOTSTRAP_SQL)
        applied = {row[0]: row[1] for row in conn.execute("SELECT version, checksum FROM schema_migrations")}
        _check_history(applied, migrations)
        _check_embedding_dim(conn, embedding_dim)

        newly_applied: list[str] = []
        for migration in migrations:
            if migration.version in applied:
                continue
            try:
                with conn.transaction():  # all of this file, or none of it
                    conn.execute(render(migration.sql, embedding_dim))
                    conn.execute(
                        "INSERT INTO schema_migrations (version, name, checksum) VALUES (%s, %s, %s)",
                        (migration.version, migration.name, migration.checksum),
                    )
            except psycopg.Error as error:
                raise MigrationError(
                    f"Migration {migration.version}_{migration.name} failed and was rolled back: {error}"
                ) from error
            newly_applied.append(migration.version)
        return newly_applied
    finally:
        if not conn.closed:  # on a dead connection the lock is gone already, and this must not hide the real error
            try:
                conn.execute(f"SELECT pg_advisory_unlock({_LOCK_KEY})")
            except psycopg.Error:
                pass


def resolve_embedding_dim(cli_value: int | None, env: Mapping[str, str]) -> int:
    """The vector size to use: the command-line value, else RAG_EMBEDDING_DIM, else the default."""
    if cli_value is not None:
        value: object = cli_value
    else:
        text = env.get("RAG_EMBEDDING_DIM", "").strip()
        if not text:
            return DEFAULT_EMBEDDING_DIM
        try:
            value = int(text)
        except ValueError:
            raise ConfigError(f"RAG_EMBEDDING_DIM must be a whole number, got {text!r}.") from None
    if type(value) is not int or not 1 <= value <= MAX_INDEXABLE_DIM:
        raise ConfigError(f"The embedding size must be a whole number from 1 to {MAX_INDEXABLE_DIM}, got {value!r}.")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create or update the database tables.")
    parser.add_argument("--embedding-dim", type=int, default=None,
                        help="vector size of the embedding model (default 1536; or set RAG_EMBEDDING_DIM)")
    args = parser.parse_args(argv)

    load_env()  # first, so RAG_EMBEDDING_DIM and RAG_DB_* written in .env are seen
    try:
        embedding_dim = resolve_embedding_dim(args.embedding_dim, os.environ)
        url = database_url_from_env()
        with connect(url, schema=schema_from_env()) as conn:
            applied = apply_migrations(conn, embedding_dim)
    except ConfigError as problem:
        print(f"Configuration problem: {problem}", file=sys.stderr)
        return 2
    except psycopg.OperationalError as problem:
        print(f"Cannot reach the database ({redact_url(database_url_from_env())}). Is it running? "
              f"Try: docker compose up -d --wait db\n  {problem}", file=sys.stderr)
        return 2
    except MigrationError as problem:
        print(f"Migration problem: {problem}", file=sys.stderr)
        return 1
    print(f"{redact_url(url)}: " + (f"applied {', '.join(applied)}" if applied else "already up to date"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
