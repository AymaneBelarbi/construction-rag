import re
import tempfile
import unittest
from pathlib import Path

from rag.db.connection import (
    ConfigError,
    database_url_from_env,
    fetch_one,
    redact_url,
    schema_from_env,
    validate_schema_name,
)
from rag.db.migrate import (
    MIGRATIONS_DIR,
    MigrationError,
    checksum,
    discover_migrations,
    render,
    resolve_embedding_dim,
)
from rag.storage.pgstore import BULK_INDEX_MEMORY_MB


class DatabaseUrlTests(unittest.TestCase):
    def test_builds_the_local_compose_url_from_defaults(self) -> None:
        url = database_url_from_env({"RAG_DB_PASSWORD": "secret"})
        self.assertEqual(url, "postgresql://rag:secret@127.0.0.1:5433/rag")

    def test_every_part_can_be_overridden(self) -> None:
        env = {"RAG_DB_PASSWORD": "pw", "RAG_DB_HOST": "db.internal", "RAG_DB_PORT": "6000",
               "RAG_DB_NAME": "prod", "RAG_DB_USER": "app"}
        self.assertEqual(database_url_from_env(env), "postgresql://app:pw@db.internal:6000/prod")

    def test_special_characters_in_the_password_are_percent_encoded(self) -> None:
        url = database_url_from_env({"RAG_DB_PASSWORD": "p@ss/w:rd#1"})
        self.assertIn("p%40ss%2Fw%3Ard%231", url)
        self.assertEqual(url.count("@"), 1)  # only the separator before the host

    def test_a_missing_password_is_an_error_that_says_what_to_set(self) -> None:
        with self.assertRaises(ConfigError) as caught:
            database_url_from_env({})
        self.assertIn("RAG_DB_PASSWORD", str(caught.exception))

    def test_a_bad_port_is_a_clear_error_that_does_not_leak_the_password(self) -> None:
        with self.assertRaises(ConfigError) as caught:
            database_url_from_env({"RAG_DB_PASSWORD": "topsecret", "RAG_DB_PORT": "abc"})
        self.assertNotIn("topsecret", str(caught.exception))

    def test_an_ipv6_host_is_written_in_brackets(self) -> None:
        url = database_url_from_env({"RAG_DB_PASSWORD": "pw", "RAG_DB_HOST": "::1"})
        self.assertEqual(url, "postgresql://rag:pw@[::1]:5433/rag")
        self.assertIn("[::1]:5433", redact_url(url))

    def test_empty_settings_count_as_not_set(self) -> None:
        env = {"RAG_DB_PASSWORD": "pw", "RAG_DB_HOST": "", "RAG_DB_PORT": "  ", "RAG_DB_USER": "", "RAG_DB_NAME": ""}
        self.assertEqual(database_url_from_env(env), "postgresql://rag:pw@127.0.0.1:5433/rag")

    def test_the_schema_setting_is_optional_and_validated(self) -> None:
        self.assertIsNone(schema_from_env({}))
        self.assertIsNone(schema_from_env({"RAG_DB_SCHEMA": "  "}))
        self.assertEqual(schema_from_env({"RAG_DB_SCHEMA": "rag_dev"}), "rag_dev")
        with self.assertRaises(ConfigError):
            schema_from_env({"RAG_DB_SCHEMA": "bad; name"})

    def test_redact_hides_the_password_but_keeps_the_rest(self) -> None:
        redacted = redact_url("postgresql://rag:topsecret@127.0.0.1:5433/rag")
        self.assertNotIn("topsecret", redacted)
        self.assertIn("127.0.0.1:5433/rag", redacted)

    def test_schema_names_are_restricted_to_safe_identifiers(self) -> None:
        self.assertEqual(validate_schema_name("test_ab12"), "test_ab12")
        for bad in ("", "a;drop table x", 'a"b', "a b", "1abc", "x" * 64):
            with self.subTest(name=bad):
                with self.assertRaises(ValueError):
                    validate_schema_name(bad)


class RenderTests(unittest.TestCase):
    def test_substitutes_the_embedding_dimension(self) -> None:
        self.assertEqual(render("vector({{EMBEDDING_DIM}})", 1536), "vector(1536)")

    def test_rejects_dimensions_pgvector_cannot_index(self) -> None:
        for bad in (0, -1, 2001, "1536", 1.5, True):
            with self.subTest(dim=bad):
                with self.assertRaises(ValueError):
                    render("vector({{EMBEDDING_DIM}})", bad)  # type: ignore[arg-type]

    def test_an_unknown_placeholder_is_an_error_not_silently_left_in(self) -> None:
        with self.assertRaises(MigrationError):
            render("select {{SOMETHING_ELSE}}", 256)

    def test_sql_without_placeholders_is_unchanged(self) -> None:
        self.assertEqual(render("SELECT 1;", 256), "SELECT 1;")


class EmbeddingDimTests(unittest.TestCase):
    def test_the_command_line_wins_over_the_environment_which_wins_over_the_default(self) -> None:
        self.assertEqual(resolve_embedding_dim(256, {"RAG_EMBEDDING_DIM": "512"}), 256)
        self.assertEqual(resolve_embedding_dim(None, {"RAG_EMBEDDING_DIM": "512"}), 512)
        self.assertEqual(resolve_embedding_dim(None, {}), 1536)
        self.assertEqual(resolve_embedding_dim(None, {"RAG_EMBEDDING_DIM": " "}), 1536)

    def test_bad_values_are_a_clear_config_error_not_a_traceback(self) -> None:
        for cli, env in ((None, {"RAG_EMBEDDING_DIM": "lots"}), (3000, {}), (0, {}), (None, {"RAG_EMBEDDING_DIM": "-5"})):
            with self.subTest(cli=cli, env=env):
                with self.assertRaises(ConfigError):
                    resolve_embedding_dim(cli, env)


class DiscoverTests(unittest.TestCase):
    def write(self, directory: Path, files: dict[str, str]) -> None:
        for name, sql in files.items():
            (directory / name).write_text(sql, encoding="utf-8")

    def test_migrations_are_found_in_version_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.write(Path(tmp), {"0002_second.sql": "select 2;", "0001_first.sql": "select 1;"})
            found = discover_migrations(Path(tmp))
        self.assertEqual([m.version for m in found], ["0001", "0002"])
        self.assertEqual(found[0].name, "first")

    def test_badly_named_files_are_rejected_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.write(Path(tmp), {"0001_ok.sql": "select 1;", "notes.sql": "select 2;"})
            with self.assertRaises(MigrationError):
                discover_migrations(Path(tmp))

    def test_duplicate_versions_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.write(Path(tmp), {"0001_a.sql": "select 1;", "0001_b.sql": "select 2;"})
            with self.assertRaises(MigrationError):
                discover_migrations(Path(tmp))

    def test_a_gap_in_the_numbering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.write(Path(tmp), {"0001_a.sql": "select 1;", "0003_c.sql": "select 3;"})
            with self.assertRaises(MigrationError):
                discover_migrations(Path(tmp))

    def test_checksum_ignores_windows_line_endings(self) -> None:
        self.assertEqual(checksum("select 1;\r\nselect 2;\r\n"), checksum("select 1;\nselect 2;\n"))

    def test_checksum_changes_when_the_sql_changes(self) -> None:
        self.assertEqual(checksum("select 1;"), checksum("select 1;"))
        self.assertNotEqual(checksum("select 1;"), checksum("select 2;"))


class ShippedMigrationTests(unittest.TestCase):
    def test_the_shipped_migrations_load_and_render_for_the_real_dimension(self) -> None:
        migrations = discover_migrations(MIGRATIONS_DIR)
        self.assertGreaterEqual(len(migrations), 1)
        for migration in migrations:
            with self.subTest(migration=migration.version):
                self.assertNotIn("{{", render(migration.sql, 1536))

    def test_the_initial_migration_defines_the_core_tables(self) -> None:
        sql = discover_migrations(MIGRATIONS_DIR)[0].sql.lower()
        for table in ("projects", "chunks", "nodes", "edges", "rag_settings"):
            with self.subTest(table=table):
                self.assertIn(f"create table {table}", sql)
        self.assertIn("using hnsw", sql)


class FetchOneTests(unittest.TestCase):
    class FakeConnection:
        def __init__(self, row: tuple | None) -> None:
            self.row = row
            self.seen: tuple[str, object] | None = None

        def execute(self, sql: str, params: object = None) -> FetchOneTests.FakeConnection:
            self.seen = (sql, params)
            return self

        def fetchone(self) -> tuple | None:
            return self.row

    def test_it_returns_the_row_and_passes_the_query_on(self) -> None:
        conn = self.FakeConnection((7, "x"))
        self.assertEqual(fetch_one(conn, "SELECT 1", (1,)), (7, "x"))  # type: ignore[arg-type]
        self.assertEqual(conn.seen, ("SELECT 1", (1,)))

    def test_a_query_that_returns_nothing_is_an_error_not_a_none(self) -> None:
        with self.assertRaises(RuntimeError):
            fetch_one(self.FakeConnection(None), "SELECT 1")  # type: ignore[arg-type]


class ComposeSharedMemoryTests(unittest.TestCase):
    def test_the_container_has_more_shared_memory_than_a_bulk_index_build_asks_for(self) -> None:
        # Docker gives a container 64 MB of shared memory unless told otherwise, and the parallel HNSW build
        # needs as much as maintenance_work_mem: with the default the first big load failed with "No space left".
        compose = Path("docker-compose.yml").read_text(encoding="utf-8")
        match = re.search(r"^\s*shm_size:\s*(\d+)\s*(mb|gb)\s*$", compose, re.IGNORECASE | re.MULTILINE)
        assert match is not None, "docker-compose.yml must set shm_size for the db service"
        megabytes = int(match.group(1)) * (1024 if match.group(2).lower() == "gb" else 1)
        self.assertGreater(megabytes, BULK_INDEX_MEMORY_MB)


if __name__ == "__main__":
    unittest.main()
