"""Tests against a real Postgres with pgvector. Skipped when the database is not running."""
import tempfile
import threading
import unittest
from pathlib import Path

import psycopg

from rag.db.connection import connect, fetch_one
from rag.db.migrate import MIGRATIONS_DIR, MigrationError, apply_migrations
from rag.providers.embed import hashing_embed
from tests.support.db_support import database_url_for_tests, requires_database, temporary_schema


def vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vector) + "]"


def table_names(conn: psycopg.Connection, schema: str) -> set[str]:
    rows = conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = %s", (schema,))
    return {row[0] for row in rows}


def write_migrations(directory: Path, files: dict[str, str]) -> Path:
    for name, sql in files.items():
        (directory / name).write_text(sql, encoding="utf-8")
    return directory


@requires_database
class MigrationTests(unittest.TestCase):
    def test_a_fresh_schema_gets_every_table_and_index(self) -> None:
        with temporary_schema() as (conn, schema):
            self.assertEqual(apply_migrations(conn, embedding_dim=256), ["0001", "0002", "0003", "0004", "0005", "0006", "0007"])
            self.assertTrue({"projects", "chunks", "nodes", "edges", "rag_settings", "schema_migrations"}
                            <= table_names(conn, schema))
            indexes = {r[0] for r in conn.execute("SELECT indexname FROM pg_indexes WHERE schemaname = %s", (schema,))}
            for expected in ("chunks_embedding_hnsw", "chunks_search_gin", "chunks_metadata_gin", "edges_dst_src"):
                self.assertIn(expected, indexes)
            self.assertNotIn("edges_dst", indexes)  # replaced by edges_dst_src in migration 0004

    def test_running_the_migrations_again_changes_nothing(self) -> None:
        with temporary_schema() as (conn, _):
            apply_migrations(conn, embedding_dim=256)
            self.assertEqual(apply_migrations(conn, embedding_dim=256), [])

    def test_the_embedding_column_has_the_requested_dimension(self) -> None:
        with temporary_schema() as (conn, schema):
            apply_migrations(conn, embedding_dim=256)
            (column_type,) = fetch_one(conn,
                "SELECT format_type(a.atttypid, a.atttypmod) FROM pg_attribute a "
                "WHERE a.attrelid = (%s || '.chunks')::regclass AND a.attname = 'embedding'", (f'"{schema}"',)
            )
            self.assertEqual(column_type, "vector(256)")

    def test_asking_for_a_different_dimension_later_is_refused(self) -> None:
        with temporary_schema() as (conn, _):
            apply_migrations(conn, embedding_dim=256)
            with self.assertRaises(MigrationError) as caught:
                apply_migrations(conn, embedding_dim=512)
            self.assertIn("embedding_dim", str(caught.exception))

    def test_an_edited_migration_is_detected(self) -> None:
        with temporary_schema() as (conn, _):
            apply_migrations(conn, embedding_dim=256)
            conn.execute("UPDATE schema_migrations SET checksum = 'tampered' WHERE version = '0001'")
            with self.assertRaises(MigrationError) as caught:
                apply_migrations(conn, embedding_dim=256)
            self.assertIn("changed after", str(caught.exception))

    def test_an_applied_migration_whose_file_disappeared_is_detected(self) -> None:
        with temporary_schema() as (conn, _), tempfile.TemporaryDirectory() as tmp:
            both = write_migrations(Path(tmp), {"0001_a.sql": "CREATE TABLE a (x int);", "0002_b.sql": "CREATE TABLE b (x int);"})
            apply_migrations(conn, 256, directory=both)
            (both / "0002_b.sql").unlink()
            with self.assertRaises(MigrationError) as caught:
                apply_migrations(conn, 256, directory=both)
            self.assertIn("0002", str(caught.exception))

    def test_a_failing_migration_rolls_back_completely_and_is_not_recorded(self) -> None:
        with temporary_schema() as (conn, schema), tempfile.TemporaryDirectory() as tmp:
            directory = write_migrations(Path(tmp), {
                "0001_ok.sql": "CREATE TABLE good (x int);",
                "0002_bad.sql": "CREATE TABLE half_done (x int); SELECT 1/0;",
            })
            with self.assertRaises(MigrationError):
                apply_migrations(conn, 256, directory=directory)
            tables = table_names(conn, schema)
            self.assertIn("good", tables)  # the earlier migration stays applied
            self.assertNotIn("half_done", tables)  # nothing from the failed one survives
            versions = {r[0] for r in conn.execute("SELECT version FROM schema_migrations")}
            self.assertEqual(versions, {"0001"})

    def test_a_connection_that_is_not_in_autocommit_mode_is_refused(self) -> None:
        # Otherwise the per-file transactions are only savepoints inside an outer transaction that nobody
        # commits, and the function would report success for changes that are then rolled back.
        with temporary_schema() as (_, schema):
            with psycopg.connect(str(database_url_for_tests()), options=f"-c search_path={schema},public") as plain:
                with self.assertRaises(ValueError):
                    apply_migrations(plain, 256)

    def test_a_migration_and_its_record_succeed_or_fail_together(self) -> None:
        # The SQL itself works, but recording it as applied is made to fail by a trigger that the migration
        # installs. Without a surrounding transaction the table would be left behind with no record, and the
        # next run would try to create it again and crash.
        trap = (
            "CREATE TABLE made_before_failure (x int);\n"
            "CREATE FUNCTION refuse_record() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN RAISE EXCEPTION 'no'; END $$;\n"
            "CREATE TRIGGER refuse BEFORE INSERT ON schema_migrations FOR EACH ROW EXECUTE FUNCTION refuse_record();\n"
        )
        with temporary_schema() as (conn, schema), tempfile.TemporaryDirectory() as tmp:
            directory = write_migrations(Path(tmp), {"0001_trap.sql": trap})
            with self.assertRaises(MigrationError):
                apply_migrations(conn, 256, directory=directory)
            self.assertNotIn("made_before_failure", table_names(conn, schema))
            functions = {r[0] for r in conn.execute(
                "SELECT proname FROM pg_proc WHERE pronamespace = %s::regnamespace", (f'"{schema}"',))}
            self.assertNotIn("refuse_record", functions)

    def test_two_processes_migrating_at_once_apply_each_migration_exactly_once(self) -> None:
        with temporary_schema() as (conn, schema):
            url = str(database_url_for_tests())
            results: list[list[str]] = []
            errors: list[BaseException] = []

            def worker() -> None:
                try:
                    with connect(url, schema=schema) as own:
                        results.append(apply_migrations(own, 256))
                except BaseException as problem:  # collected and asserted on below
                    errors.append(problem)

            threads = [threading.Thread(target=worker) for _ in range(4)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=60)
            self.assertFalse(any(thread.is_alive() for thread in threads), "a migration worker hung")
            self.assertEqual(errors, [])
            self.assertEqual(sorted(sum(results, [])), ["0001", "0002", "0003", "0004", "0005", "0006", "0007"])  # each applied once


@requires_database
class SchemaBehaviourTests(unittest.TestCase):
    def setUp(self) -> None:
        self._cm = temporary_schema()
        self.conn, self.schema = self._cm.__enter__()
        self.addCleanup(self._cm.__exit__, None, None, None)  # runs even if the next lines fail
        apply_migrations(self.conn, embedding_dim=256)
        (self.project,) = fetch_one(self.conn, "INSERT INTO projects (slug) VALUES ('p') RETURNING id")

    def add_chunk(self, record_id: str, text: str, embedding: list[float] | None = None, revision: int = 0) -> None:
        self.conn.execute(
            "INSERT INTO chunks (project_id, record_id, revision, record_type, text, text_hash, embedding, embedder_id) "
            "VALUES (%s, %s, %s, 'rfi', %s, 'h', %s::vector, %s)",
            (self.project, record_id, revision, text,
             vector_literal(embedding) if embedding else None, "hashing" if embedding else None),
        )

    def test_a_vector_without_an_embedder_id_is_rejected(self) -> None:
        with self.assertRaises(psycopg.errors.CheckViolation):
            self.conn.execute(
                "INSERT INTO chunks (project_id, record_id, record_type, text, text_hash, embedding) "
                "VALUES (%s, 'X', 'rfi', 't', 'h', %s::vector)", (self.project, vector_literal([0.1] * 256)))

    def test_an_embedder_id_without_a_vector_is_rejected_too(self) -> None:
        with self.assertRaises(psycopg.errors.CheckViolation):
            self.conn.execute(
                "INSERT INTO chunks (project_id, record_id, record_type, text, text_hash, embedder_id) "
                "VALUES (%s, 'X', 'rfi', 't', 'h', 'model')", (self.project,))

    def test_the_same_record_and_revision_cannot_be_stored_twice(self) -> None:
        self.add_chunk("RFI-1", "first")
        with self.assertRaises(psycopg.errors.UniqueViolation):
            self.add_chunk("RFI-1", "again")

    def test_submittal_revisions_are_separate_rows(self) -> None:
        self.add_chunk("S-001", "revision zero", revision=0)
        self.add_chunk("S-001", "revision one", revision=1)
        (count,) = fetch_one(self.conn, "SELECT count(*) FROM chunks WHERE record_id = 'S-001'")
        self.assertEqual(count, 2)

    def test_dropping_a_project_removes_its_chunks_nodes_and_edges(self) -> None:
        self.add_chunk("RFI-1", "text")
        a = fetch_one(self.conn, "INSERT INTO nodes (project_id, kind, key, label) VALUES (%s, 'rfi', 'RFI-1', 'RFI-1') RETURNING id", (self.project,))[0]
        b = fetch_one(self.conn, "INSERT INTO nodes (project_id, kind, key, label) VALUES (%s, 'change_event', 'CE-1', 'CE-1') RETURNING id", (self.project,))[0]
        self.conn.execute("INSERT INTO edges (project_id, src, dst, relation) VALUES (%s, %s, %s, 'related_rfi')", (self.project, a, b))
        self.conn.execute("DELETE FROM projects WHERE id = %s", (self.project,))
        for table in ("chunks", "nodes", "edges"):
            with self.subTest(table=table):
                (count,) = fetch_one(self.conn, f"SELECT count(*) FROM {table}")
                self.assertEqual(count, 0)

    def test_an_edge_cannot_point_at_itself_or_use_a_malformed_relation(self) -> None:
        a = fetch_one(self.conn, "INSERT INTO nodes (project_id, kind, key, label) VALUES (%s, 'rfi', 'RFI-1', 'RFI-1') RETURNING id", (self.project,))[0]
        b = fetch_one(self.conn, "INSERT INTO nodes (project_id, kind, key, label) VALUES (%s, 'rfi', 'RFI-2', 'RFI-2') RETURNING id", (self.project,))[0]
        with self.assertRaises(psycopg.errors.CheckViolation):
            self.conn.execute("INSERT INTO edges (project_id, src, dst, relation) VALUES (%s, %s, %s, 'x')", (self.project, a, a))
        with self.assertRaises(psycopg.errors.CheckViolation):
            self.conn.execute("INSERT INTO edges (project_id, src, dst, relation) VALUES (%s, %s, %s, 'Bad Relation!')", (self.project, a, b))

    def test_vector_search_returns_the_nearest_chunk_first(self) -> None:
        texts = {"RFI-1": "abandoned conduit in the ceiling of the reading room",
                 "RFI-2": "interior paint colour for radiators",
                 "RFI-3": "structural support for the rooftop unit"}
        for record_id, vector in zip(texts, hashing_embed(list(texts.values())), strict=True):
            self.add_chunk(record_id, texts[record_id], vector)
        query = vector_literal(hashing_embed(["conduit in the ceiling"])[0])
        rows = self.conn.execute("SELECT record_id FROM chunks ORDER BY embedding <=> %s::vector LIMIT 3", (query,)).fetchall()
        self.assertEqual(rows[0][0], "RFI-1")

    def test_full_text_search_finds_an_exact_record_id_and_a_plain_word(self) -> None:
        self.add_chunk("RFI-003", "RFI RFI-003 (closed). Subject: Abandoned conduit found in ceiling.")
        self.add_chunk("RFI-004", "RFI RFI-004 (answered). Subject: Floor leveling compound depth.")
        by_id = self.conn.execute("SELECT record_id FROM chunks WHERE search_text @@ to_tsquery('simple', '(rfi <-> 003)')").fetchall()
        self.assertEqual([r[0] for r in by_id], ["RFI-003"])
        by_word = self.conn.execute("SELECT record_id FROM chunks WHERE search_text @@ plainto_tsquery('simple', 'leveling')").fetchall()
        self.assertEqual([r[0] for r in by_word], ["RFI-004"])

    def test_an_id_matches_only_that_record_not_one_that_merely_mentions_a_sibling(self) -> None:
        # Regression for a review finding: the default parser read RFI-003 as 'rfi' + '-003', so a record that
        # mentions "RFI-011 ... CE-003" matched the query RFI-003. IDs are now exact phrases.
        self.add_chunk("RFI-003", "RFI RFI-003 (closed). Subject: Abandoned conduit.")
        self.add_chunk("CE-009", "Change event CE-009. Related RFI: RFI-011. Also see CE-003.")
        self.add_chunk("PL-001", "Punch item PL-001 (open). Contractor's note.")
        self.add_chunk("S-001", "Submittal S-001 revision 0.")
        phrase = "SELECT record_id FROM chunks WHERE search_text @@ to_tsquery('simple', %s) ORDER BY record_id"
        self.assertEqual([r[0] for r in self.conn.execute(phrase, ("(rfi <-> 003)",))], ["RFI-003"])
        self.assertEqual([r[0] for r in self.conn.execute(phrase, ("(s <-> 001)",))], ["S-001"])  # not PL-001

    def test_a_daily_report_is_found_by_its_id_even_though_the_id_is_not_in_its_text(self) -> None:
        self.add_chunk("DR-2026-08-03", "Daily report 2026-08-03. Weather: Dry.")
        rows = self.conn.execute("SELECT record_id FROM chunks WHERE search_text @@ to_tsquery('simple', %s)",
                                 ("(dr <-> 2026 <-> 08 <-> 03)",)).fetchall()
        self.assertEqual([r[0] for r in rows], ["DR-2026-08-03"])

    def test_an_edge_cannot_join_nodes_of_different_projects(self) -> None:
        (other,) = fetch_one(self.conn, "INSERT INTO projects (slug) VALUES ('other') RETURNING id")
        mine = fetch_one(self.conn, "INSERT INTO nodes (project_id, kind, key, label) VALUES (%s, 'rfi', 'RFI-1', 'RFI-1') RETURNING id", (self.project,))[0]
        theirs = fetch_one(self.conn, "INSERT INTO nodes (project_id, kind, key, label) VALUES (%s, 'rfi', 'RFI-2', 'RFI-2') RETURNING id", (other,))[0]
        with self.assertRaises(psycopg.errors.ForeignKeyViolation):
            self.conn.execute("INSERT INTO edges (project_id, src, dst, relation) VALUES (%s, %s, %s, 'related_rfi')", (self.project, mine, theirs))

    def test_a_metadata_key_exists_filter_works(self) -> None:
        self.conn.execute(
            "INSERT INTO chunks (project_id, record_id, record_type, text, text_hash, metadata) "
            "VALUES (%s, 'RFI-8', 'rfi', 't', 'h', %s::jsonb)", (self.project, '{"injection_suspected": "yes"}'))
        rows = self.conn.execute("SELECT record_id FROM chunks WHERE metadata ? 'injection_suspected'").fetchall()
        self.assertEqual([r[0] for r in rows], ["RFI-8"])

    def test_a_filtered_vector_search_in_a_big_table_still_returns_k_rows(self) -> None:
        # Regression for a review finding: with the HNSW index and a filter that keeps 1% of rows, a plain
        # query returned 1 row instead of 10, because the index only looks at its 40 nearest neighbours before
        # the filter is applied. PostgresStore sets hnsw.iterative_scan so the search keeps going.
        import random

        from rag.storage.pgstore import PostgresStore

        rng = random.Random(7)
        rows = []
        for i in range(6000):
            vector = [rng.random() - 0.5 for _ in range(256)]
            rows.append((self.project, f"BIG-{i}", 1 if i % 100 == 0 else 0, "rfi", "text", "h", vector_literal(vector), "m"))
        with self.conn.cursor() as cursor:
            cursor.executemany(
                "INSERT INTO chunks (project_id, record_id, revision, record_type, text, text_hash, embedding, embedder_id) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s::vector, %s)", rows)
        self.conn.execute("ANALYZE chunks")
        self.conn.execute("SET enable_seqscan = off")  # make sure the planner really uses the HNSW index
        store = PostgresStore(self.conn, "p")
        self.conn.execute("UPDATE chunks SET metadata = jsonb_build_object('rare', (revision = 1)::text)")
        query = [rng.random() - 0.5 for _ in range(256)]
        hits = store.vector_search(query, limit=10, filters={"rare": "true"})
        self.assertEqual(len(hits), 10)

    def test_the_metadata_filter_works_on_jsonb(self) -> None:
        self.conn.execute(
            "INSERT INTO chunks (project_id, record_id, record_type, text, text_hash, metadata) "
            "VALUES (%s, 'RFI-9', 'rfi', 't', 'h', %s::jsonb)", (self.project, '{"status": "open"}'))
        rows = self.conn.execute("SELECT record_id FROM chunks WHERE metadata @> %s::jsonb", ('{"status": "open"}',)).fetchall()
        self.assertEqual([r[0] for r in rows], ["RFI-9"])


@requires_database
class ShippedMigrationTests(unittest.TestCase):
    def test_the_shipped_migrations_apply_at_the_real_embedding_size(self) -> None:
        with temporary_schema() as (conn, _):
            self.assertEqual(apply_migrations(conn, embedding_dim=1536), ["0001", "0002", "0003", "0004", "0005", "0006", "0007"])

    def test_the_default_directory_is_the_one_next_to_the_code(self) -> None:
        self.assertTrue((MIGRATIONS_DIR / "0001_initial.sql").exists())


@requires_database
class QueryConnectionTests(unittest.TestCase):
    """The agent only reads. The database is told so, instead of leaving that to a code review, and a reader is told
    not to wait for ever behind a long ingest."""

    def test_a_read_only_connection_can_read_and_cannot_write(self) -> None:
        with temporary_schema() as (setup, schema):
            apply_migrations(setup, 256)
            with connect(str(database_url_for_tests()), schema=schema, read_only=True) as reader:
                self.assertEqual(fetch_one(reader, "SELECT count(*) FROM projects"), (0,))
                with self.assertRaises(psycopg.errors.ReadOnlySqlTransaction):
                    reader.execute("INSERT INTO projects (slug) VALUES ('x')")
                with self.assertRaises(psycopg.errors.ReadOnlySqlTransaction):
                    reader.execute("DELETE FROM projects")
                with self.assertRaises(psycopg.errors.ReadOnlySqlTransaction):
                    reader.execute("DROP TABLE projects")

    def test_a_read_only_connection_gives_up_on_a_long_query_and_a_long_lock_wait(self) -> None:
        with temporary_schema() as (_, schema):
            with connect(str(database_url_for_tests()), schema=schema, read_only=True) as reader:
                self.assertEqual(fetch_one(reader, "SHOW statement_timeout"), ("30s",))
                self.assertEqual(fetch_one(reader, "SHOW lock_timeout"), ("10s",))

    def test_a_reader_blocked_by_a_long_write_fails_after_the_lock_timeout_instead_of_hanging(self) -> None:
        with temporary_schema() as (setup, schema):
            apply_migrations(setup, 256)
            url = str(database_url_for_tests())
            with connect(url, schema=schema) as writer, connect(url, schema=schema, read_only=True, lock_timeout_ms=300) as reader:
                with writer.transaction():
                    writer.execute("LOCK TABLE projects IN ACCESS EXCLUSIVE MODE")  # what a bulk-load index drop does to a table
                    with self.assertRaises(psycopg.errors.LockNotAvailable):
                        reader.execute("SELECT count(*) FROM projects")

    def test_an_ordinary_connection_stays_writable_and_untimed(self) -> None:
        with temporary_schema() as (conn, _):
            self.assertEqual(fetch_one(conn, "SHOW statement_timeout"), ("0",))
            self.assertEqual(fetch_one(conn, "SHOW default_transaction_read_only"), ("off",))


if __name__ == "__main__":
    unittest.main()
