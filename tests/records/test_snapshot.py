import shutil
import tempfile
import unittest
from pathlib import Path

from rag.agent import make_cpr_runner
from rag.db.connection import connect
from rag.db.migrate import apply_migrations
from rag.providers.embed import hashing_embed
from rag.records.snapshot import capture, materialize, version
from rag.storage.pgstore import PostgresStore
from rag.storage.store import load_chunks, load_or_build
from rag.storage.store_api import StoreError
from tests.support.db_support import database_url_for_tests, requires_database, temporary_schema

SAMPLE = Path("data/records/samples/library-renovation")


class SnapshotTests(unittest.TestCase):
    def test_checker_uses_captured_data_after_original_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "project"
            shutil.copytree(SAMPLE, folder)
            files = capture(folder)
            checker = make_cpr_runner(Path("data/records"), folder, files)
            original = checker("2026-09-20").splitlines()[1:]
            (folder / "rfis.csv").write_text("rfi_number\n", encoding="utf-8")
            self.assertEqual(checker("2026-09-20").splitlines()[1:], original)
            self.assertNotEqual(version(capture(folder)), version(files))

    def test_memory_index_keeps_the_files_used_to_create_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "project"
            shutil.copytree(SAMPLE, folder)
            index = load_or_build(folder, hashing_embed, "hashing", Path(directory) / "cache.json")
            files = capture(folder)
            self.assertEqual(index.source_files, files)
            (folder / "rfis.csv").unlink()
            self.assertEqual(index.source_files, files)


@requires_database
class DatabaseSnapshotTests(unittest.TestCase):
    def test_concurrent_ingest_does_not_change_an_in_progress_read(self) -> None:
        files = capture(SAMPLE)
        changed = dict(files)
        changed["rfis.csv"] = files["rfis.csv"].replace("Existing beam size", "Changed beam size", 1)
        with temporary_schema() as (reader, schema):
            apply_migrations(reader, embedding_dim=256)
            store = PostgresStore(reader, "snapshot-test")
            store.sync(load_chunks(SAMPLE), hashing_embed, "hashing", source_files=files)
            url = database_url_for_tests()
            assert url is not None
            with reader.transaction():
                reader.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
                self.assertEqual(store.source_snapshot()[0], version(files))  # type: ignore[index]
                self.assertIn("Existing beam size", store.get("RFI-001")[0].text)
                with materialize(changed) as folder, connect(url, schema=schema) as writer:
                    PostgresStore(writer, "snapshot-test").sync(load_chunks(folder), hashing_embed, "hashing", source_files=changed)
                self.assertEqual(store.source_snapshot()[0], version(files))  # type: ignore[index]
                self.assertIn("Existing beam size", store.get("RFI-001")[0].text)
            self.assertEqual(store.source_snapshot()[0], version(changed))  # type: ignore[index]
            self.assertIn("Changed beam size", store.get("RFI-001")[0].text)

    def test_snapshot_must_match_indexed_chunks(self) -> None:
        files = capture(SAMPLE)
        files["rfis.csv"] = files["rfis.csv"].replace("Existing beam size", "Changed beam size", 1)
        with temporary_schema() as (conn, _):
            apply_migrations(conn, embedding_dim=256)
            store = PostgresStore(conn, "snapshot-test")
            with self.assertRaises(StoreError):
                store.sync(load_chunks(SAMPLE), hashing_embed, "hashing", source_files=files)
            row = conn.execute("SELECT count(*) FROM projects").fetchone()
            assert row is not None
            self.assertEqual(row[0], 0)

    def test_sync_without_source_files_invalidates_the_previous_snapshot(self) -> None:
        files = capture(SAMPLE)
        chunks = load_chunks(SAMPLE)
        with temporary_schema() as (conn, _):
            apply_migrations(conn, embedding_dim=256)
            store = PostgresStore(conn, "snapshot-test")
            store.sync(chunks, hashing_embed, "hashing", source_files=files)
            self.assertIsNotNone(store.source_snapshot())
            store.sync(chunks, hashing_embed, "hashing")
            self.assertIsNone(store.source_snapshot())

    def test_every_ingest_changes_generation_even_when_source_is_identical(self) -> None:
        files = capture(SAMPLE)
        chunks = load_chunks(SAMPLE)
        with temporary_schema() as (conn, _):
            apply_migrations(conn, embedding_dim=256)
            store = PostgresStore(conn, "snapshot-test")
            store.sync(chunks, hashing_embed, "hashing", source_files=files)
            before = store.source_generation()
            store.sync(chunks, hashing_embed, "hashing", source_files=files)
            self.assertNotEqual(before, store.source_generation())
