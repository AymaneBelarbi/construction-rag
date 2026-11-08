import io
import unittest
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from typing import Any
from unittest import mock

from rag.cli.ingest import main
from rag.db.connection import fetch_one
from rag.db.migrate import apply_migrations
from rag.infra.resilience import TransientError
from rag.providers.embed import hashing_embed
from tests.support.db_support import database_url_for_tests, requires_database, temporary_schema


@requires_database
class IngestCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self._schema = temporary_schema()
        self.conn, self.schema = self._schema.__enter__()
        self.addCleanup(self._schema.__exit__, None, None, None)
        apply_migrations(self.conn, embedding_dim=256)
        self.url = database_url_for_tests()

    def run_ingest(self, argv: list[str] | None = None, embedder: Any = (hashing_embed, "hashing"),
                   url: str | None = None) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with ExitStack() as stack:
            stack.enter_context(mock.patch("rag.cli.ingest.load_env"))
            stack.enter_context(mock.patch("rag.cli.ingest.database_url_from_env", return_value=url or self.url))
            stack.enter_context(mock.patch("rag.cli.ingest.embedder_from_env", return_value=embedder))
            stack.enter_context(mock.patch.dict("os.environ", {"RAG_DB_SCHEMA": self.schema}))
            stack.enter_context(redirect_stdout(out))
            stack.enter_context(redirect_stderr(err))
            code = main(argv or [])
        return code, out.getvalue(), err.getvalue()

    def chunk_count(self) -> int:
        return fetch_one(self.conn, "SELECT count(*) FROM chunks")[0]

    def test_first_run_loads_everything_and_second_run_changes_nothing(self) -> None:
        code, out, err = self.run_ingest()
        self.assertEqual(code, 0, err)
        self.assertIn("inserted 88", out)
        self.assertIn("88 texts embedded", out)
        code, out, err = self.run_ingest()
        self.assertEqual(code, 0, err)
        self.assertIn("unchanged 88", out)
        self.assertIn("0 texts embedded", out)
        self.assertEqual(self.chunk_count(), 88)

    def project_without_rfis(self) -> str:
        import shutil
        import tempfile
        from pathlib import Path

        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, True)
        shutil.copytree("data/records/samples/library-renovation", Path(root) / "proj")
        (Path(root) / "proj" / "rfis.csv").unlink()
        return root

    def test_a_folder_that_lost_a_whole_csv_is_refused_instead_of_deleting_those_records(self) -> None:
        self.run_ingest()
        root = self.project_without_rfis()
        code, out, err = self.run_ingest(["--records", root, "--project", "proj", "--slug", "library-renovation"])
        self.assertEqual(code, 1)
        self.assertIn("rfi", err)
        self.assertIn("--allow-missing-types", err)
        self.assertEqual(self.chunk_count(), 88)  # nothing was deleted

    def test_the_operator_can_confirm_that_a_record_type_is_meant_to_go(self) -> None:
        self.run_ingest()
        root = self.project_without_rfis()
        code, out, err = self.run_ingest(["--records", root, "--project", "proj", "--slug", "library-renovation", "--allow-missing-types"])
        self.assertEqual(code, 0, err)
        self.assertIn("deleted 14", out)

    def test_a_malformed_csv_is_a_clear_message_not_a_traceback(self) -> None:
        import shutil
        import tempfile
        from pathlib import Path

        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, True)
        shutil.copytree("data/records/samples/library-renovation", Path(root) / "proj")
        (Path(root) / "proj" / "rfis.csv").write_text("project_id,subject\nP-1,no number column\n", encoding="utf-8")
        code, out, err = self.run_ingest(["--records", root, "--project", "proj"])
        self.assertEqual(code, 2)
        self.assertIn("Cannot load the project records", err)
        self.assertIn("rfi_number", err)

    def test_an_error_inside_the_embedder_is_not_reported_as_a_loading_problem(self) -> None:
        def broken(texts: list[str]) -> list[list[float]]:
            raise ValueError("the embedding service sent nonsense")

        code, out, err = self.run_ingest(embedder=(broken, "hashing"))
        self.assertEqual(code, 1)
        self.assertNotIn("Cannot load the project records", err)
        self.assertIn("Ingest failed", err)

    def test_the_project_name_defaults_to_the_folder_and_can_be_overridden(self) -> None:
        self.run_ingest()
        self.run_ingest(["--slug", "second-copy"])
        slugs = {r[0] for r in self.conn.execute("SELECT slug FROM projects")}
        self.assertEqual(slugs, {"library-renovation", "second-copy"})

    def test_a_missing_records_folder_is_a_clear_error(self) -> None:
        code, out, err = self.run_ingest(["--project", "samples/typo"])
        self.assertEqual(code, 2)
        self.assertIn("Project folder not found", err)

    def test_the_wrong_embedding_size_is_refused_and_nothing_is_written(self) -> None:
        code, out, err = self.run_ingest(embedder=(lambda texts: [[0.1] * 10 for _ in texts], "tiny"))
        self.assertEqual(code, 1)
        self.assertIn("Ingest failed", err)
        self.assertEqual(self.chunk_count(), 0)

    def test_an_embedding_service_outage_writes_nothing(self) -> None:
        def down(texts: list[str]) -> list[list[float]]:
            raise TransientError("embedding service down")

        code, out, err = self.run_ingest(embedder=(down, "hashing"))
        self.assertEqual(code, 1)
        self.assertIn("Nothing was written", err)
        self.assertEqual(self.chunk_count(), 0)

    def test_a_database_error_during_the_write_is_a_clean_failure_not_a_traceback(self) -> None:
        import psycopg

        with mock.patch("rag.cli.ingest.PostgresStore.sync", side_effect=psycopg.errors.DataError("bad value")):
            code, out, err = self.run_ingest()
        self.assertEqual(code, 1)
        self.assertIn("Ingest failed (DataError)", err)

    def test_an_unreachable_database_is_a_clear_error_without_the_password(self) -> None:
        code, out, err = self.run_ingest(url="postgresql://rag:supersecretvalue@127.0.0.1:1/rag")
        self.assertEqual(code, 2)
        self.assertIn("Cannot reach the database", err)
        self.assertNotIn("supersecretvalue", err)


if __name__ == "__main__":
    unittest.main()
