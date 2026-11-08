import io
import unittest
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any
from unittest import mock

from rag.answering.answer import Answer
from rag.answering.evidence import ToolEvidence
from rag.cli.main import main
from rag.db.migrate import apply_migrations
from rag.infra.resilience import TransientError
from rag.providers.embed import hashing_embed
from rag.records.snapshot import capture, materialize
from rag.storage.pgstore import PostgresStore
from rag.storage.store import load_chunks
from tests.support.db_support import database_url_for_tests, requires_database, temporary_schema
from tests.support.fakes import ScriptedChat, text_response, tool_response

SAMPLE = Path("data/records/samples/library-renovation")


def run_cli(argv: list[str], chat: Any = None, extra_patches: list | None = None, env: dict | None = None) -> tuple[int, str, str]:
    """Run main() with a fake chat function and no real .env, capturing what it prints."""
    out, err = io.StringIO(), io.StringIO()
    with ExitStack() as stack:
        stack.enter_context(mock.patch("rag.cli.main.load_env"))
        stack.enter_context(mock.patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key", **(env or {})}))
        if chat is not None:
            stack.enter_context(mock.patch("rag.cli.main.make_anthropic_chat", return_value=chat))
        for patch in extra_patches or []:
            stack.enter_context(patch)
        stack.enter_context(redirect_stdout(out))
        stack.enter_context(redirect_stderr(err))
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


class MemoryStoreCliTests(unittest.TestCase):
    """--store memory needs no database and no embedding key, so these run anywhere."""

    def test_budget_and_retrieval_failures_exit_unsuccessfully(self) -> None:
        for status in ("budget_exhausted", "retrieval_failed"):
            with self.subTest(status=status):
                result = Answer("Could not answer.", [], [], status=status)
                code, _, _ = run_cli(["--store", "memory", "question"],
                                     extra_patches=[mock.patch("rag.cli.main.run_agent", return_value=result)])
                self.assertEqual(code, 1)

    def test_success_prints_the_answer_and_a_cost_summary_line(self) -> None:
        chat = ScriptedChat([
            {**tool_response("get_record", {"record_id": "RFI-003"}), "usage": {"input_tokens": 100, "output_tokens": 10}},
            {**text_response("RFI-003 is closed [RFI-003]."), "usage": {"input_tokens": 200, "output_tokens": 20}},
        ])
        code, out, err = run_cli(["--store", "memory", "--as-of", "2026-09-20", "What about RFI-003?"], chat)
        self.assertEqual(code, 0)
        self.assertIn("RFI-003 is closed", out)
        self.assertIn("3 LLM calls", err)  # includes the semantic verifier
        self.assertIn("Verifier: supported", err)
        self.assertIn("Dataset:", err)
        self.assertIn("as of 2026-09-20", err)
        self.assertIn("1 tool calls", err)

    def test_cited_calculation_is_shown_to_the_reader(self) -> None:
        evidence = ToolEvidence("aggregate_records", "v", "result", False, computed="sum: $100",
                                computed_id="CALC-123")
        result = Answer("The sum is $100 [CALC-123].", ["CALC-123"], [], evidence=(evidence,))
        code, out, _ = run_cli(["--store", "memory", "question"],
                               extra_patches=[mock.patch("rag.cli.main.run_agent", return_value=result)])
        self.assertEqual(code, 0)
        self.assertIn("[CALC-123] sum: $100", out)

    def test_a_figure_the_cited_record_does_not_state_is_warned_about_on_stderr(self) -> None:
        wrong = {**text_response("CE-004 is estimated at $20,000 [CE-004]."), "usage": {"input_tokens": 100, "output_tokens": 10}}
        chat = ScriptedChat([{**tool_response("get_record", {"record_id": "CE-004"}), "usage": {"input_tokens": 100, "output_tokens": 10}},
                             wrong, wrong])
        code, out, err = run_cli(["--store", "memory", "--as-of", "2026-09-20", "What does CE-004 cost?"], chat)
        self.assertEqual(code, 1)
        self.assertNotIn("$20,000", out)
        self.assertIn("could not be verified", out)
        self.assertIn("Withheld because: figures the cited records do not state (1)", err)
        self.assertNotIn("$20,000", err)

    def test_an_api_failure_exits_with_1_and_still_prints_the_summary_line(self) -> None:
        def down(system: str, messages: list[dict], tools: list[dict]) -> dict:
            raise TransientError("service overloaded")

        code, out, err = run_cli(["--store", "memory", "--as-of", "2026-09-20", "anything"], down)
        self.assertEqual(code, 1)
        self.assertEqual(out, "")
        self.assertIn("request failed", err)
        self.assertIn("TransientError", err)
        self.assertIn("[request ", err)  # the cost/time line is still shown

    def test_a_bad_date_is_rejected_before_any_api_call(self) -> None:
        chat = ScriptedChat([])
        code, out, err = run_cli(["--store", "memory", "--as-of", "tomorrow", "anything"], chat)
        self.assertEqual(code, 2)
        self.assertIn("YYYY-MM-DD", err)
        self.assertEqual(chat.calls, [])

    def test_a_missing_project_folder_is_a_clear_error(self) -> None:
        code, out, err = run_cli(["--store", "memory", "--project", "samples/typo", "anything"], ScriptedChat([]))
        self.assertEqual(code, 2)
        self.assertIn("Project folder not found", err)

    def test_missing_api_key_is_reported(self) -> None:
        out, err = io.StringIO(), io.StringIO()
        with mock.patch("rag.cli.main.load_env"), \
                mock.patch.dict("os.environ", {}, clear=True), \
                redirect_stdout(out), redirect_stderr(err):
            code = main(["--store", "memory", "anything"])
        self.assertEqual(code, 2)
        self.assertIn("ANTHROPIC_API_KEY", err.getvalue())

    def test_search_only_needs_no_anthropic_key(self) -> None:
        out, err = io.StringIO(), io.StringIO()
        with mock.patch("rag.cli.main.load_env"), \
                mock.patch.dict("os.environ", {}, clear=True), \
                redirect_stdout(out), redirect_stderr(err):
            code = main(["--store", "memory", "--search-only", "RFI-003"])
        self.assertEqual(code, 0)
        self.assertIn("[RFI-003]", out.getvalue())


@requires_database
class PostgresCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self._schema = temporary_schema()
        conn, self.schema = self._schema.__enter__()
        self.addCleanup(self._schema.__exit__, None, None, None)
        apply_migrations(conn, embedding_dim=256)
        PostgresStore(conn, "library-renovation").sync(load_chunks(SAMPLE), hashing_embed, "hashing", source_files=capture(SAMPLE))
        self.url = database_url_for_tests()

    def run_pg(self, argv: list[str], chat: Any = None, embedder: Any = (hashing_embed, "hashing"), url: str | None = None,
               env_extra: dict | None = None) -> tuple[int, str, str]:
        patches = [
            mock.patch("rag.cli.main.database_url_from_env", return_value=url or self.url),
            mock.patch("rag.cli.main.embedder_from_env", return_value=embedder),
        ]
        return run_cli(argv, chat, patches, env={"RAG_DB_SCHEMA": self.schema, **(env_extra or {})})

    def test_dataset_changes_and_reverts_during_a_question_are_detected(self) -> None:
        from rag.db.connection import connect

        original = capture(SAMPLE)
        changed = dict(original)
        changed["rfis.csv"] = original["rfis.csv"].replace("Existing beam size", "Changed beam size", 1)

        def answer_after_two_ingests(*args: Any, **kwargs: Any) -> Answer:
            assert self.url is not None
            with connect(self.url, schema=self.schema) as writer:
                store = PostgresStore(writer, "library-renovation")
                with materialize(changed) as folder:
                    store.sync(load_chunks(folder), hashing_embed, "hashing", source_files=changed)
                store.sync(load_chunks(SAMPLE), hashing_embed, "hashing", source_files=original)
            return Answer("A stale answer", [], [])

        with mock.patch("rag.cli.main.run_agent", side_effect=answer_after_two_ingests):
            code, out, err = self.run_pg(["--as-of", "2026-09-20", "question"])
        self.assertEqual(code, 1)
        self.assertNotIn("A stale answer", out)
        self.assertIn("changed during the question", err)

    def test_the_retriever_gets_the_similarity_cut_off_of_the_embedding_model_in_use(self) -> None:
        captured: dict = {}
        real_retriever = __import__("rag.cli.main", fromlist=["Retriever"]).Retriever

        def spy(store: Any, embed: Any, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return real_retriever(store, embed, **kwargs)

        url = "api:https://api.openai.com/v1:text-embedding-3-small"
        with mock.patch("rag.cli.main.Retriever", side_effect=spy):
            self.run_pg(["--search-only", "RFI-003"], embedder=(hashing_embed, url))
        self.assertEqual(captured, {})  # the store holds 'hashing' vectors, so a different model is refused first
        with mock.patch("rag.cli.main.Retriever", side_effect=spy):
            self.run_pg(["--search-only", "RFI-003"])
        self.assertEqual(captured["min_vector_score"], 0.2)  # the hashing embedder's cut-off
        captured.clear()
        with mock.patch("rag.cli.main.Retriever", side_effect=spy):
            self.run_pg(["--search-only", "RFI-003"], env_extra={"RAG_MIN_VECTOR_SCORE": "0.33"})
        self.assertEqual(captured["min_vector_score"], 0.33)  # and it can be set in the environment

    def test_a_bad_cut_off_setting_is_a_clear_error(self) -> None:
        code, out, err = self.run_pg(["--search-only", "x"], env_extra={"RAG_MIN_VECTOR_SCORE": "high"})
        self.assertEqual(code, 2)
        self.assertIn("RAG_MIN_VECTOR_SCORE", err)

    def test_a_database_error_during_a_search_is_a_clean_failure(self) -> None:
        import psycopg

        with mock.patch("rag.retrieval.retrieve.Retriever.search", side_effect=psycopg.errors.OperationalError("connection lost")):
            code, out, err = self.run_pg(["--search-only", "x"])
        self.assertEqual(code, 1)
        self.assertIn("Search failed", err)

    def test_questions_are_asked_over_a_read_only_connection(self) -> None:
        from rag.db.connection import connect as real_connect

        seen: list[bool] = []

        def spy(*args: Any, **kwargs: Any) -> Any:
            seen.append(bool(kwargs.get("read_only")))
            return real_connect(*args, **kwargs)

        with mock.patch("rag.cli.main.connect", side_effect=spy):
            code, out, err = self.run_pg(["--search-only", "RFI-003"])
        self.assertEqual(code, 0, err)
        self.assertEqual(seen, [True])

    def test_search_only_reads_from_postgres(self) -> None:
        code, out, err = self.run_pg(["--search-only", "RFI-003"])
        self.assertEqual(code, 0, err)
        self.assertIn("[RFI-003]", out)

    def test_a_full_question_is_answered_from_postgres(self) -> None:
        chat = ScriptedChat([tool_response("get_record", {"record_id": "RFI-003"}),
                             text_response("RFI-003 is closed and linked to CE-001 [RFI-003] [CE-001].")])
        code, out, err = self.run_pg(["--as-of", "2026-09-20", "What about RFI-003?"], chat)
        self.assertEqual(code, 0, err)
        self.assertIn("RFI-003 is closed", out)
        self.assertNotIn("WARNING", err)  # the linked record came from the graph tables and counts as evidence

    def test_a_project_that_was_never_ingested_says_how_to_ingest_it(self) -> None:
        code, out, err = self.run_pg(["--slug", "never-ingested", "--search-only", "x"])
        self.assertEqual(code, 2)
        self.assertIn("python -m rag.cli.ingest", err)

    def test_questions_cannot_be_embedded_with_a_different_model_than_the_records(self) -> None:
        code, out, err = self.run_pg(["--search-only", "x"], embedder=(hashing_embed, "some-other-model"))
        self.assertEqual(code, 2)
        self.assertIn("embedded with", err)

    def test_an_unreachable_database_is_a_clear_error_without_the_password(self) -> None:
        dead = "postgresql://rag:supersecretvalue@127.0.0.1:1/rag"
        code, out, err = self.run_pg(["--search-only", "x"], url=dead)
        self.assertEqual(code, 2)
        self.assertIn("Cannot reach the database", err)
        self.assertNotIn("supersecretvalue", err)
        self.assertIn("--store memory", err)


if __name__ == "__main__":
    unittest.main()
