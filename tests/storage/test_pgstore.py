import dataclasses
import json
import unittest
from pathlib import Path
from unittest import mock

from rag.db.connection import fetch_one
from rag.db.migrate import apply_migrations
from rag.infra.resilience import TransientError
from rag.infra.telemetry import EVENT_LOGGER, log_event
from rag.providers.embed import hashing_embed
from rag.records.chunker import Chunk
from rag.storage.pgstore import PostgresStore, SyncReport, build_terms, build_tsquery, idf, revision_of, text_hash, vector_literal
from rag.storage.store import load_chunks
from rag.storage.store_api import StoreError
from tests.support.db_support import requires_database, temporary_schema

SAMPLE = Path("data/records/samples/library-renovation")
DIM = 256


class CountingEmbedder:
    def __init__(self) -> None:
        self.texts_embedded = 0

    def __call__(self, texts: list[str]) -> list[list[float]]:
        self.texts_embedded += len(texts)
        return hashing_embed(texts)


class SortingContractTests(unittest.TestCase):
    """vector_search must return best-first with ties broken by record id and revision, WHATEVER order the
    database hands the rows back (the index can return them out of order, and equal scores have no order).
    A stand-in connection returns the rows scrambled, so this holds without depending on what Postgres does."""

    class FakeResult:
        def __init__(self, rows: list[tuple]) -> None:
            self._rows = rows

        def fetchall(self) -> list[tuple]:
            return self._rows

    class FakeConnection:
        def __init__(self, rows: list[tuple]) -> None:
            self.rows = rows

        def execute(self, sql: str, params: tuple = ()) -> SortingContractTests.FakeResult:
            return SortingContractTests.FakeResult(self.rows)

    def store_returning(self, rows: list[tuple]) -> PostgresStore:
        store = PostgresStore.__new__(PostgresStore)  # skip __init__: no real database is involved
        store.conn = self.FakeConnection(rows)  # type: ignore[assignment]
        store.embedding_dim = 2
        store._project_id = 1
        return store

    def test_best_first_then_record_id_then_revision_and_never_nan(self) -> None:
        def row(record_id: str, score: float, revision: str = "0") -> tuple:
            return (record_id, "rfi", "text", {"revision": revision}, score)

        rows = [row("RFI-3", 0.5), row("RFI-1", 0.9), row("S-1", 0.5, "1"), row("RFI-2", 0.5),
                row("S-1", 0.5, "0"), row("RFI-9", float("nan"))]
        hits = self.store_returning(rows).vector_search([1.0, 0.0], limit=10)
        self.assertEqual([(h.chunk.record_id, h.chunk.metadata["revision"]) for h in hits],
                         [("RFI-1", "0"), ("RFI-2", "0"), ("RFI-3", "0"), ("S-1", "0"), ("S-1", "1")])


class PureHelperTests(unittest.TestCase):
    def test_vector_literal_round_trips_floats_exactly(self) -> None:
        values = [0.5, 1 / 3, -2.25e-07, 0.0]
        parsed = [float(x) for x in vector_literal(values).strip("[]").split(",")]
        self.assertEqual(parsed, values)

    def test_text_hash_is_stable_and_sensitive_to_every_character(self) -> None:
        self.assertEqual(text_hash("abc"), text_hash("abc"))
        self.assertNotEqual(text_hash("abc"), text_hash("abd"))

    def test_revision_comes_from_metadata_and_defaults_to_zero(self) -> None:
        self.assertEqual(revision_of(Chunk("submittal", "S-001", "t", {"revision": "2"})), 2)
        self.assertEqual(revision_of(Chunk("rfi", "RFI-1", "t", {})), 0)
        self.assertEqual(revision_of(Chunk("submittal", "S-001", "t", {"revision": "x"})), 0)

    def test_tsquery_keeps_content_words_and_splits_ids(self) -> None:
        self.assertEqual(build_tsquery("Which RFIs are overdue?"), "rfis | overdue")
        self.assertEqual(build_tsquery("RFI-003"), "(rfi <-> 003)")  # an ID is an exact phrase
        self.assertEqual(build_tsquery("DR-2026-08-03 conduit"), "(dr <-> 2026 <-> 08 <-> 03) | conduit")

    def test_tsquery_keeps_accented_words_and_drops_lone_letters(self) -> None:
        self.assertEqual(build_tsquery("Müller café"), "müller | café")
        self.assertEqual(build_tsquery("contractor's note"), "contractor | note")
        self.assertEqual(build_tsquery("S-001"), "(s <-> 001)")  # inside an ID the letter is needed

    def test_terms_are_the_pieces_of_the_query(self) -> None:
        self.assertEqual(build_terms("abandoned conduit RFI-003"), ["abandoned", "conduit", "(rfi <-> 003)"])
        self.assertEqual(build_terms("the of"), [])

    def test_idf_weights_rare_terms_above_common_ones_and_never_goes_negative(self) -> None:
        self.assertGreater(idf(1, 100), idf(50, 100))
        self.assertGreater(idf(100, 100), 0.0)  # a word in every record still scores a little, never below zero
        self.assertGreater(idf(0, 100), idf(1, 100))

    def test_tsquery_is_none_when_nothing_searchable_is_left(self) -> None:
        for text in ("", "   ", "the of and", "?!"):
            with self.subTest(text=text):
                self.assertIsNone(build_tsquery(text))

    def test_tsquery_can_only_contain_letters_digits_and_the_operators_we_add(self) -> None:
        query = build_tsquery("'; DROP TABLE chunks; -- & ! ( ) : * <-> \\x00")
        assert query is not None
        self.assertRegex(query, r"^[\w |()<>-]+$")  # letters and digits (any language), spaces and our operators only
        self.assertNotRegex(query, r"[':&!*\\]")

    def test_tsquery_drops_repeats_and_caps_the_length(self) -> None:
        self.assertEqual(build_tsquery("beam beam BEAM"), "beam")
        long_query = build_tsquery(" ".join(f"w{i}" for i in range(500)))
        assert long_query is not None
        self.assertLessEqual(len(long_query.split(" | ")), 32)


@requires_database
class SyncTests(unittest.TestCase):
    def setUp(self) -> None:
        self._schema = temporary_schema()
        self.conn, self.schema = self._schema.__enter__()
        apply_migrations(self.conn, embedding_dim=DIM)
        self.store = PostgresStore(self.conn, "library-renovation")
        self.chunks = load_chunks(SAMPLE)

    def tearDown(self) -> None:
        self._schema.__exit__(None, None, None)

    def scalar(self, sql: str) -> int:
        return fetch_one(self.conn, sql)[0]

    def test_the_first_sync_inserts_and_embeds_everything(self) -> None:
        embedder = CountingEmbedder()
        report = self.store.sync(self.chunks, embedder, "hashing")
        self.assertEqual(report, SyncReport(inserted=88, updated=0, unchanged=0, deleted=0, embedded=88))
        self.assertEqual(embedder.texts_embedded, 88)
        self.assertEqual(self.store.count(), 88)

    def test_syncing_the_same_data_again_does_nothing_and_embeds_nothing(self) -> None:
        self.store.sync(self.chunks, hashing_embed, "hashing")
        embedder = CountingEmbedder()
        report = self.store.sync(self.chunks, embedder, "hashing")
        self.assertEqual(report, SyncReport(inserted=0, updated=0, unchanged=88, deleted=0, embedded=0))
        self.assertEqual(embedder.texts_embedded, 0)

    def test_a_changed_text_updates_and_re_embeds_only_that_record(self) -> None:
        self.store.sync(self.chunks, hashing_embed, "hashing")
        edited = [dataclasses.replace(c, text=c.text + " Edited.") if c.record_id == "RFI-003" else c for c in self.chunks]
        embedder = CountingEmbedder()
        report = self.store.sync(edited, embedder, "hashing")
        self.assertEqual((report.updated, report.unchanged, report.embedded), (1, 87, 1))
        self.assertTrue(self.store.get("RFI-003")[0].text.endswith("Edited."))

    def test_a_metadata_only_change_updates_without_re_embedding(self) -> None:
        self.store.sync(self.chunks, hashing_embed, "hashing")
        edited = [dataclasses.replace(c, metadata={**c.metadata, "status": "void"}) if c.record_id == "RFI-003" else c
                  for c in self.chunks]
        embedder = CountingEmbedder()
        report = self.store.sync(edited, embedder, "hashing")
        self.assertEqual((report.updated, report.embedded), (1, 0))
        self.assertEqual(embedder.texts_embedded, 0)
        self.assertEqual(self.store.get("RFI-003")[0].metadata["status"], "void")

    def test_a_record_that_disappeared_from_the_source_is_deleted(self) -> None:
        self.store.sync(self.chunks, hashing_embed, "hashing")
        report = self.store.sync(self.chunks[:-1], hashing_embed, "hashing")
        self.assertEqual((report.deleted, report.unchanged), (1, 87))
        self.assertEqual(self.store.count(), 87)

    def test_a_new_embedding_model_re_embeds_everything(self) -> None:
        self.store.sync(self.chunks, hashing_embed, "model-a")
        embedder = CountingEmbedder()
        report = self.store.sync(self.chunks, embedder, "model-b")
        self.assertEqual((report.updated, report.embedded), (88, 88))
        self.assertEqual(self.scalar("SELECT count(DISTINCT embedder_id) FROM chunks"), 1)
        self.assertEqual(self.scalar("SELECT count(*) FROM chunks WHERE embedder_id = 'model-b'"), 88)

    def test_vectors_of_the_wrong_size_are_refused_before_anything_is_written(self) -> None:
        with self.assertRaises(StoreError) as caught:
            self.store.sync(self.chunks, lambda texts: [[0.1] * 10 for _ in texts], "tiny")
        self.assertIn("10", str(caught.exception))
        self.assertIn(str(DIM), str(caught.exception))
        self.assertEqual(self.scalar("SELECT count(*) FROM projects"), 0)

    # ---- bulk loading ---------------------------------------------------------------------------
    def index_definition(self) -> str | None:
        row = fetch_one(self.conn, "SELECT indexdef FROM pg_indexes WHERE schemaname = %s AND indexname = 'chunks_embedding_hnsw'",
                                (self.schema,))
        return row[0].replace(self.schema, "S") if row else None

    def test_a_big_first_load_rebuilds_the_vector_index_instead_of_updating_it_row_by_row(self) -> None:
        normal_definition = None
        self.store.sync(self.chunks, hashing_embed, "hashing")  # below the default threshold: the normal path
        normal_definition = self.index_definition()
        self.assertIsNotNone(normal_definition)

        bulk = PostgresStore(self.conn, "bulk-job", bulk_threshold=10)
        with self.assertLogs(EVENT_LOGGER, level="INFO") as logs:
            bulk.sync(self.chunks, hashing_embed, "hashing")
        events = [json.loads(r.getMessage()) for r in logs.records]
        self.assertTrue(any(e["event"] == "bulk_load" and e["rows"] == 88 for e in events))
        self.assertEqual(self.index_definition(), normal_definition)  # exactly the index the migration made
        hits = bulk.vector_search(hashing_embed(["abandoned conduit in the ceiling"])[0], limit=3)
        self.assertIn("RFI-003", [h.chunk.record_id for h in hits])  # and it works

    def test_a_small_sync_never_touches_the_index(self) -> None:
        with self.assertLogs(EVENT_LOGGER, level="INFO") as logs:
            self.store.sync(self.chunks[:5], hashing_embed, "hashing")
            log_event("marker")  # assertLogs needs at least one record
        self.assertFalse(any(json.loads(r.getMessage())["event"] == "bulk_load" for r in logs.records))

    def test_the_threshold_itself_counts_as_a_bulk_load_and_one_row_below_does_not(self) -> None:
        def bulk_events(threshold: int, project: str) -> int:
            with self.assertLogs(EVENT_LOGGER, level="INFO") as logs:
                PostgresStore(self.conn, project, bulk_threshold=threshold).sync(self.chunks[:5], hashing_embed, "hashing")
                log_event("marker")
            return sum(json.loads(r.getMessage())["event"] == "bulk_load" for r in logs.records)

        self.assertEqual(bulk_events(5, "at-threshold"), 1)  # 5 rows to embed, threshold 5
        self.assertEqual(bulk_events(6, "below-threshold"), 0)  # 5 rows to embed, threshold 6

    def test_a_failure_during_a_bulk_load_puts_the_index_back(self) -> None:
        bulk = PostgresStore(self.conn, "bulk-job", bulk_threshold=10)
        before = self.index_definition()
        with mock.patch.object(PostgresStore, "_sync_graph", side_effect=RuntimeError("disk full")):
            with self.assertRaises(RuntimeError):
                bulk.sync(self.chunks, hashing_embed, "hashing")
        self.assertEqual(self.index_definition(), before)  # DDL is transactional: the drop was rolled back too
        self.assertEqual(self.scalar("SELECT count(*) FROM chunks"), 0)

    # ---- a record type that vanishes from the folder ---------------------------------------------------
    def test_a_record_type_that_is_stored_but_missing_from_the_new_data_is_refused_not_deleted(self) -> None:
        self.store.sync(self.chunks, hashing_embed, "hashing")
        rfis_only = [c for c in self.chunks if c.record_type == "rfi"]
        before = self.scalar("SELECT count(*) FROM chunks")
        with self.assertRaises(StoreError) as caught:
            self.store.sync(rfis_only, hashing_embed, "hashing")
        message = str(caught.exception)
        for missing in ("punch_item", "daily_report", "submittal", "change_event"):
            self.assertIn(missing, message)
        self.assertIn("allow_missing_types", message)
        self.assertEqual(self.scalar("SELECT count(*) FROM chunks"), before)  # nothing was deleted

    def test_the_operator_can_allow_a_type_to_disappear_on_purpose(self) -> None:
        self.store.sync(self.chunks, hashing_embed, "hashing")
        rfis_only = [c for c in self.chunks if c.record_type == "rfi"]
        report = self.store.sync(rfis_only, hashing_embed, "hashing", allow_missing_types=True)
        self.assertEqual(report.deleted, len(self.chunks) - len(rfis_only))
        self.assertEqual(self.scalar("SELECT count(*) FROM chunks"), len(rfis_only))

    def test_a_first_load_of_a_subset_and_dropping_single_records_are_not_refused(self) -> None:
        rfis_only = [c for c in self.chunks if c.record_type == "rfi"]
        self.store.sync(rfis_only, hashing_embed, "hashing")  # nothing of the other types was ever stored
        report = self.store.sync(rfis_only[:-1], hashing_embed, "hashing")  # one RFI fewer: an ordinary deletion
        self.assertEqual(report.deleted, 1)

    def test_a_failing_embedder_leaves_the_database_untouched(self) -> None:
        def down(texts: list[str]) -> list[list[float]]:
            raise TransientError("embedding service down")

        with self.assertRaises(TransientError):
            self.store.sync(self.chunks, down, "hashing")
        self.assertEqual(self.scalar("SELECT count(*) FROM chunks"), 0)
        self.assertEqual(self.scalar("SELECT count(*) FROM projects"), 0)

    def test_a_failure_while_writing_rolls_everything_back(self) -> None:
        with mock.patch.object(PostgresStore, "_rebuild_graph", side_effect=RuntimeError("disk full")):
            with self.assertRaises(RuntimeError):
                self.store.sync(self.chunks, hashing_embed, "hashing")
        self.assertEqual(self.scalar("SELECT count(*) FROM chunks"), 0)
        self.assertEqual(self.scalar("SELECT count(*) FROM projects"), 0)

    def test_an_embedder_returning_the_wrong_number_of_vectors_is_refused(self) -> None:
        with self.assertRaises(StoreError):
            self.store.sync(self.chunks, lambda texts: hashing_embed(texts)[:-1], "hashing")

    def test_two_chunks_with_the_same_id_and_revision_are_refused(self) -> None:
        with self.assertRaises(StoreError) as caught:
            self.store.sync([self.chunks[0], self.chunks[0]], hashing_embed, "hashing")
        self.assertIn("RFI-001", str(caught.exception))

    def test_projects_are_isolated_from_each_other(self) -> None:
        other = PostgresStore(self.conn, "other-job")
        self.store.sync(self.chunks, hashing_embed, "hashing")
        other.sync(self.chunks[:10], hashing_embed, "hashing")
        self.assertEqual((self.store.count(), other.count()), (88, 10))
        # Both projects contain RFI-001; every read must see only its own copy.
        for store in (self.store, other):
            with self.subTest(project=store.project_slug):
                self.assertEqual(len(store.get("RFI-001")), 1)
                self.assertEqual([h.chunk.record_id for h in store.keyword_search("RFI-001", limit=5)].count("RFI-001"), 1)
                self.assertEqual(len([h for h in store.vector_search(hashing_embed(["rfi"])[0], limit=500) if h.chunk.record_id == "RFI-001"]), 1)
        self.assertEqual(len(other.linked("CE-001")), 0)  # CE-001 is not among other's first 10 chunks
        other.sync([], hashing_embed, "hashing", allow_missing_types=True)  # emptying one project, on purpose...
        self.assertEqual((self.store.count(), other.count()), (88, 0))  # ...leaves the other alone
        self.assertEqual([c.record_id for c in self.store.get("RFI-001")], ["RFI-001"])

    def test_reading_a_project_that_was_never_ingested_says_how_to_ingest_it(self) -> None:
        with self.assertRaises(StoreError) as caught:
            PostgresStore(self.conn, "never-ingested").count()
        self.assertIn("ingest", str(caught.exception))

    def test_a_keyword_search_for_an_id_does_not_return_a_record_that_only_mentions_sibling_ids(self) -> None:
        wanted = Chunk("rfi", "RFI-003", "RFI RFI-003 (closed). Abandoned conduit.", {"record_type": "rfi"})
        decoy = Chunk("change_event", "CE-777", "Change event CE-777. Related RFI: RFI-011. Compare CE-003.", {"record_type": "change_event"})
        self.store.sync([wanted, decoy], hashing_embed, "hashing")
        found = [hit.chunk.record_id for hit in self.store.keyword_search("RFI-003", limit=10)]
        self.assertEqual(found, ["RFI-003"])  # not CE-777, which has the words 'rfi' and '003' but not the ID

    def test_words_with_accents_are_found_by_keyword_search(self) -> None:
        chunk = Chunk("punch_item", "PL-800", "Punch item PL-800. Responsible party: Müller Électrique. Café area.", {})
        self.store.sync([chunk], hashing_embed, "hashing")
        self.assertEqual([h.chunk.record_id for h in self.store.keyword_search("Müller", limit=5)], ["PL-800"])
        self.assertEqual([h.chunk.record_id for h in self.store.keyword_search("CAFÉ", limit=5)], ["PL-800"])

    def test_a_nul_character_is_refused_before_anything_is_embedded_and_names_the_record(self) -> None:
        bad_text = Chunk("rfi", "RFI-801", "text with a NUL \x00 inside", {})
        bad_meta = Chunk("rfi", "RFI-802", "fine", {"note": "nul \x00"})
        for chunk in (bad_text, bad_meta):
            with self.subTest(record=chunk.record_id):
                embedder = CountingEmbedder()
                with self.assertRaises(StoreError) as caught:
                    self.store.sync([self.chunks[0], chunk], embedder, "hashing")
                self.assertIn(chunk.record_id, str(caught.exception))
                self.assertEqual(embedder.texts_embedded, 0)  # nothing paid for
        self.assertEqual(self.scalar("SELECT count(*) FROM chunks"), 0)

    def test_equal_scores_come_back_in_a_stable_order_and_a_zero_vector_does_not_poison_the_list(self) -> None:
        same = hashing_embed(["identical text"])[0]
        numbers = [14, 3, 11, 1, 9, 13, 5, 2, 12, 7, 4, 10, 6, 8]  # stored in a scrambled order
        chunks = [Chunk("rfi", f"RFI-9{n:02d}", "identical text", {}) for n in numbers]
        self.store.sync(chunks, lambda texts: [list(same) for _ in texts], "fixed")
        hits = self.store.vector_search(same, limit=20)
        self.assertEqual([h.chunk.record_id for h in hits], [f"RFI-9{n:02d}" for n in sorted(numbers)])
        nothing = self.store.vector_search([0.0] * DIM, limit=10)  # cosine with a zero vector is not a number
        self.assertTrue(all(h.score == h.score for h in nothing))  # no NaN scores come back

    def test_a_date_style_id_is_found_by_keyword_search(self) -> None:
        self.store.sync(self.chunks, hashing_embed, "hashing")
        hits = self.store.keyword_search("DR-2026-08-03", limit=3)
        self.assertEqual(hits[0].chunk.record_id, "DR-2026-08-03")

    def test_an_ingest_that_loses_a_race_writes_nothing_and_says_so(self) -> None:
        self.store.sync(self.chunks, hashing_embed, "hashing")
        edited = [dataclasses.replace(c, text=c.text + " Edited.") if c.record_id == "RFI-003" else c for c in self.chunks]

        def embed_while_someone_else_writes(texts: list[str]) -> list[list[float]]:
            # Stands in for a second ingest finishing while this one waits for the embedding service.
            self.conn.execute("UPDATE chunks SET text_hash = 'changed-by-someone-else' WHERE record_id = 'RFI-001'")
            return hashing_embed(texts)

        with self.assertRaises(StoreError) as caught:
            self.store.sync(edited, embed_while_someone_else_writes, "hashing")
        self.assertIn("changed by someone else", str(caught.exception))
        self.assertFalse(self.store.get("RFI-003")[0].text.endswith("Edited."))  # nothing was written

    def test_the_graph_has_one_node_per_record_and_one_edge_per_link(self) -> None:
        self.store.sync(self.chunks, hashing_embed, "hashing")
        self.assertEqual(self.scalar("SELECT count(*) FROM nodes WHERE kind IN ('rfi','submittal','change_event','daily_report','punch_item')"), 86)
        self.assertEqual(self.scalar("SELECT count(*) FROM edges WHERE relation = 'related_rfi'"), 4)
        before = (self.scalar("SELECT count(*) FROM nodes"), self.scalar("SELECT count(*) FROM edges"))
        self.store.sync(self.chunks, hashing_embed, "hashing")  # again: the graph is rebuilt, not doubled
        self.assertEqual((self.scalar("SELECT count(*) FROM nodes"), self.scalar("SELECT count(*) FROM edges")), before)

    def test_the_graph_tables_match_the_pure_graph_built_from_the_chunks(self) -> None:
        from rag.records.graph import build_graph

        self.store.sync(self.chunks, hashing_embed, "hashing")
        graph = build_graph(self.chunks)
        self.assertEqual(self.scalar("SELECT count(*) FROM nodes"), len(graph.nodes))
        self.assertEqual(self.scalar("SELECT count(*) FROM edges"), len(graph.edges))
        (label,) = fetch_one(self.conn, "SELECT label FROM nodes WHERE kind = 'trade' AND key = 'sample painting'")
        self.assertEqual(label, "Sample Painting")  # the key is for matching, the label for display

    def test_two_projects_that_reuse_the_same_record_ids_never_mix_their_graphs(self) -> None:
        from rag.records.chunker import row_to_chunk

        def punch(record_id: str, room: str) -> Chunk:
            return row_to_chunk("punch_item", {"project_id": "P", "item_number": record_id, "location": room, "description": "d",
                                               "responsible_party": f"Trade {room}", "created_date": "x", "status": "open"})

        other = PostgresStore(self.conn, "other-job")
        self.store.sync([punch("PL-1", "Room 1"), punch("PL-7", "Room 1")], hashing_embed, "hashing")
        other.sync([punch("PL-1", "Room 2"), punch("PL-7", "Room 2")], hashing_embed, "hashing")
        (mine,) = [r for r in self.store.related("PL-1") if r.chunk.record_id == "PL-7"]
        self.assertEqual(mine.via, "shares room: Room 1; shares trade: Trade Room 1")  # nothing of the other job's Room 2
        (theirs,) = [r for r in other.related("PL-1") if r.chunk.record_id == "PL-7"]
        self.assertEqual(theirs.via, "shares room: Room 2; shares trade: Trade Room 2")

    def test_syncing_unchanged_data_keeps_the_same_graph_nodes_and_does_not_rebuild_them(self) -> None:
        self.store.sync(self.chunks, hashing_embed, "hashing")
        ids_before = [r[0] for r in self.conn.execute("SELECT id FROM nodes ORDER BY id")]
        with mock.patch.object(PostgresStore, "_rebuild_graph") as rebuild:
            self.store.sync(self.chunks, hashing_embed, "hashing")
        rebuild.assert_not_called()
        self.assertEqual([r[0] for r in self.conn.execute("SELECT id FROM nodes ORDER BY id")], ids_before)

    def test_a_graph_damaged_behind_the_stores_back_is_repaired_by_the_next_sync(self) -> None:
        # The stored fingerprint alone would say "unchanged" and leave the damage in place for good.
        self.store.sync(self.chunks, hashing_embed, "hashing")
        edges, nodes = self.scalar("SELECT count(*) FROM edges"), self.scalar("SELECT count(*) FROM nodes")
        related_before = [(r.chunk.record_id, r.via) for r in self.store.related("PL-001")]
        self.assertGreater(len(related_before), 0)
        for damage in ("DELETE FROM edges", "DELETE FROM edges WHERE (src, dst, relation) IN (SELECT src, dst, relation FROM edges LIMIT 1)",
                       "DELETE FROM nodes WHERE kind = 'trade'"):
            with self.subTest(damage=damage):
                self.conn.execute(damage)
                self.store.sync(self.chunks, hashing_embed, "hashing")
                self.assertEqual(self.scalar("SELECT count(*) FROM edges"), edges)
                self.assertEqual(self.scalar("SELECT count(*) FROM nodes"), nodes)
                self.assertEqual([(r.chunk.record_id, r.via) for r in self.store.related("PL-001")], related_before)

    def test_a_change_to_the_graph_rebuilds_it(self) -> None:
        self.store.sync(self.chunks, hashing_embed, "hashing")
        from rag.records.chunker import row_to_chunk

        changed = [row_to_chunk("punch_item", {"project_id": "P", "item_number": c.record_id, "location": "Room 102",
                                               "description": "Fill screw pops", "responsible_party": "Brand New Trade",
                                               "created_date": "x", "status": "open"}) if c.record_id == "PL-001" else c
                   for c in self.chunks]
        self.store.sync(changed, hashing_embed, "hashing")
        labels = {r[0] for r in self.conn.execute("SELECT label FROM nodes WHERE kind = 'trade'")}
        self.assertIn("Brand New Trade", labels)

    def test_entities_from_two_projects_stay_apart(self) -> None:
        def punch(record_id: str) -> Chunk:
            from rag.records.chunker import row_to_chunk
            return row_to_chunk("punch_item", {"project_id": "P", "item_number": record_id, "location": "Room 1", "description": "d",
                                               "responsible_party": "Sample Painting", "created_date": "x", "status": "open"})

        other = PostgresStore(self.conn, "other-job")
        self.store.sync([punch("PL-1"), punch("PL-2")], hashing_embed, "hashing")
        other.sync([punch("PL-9")], hashing_embed, "hashing")
        related = self.store.related("PL-1")
        self.assertEqual([r.chunk.record_id for r in related], ["PL-2"])  # listed once, with both reasons
        self.assertEqual(related[0].via, "shares room: Room 1; shares trade: Sample Painting")
        self.assertEqual(other.related("PL-9"), [])  # nothing of the first project leaks in

    def test_related_returns_the_latest_revision_whatever_order_the_rows_were_stored_in(self) -> None:
        trade = (("responsible_party", "trade", "Acme"),)
        newer = Chunk("submittal", "S-900", "rev one", {"revision": "1", "is_latest": "yes"}, relations=trade)
        older = Chunk("submittal", "S-900", "rev zero", {"revision": "0", "is_latest": "no"}, relations=trade)
        punch = Chunk("punch_item", "PL-900", "a punch item", {"is_latest": "yes"}, relations=trade)
        self.store.sync([newer, older, punch], hashing_embed, "hashing")  # the newest revision is stored FIRST
        related = self.store.related("PL-900")
        self.assertEqual([(r.chunk.record_id, r.chunk.metadata["revision"]) for r in related], [("S-900", "1")])

    def test_a_link_to_a_record_that_does_not_exist_makes_no_edge(self) -> None:
        orphan = Chunk("change_event", "CE-900", "text", {"related_rfi": "RFI-404", "record_type": "change_event"})
        self.store.sync([orphan], hashing_embed, "hashing")
        self.assertEqual(self.scalar("SELECT count(*) FROM edges"), 0)

    def test_search_settings_for_good_recall_are_applied_to_the_session(self) -> None:
        self.assertEqual(fetch_one(self.conn, "SHOW hnsw.ef_search")[0], "100")
        self.assertEqual(fetch_one(self.conn, "SHOW hnsw.iterative_scan")[0], "relaxed_order")

    def test_a_query_vector_of_the_wrong_size_is_a_clear_error(self) -> None:
        self.store.sync(self.chunks, hashing_embed, "hashing")
        with self.assertRaises(StoreError) as caught:
            self.store.vector_search([0.1] * 1536, limit=5)
        self.assertIn("1536", str(caught.exception))

    def test_chunks_without_a_vector_are_skipped_by_vector_search_but_found_by_keyword(self) -> None:
        self.store.sync(self.chunks, hashing_embed, "hashing")
        self.conn.execute("UPDATE chunks SET embedding = NULL, embedder_id = NULL WHERE record_id = 'CE-005'")
        ids = [h.chunk.record_id for h in self.store.vector_search(hashing_embed(["asbestos mastic carpet"])[0], limit=88)]
        self.assertNotIn("CE-005", ids)
        self.assertEqual(self.store.keyword_search("asbestos", limit=3)[0].chunk.record_id, "CE-005")
        report = self.store.sync(self.chunks, hashing_embed, "hashing")  # the next sync repairs it
        self.assertEqual((report.updated, report.embedded), (1, 1))


class MigrationRequiredTests(unittest.TestCase):
    @requires_database
    def test_an_unmigrated_database_says_to_run_the_migrations(self) -> None:
        with temporary_schema() as (conn, _):
            with self.assertRaises(StoreError) as caught:
                PostgresStore(conn, "x")
            self.assertIn("migrate", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
