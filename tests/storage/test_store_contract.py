"""One set of tests, run against every Store implementation (the in-memory index and PostgreSQL).

If both pass the same tests, code that uses a Store works with either. Add a new store by adding a
small class at the bottom that provides make_store()."""
import unittest
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Any

from rag.db.migrate import apply_migrations
from rag.providers.embed import hashing_embed
from rag.records.chunker import Chunk
from rag.records.loader import load_csv
from rag.storage.pgstore import PostgresStore
from rag.storage.store import build_index, load_chunks, make_index
from rag.storage.store_api import AGGREGATE_OPERATIONS, MAX_AGGREGATE_IDS, RelatedRecords, Store, StoreError
from tests.support.db_support import requires_database, temporary_schema
from tests.support.fakes import MixinBase

SAMPLE = Path("data/records/samples/library-renovation")
DIM = 256  # the size of the offline hashing embedder, used so the tests need no API key


def query_vector(text: str) -> list[float]:
    return hashing_embed([text])[0]


class StoreContract(MixinBase):
    """Not a TestCase itself (so it is not run alone); the classes below mix it with unittest.TestCase."""

    store: Store

    def test_count_is_the_number_of_chunks(self) -> None:
        self.assertEqual(self.store.count(), 14 + 11 + 7 + 32 + 24)

    def test_filtered_listing_is_ordered_and_reports_more_than_limit(self) -> None:
        filters = {"record_type": "rfi", "status": "open", "ball_in_court": "Architect"}
        self.assertEqual([c.record_id for c in self.store.list_records(filters, 10)],
                         ["RFI-009", "RFI-010", "RFI-012", "RFI-015"])
        self.assertEqual(len(self.store.list_records(filters, 2)), 3)

    def test_get_returns_every_revision_and_ignores_case(self) -> None:
        self.assertEqual([c.record_id for c in self.store.get("RFI-003")], ["RFI-003"])
        self.assertEqual(len(self.store.get("S-001")), 2)
        self.assertEqual(len(self.store.get("s-001")), 2)
        self.assertEqual(self.store.get("RFI-999"), [])

    def test_revisions_come_back_oldest_first(self) -> None:
        self.assertEqual([c.metadata["revision"] for c in self.store.get("S-004")], ["0", "1"])

    def test_linked_records_are_found_in_both_directions(self) -> None:
        self.assertIn("CE-001", [c.record_id for c in self.store.linked("RFI-003")])
        self.assertIn("RFI-003", [c.record_id for c in self.store.linked("CE-001")])
        self.assertEqual(self.store.linked("RFI-999"), [])

    def test_related_finds_a_direct_link_in_both_directions_and_names_the_relation(self) -> None:
        from_event = {r.chunk.record_id: r.via for r in self.store.related("CE-001")}
        self.assertEqual(from_event["RFI-003"], "related_rfi")
        from_rfi = {r.chunk.record_id: r.via for r in self.store.related("RFI-003")}
        self.assertEqual(from_rfi["CE-001"], "related_rfi")

    def test_related_follows_a_shared_trade_or_room_across_record_types(self) -> None:
        related = {r.chunk.record_id: r.via for r in self.store.related("PL-001", limit=50, per_entity=50)}
        self.assertEqual(related["PL-006"], "shares trade: Sample Painting")  # same trade, another punch item
        self.assertEqual(related["S-010"], "shares trade: Sample Painting")  # same trade, a submittal
        self.assertEqual(related["PL-007"], "shares room: Room 102")  # same room, different trade
        self.assertNotIn("PL-001", related)  # never the record itself

    def test_related_lists_direct_links_before_shared_entities(self) -> None:
        vias = [r.via for r in self.store.related("CE-001", limit=50)]
        self.assertEqual(vias[0], "related_rfi")

    def test_related_does_not_expand_through_a_huge_hub_like_ball_in_court(self) -> None:
        # S-004 is with the Architect, who also holds RFI-010 and others. That is not a useful neighbour.
        ids = [r.chunk.record_id for r in self.store.related("S-004", limit=100, per_entity=100)]
        self.assertNotIn("RFI-010", ids)
        self.assertIn("PL-004", ids)  # but the shared trade (Sample Flooring) is

    def test_related_caps_each_entity_and_the_total(self) -> None:
        capped = self.store.related("PL-001", limit=50, per_entity=2)
        by_via: dict[str, int] = {}
        for r in capped:
            by_via[r.via] = by_via.get(r.via, 0) + 1
        self.assertTrue(all(count <= 2 for count in by_via.values()), by_via)
        self.assertEqual(len(self.store.related("PL-001", limit=3, per_entity=50)), 3)

    def test_when_an_entity_has_to_be_cut_the_other_record_types_and_the_newest_ids_win(self) -> None:
        # Sample Painting has 4 other punch items (PL-006, 011, 016, 021) and the submittal S-010.
        trade = {r.chunk.record_id for r in self.store.related("PL-001", limit=50, per_entity=2) if r.via == "shares trade: Sample Painting"}
        self.assertEqual(trade, {"S-010", "PL-021"})  # a different type first, then the newest id
        room = {r.chunk.record_id for r in self.store.related("PL-001", limit=50, per_entity=2) if r.via == "shares room: Room 102"}
        self.assertEqual(room, {"PL-019", "PL-013"})  # three share the room; the newest two are shown

    def test_a_record_of_another_type_in_the_same_spec_section_is_reached_under_the_default_caps(self) -> None:
        # RFI-001 shares its spec section with 6+ other RFIs and with the submittal S-003; alphabetical order
        # used to cut S-003, the most useful link, because "S-" sorts after "RFI-".
        ids = [r.chunk.record_id for r in self.store.related("RFI-001")]
        self.assertIn("S-003", ids)

    def test_related_says_how_many_were_not_shown(self) -> None:
        capped = self.store.related("PL-001", limit=50, per_entity=2)
        self.assertEqual(capped.omitted["shares trade: Sample Painting"], 3)  # 5 share it, 2 shown
        self.assertEqual(capped.omitted["shares room: Room 102"], 1)
        limited = self.store.related("PL-001", limit=3, per_entity=50)
        self.assertGreater(limited.omitted["beyond the overall limit"], 0)
        self.assertEqual(self.store.related("CE-001").omitted, {})  # nothing was cut

    def test_a_direct_link_is_listed_before_records_that_only_share_an_entity(self) -> None:
        # RFI-004 has a direct link (CE-003) and shares spec 09 21 16 with others; "related_rfi" sorts AFTER
        # "references_spec" alphabetically, so only an explicit direct-first rule gets this right.
        vias = [r.via for r in self.store.related("RFI-004", limit=50, per_entity=50)]
        self.assertEqual(vias[0], "related_rfi")
        self.assertTrue(all(v.startswith("shares ") for v in vias[1:]))

    def test_related_shows_only_the_latest_revision_of_a_submittal(self) -> None:
        s004 = [r for r in self.store.related("PL-004", limit=50, per_entity=50) if r.chunk.record_id == "S-004"]
        self.assertEqual([r.chunk.metadata["revision"] for r in s004], ["1"])

    def test_related_for_an_unknown_record_is_empty(self) -> None:
        self.assertEqual(self.store.related("RFI-999"), [])

    def test_related_ignores_case_in_the_record_id(self) -> None:
        self.assertEqual([r.chunk.key for r in self.store.related("pl-001", limit=5)],
                         [r.chunk.key for r in self.store.related("PL-001", limit=5)])

    def test_vector_search_puts_the_best_match_first(self) -> None:
        hits = self.store.vector_search(query_vector("abandoned conduit in the ceiling"), limit=5)
        self.assertIn("RFI-003", [h.chunk.record_id for h in hits[:3]])
        scores = [h.score for h in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertTrue(all(-1.0 <= s <= 1.0001 for s in scores))  # cosine similarity

    def test_vector_search_honours_limit_and_filters(self) -> None:
        self.assertEqual(len(self.store.vector_search(query_vector("paint"), limit=3)), 3)
        hits = self.store.vector_search(query_vector("paint"), limit=50, filters={"record_type": "punch_item", "status": "open"})
        self.assertTrue(hits)
        self.assertTrue(all(h.chunk.record_type == "punch_item" and h.chunk.metadata["status"] == "open" for h in hits))

    def test_keyword_search_finds_a_word_and_returns_nothing_for_nonsense(self) -> None:
        hits = self.store.keyword_search("asbestos", limit=5)
        self.assertEqual(hits[0].chunk.record_id, "CE-005")
        self.assertEqual(self.store.keyword_search("xyzzy plugh frobnicate", limit=5), [])
        self.assertEqual(self.store.keyword_search("", limit=5), [])

    def test_an_id_query_finds_only_records_that_contain_that_exact_id(self) -> None:
        # Not RFI-011, CE-003, S-003 or PL-003, which share a piece of the ID. Both stores must agree.
        found = {h.chunk.record_id for h in self.store.keyword_search("RFI-003", limit=50)}
        self.assertEqual(found, {"RFI-003", "CE-001"})  # CE-001 says "Related RFI: RFI-003"
        self.assertEqual(self.store.keyword_search("DR-2026-08-03", limit=5)[0].chunk.record_id, "DR-2026-08-03")

    def test_rare_words_outweigh_common_ones_in_keyword_ranking(self) -> None:
        # "abandoned" and "conduit" are in two records; "reading", "room" and "ceiling" are in many. Both
        # conduit records must be found in the top 3 even though other records match more of the common words.
        top = [h.chunk.record_id for h in self.store.keyword_search("abandoned conduit in the reading room ceiling", limit=3)]
        self.assertIn("RFI-003", top)
        self.assertIn("CE-001", top)

    def test_one_rare_word_outweighs_two_common_words_wherever_it_sits_in_the_question(self) -> None:
        # "work" and "performed" are in 32 records each (the daily reports), "conduit" in only two. Counting matched
        # words would put the daily reports first; scoring only the first word would put the conduit records last.
        for query in ("work performed conduit", "conduit work performed", "work conduit performed"):
            with self.subTest(query=query):
                top = {h.chunk.record_id for h in self.store.keyword_search(query, limit=2)}
                self.assertEqual(top, {"RFI-003", "CE-001"})

    def test_records_with_the_same_matched_words_are_ordered_by_how_much_the_text_is_about_them(self) -> None:
        # Many records contain "ceiling" and score the same, so without a second criterion the order would be
        # alphabetical by id: the earliest daily reports first and the two ceiling RFIs nowhere.
        top = [h.chunk.record_id for h in self.store.keyword_search("ceiling", limit=3)]
        self.assertIn("RFI-010", top)
        self.assertIn("RFI-003", top)

    def test_an_exact_amount_finds_the_record_that_states_it_however_it_is_typed(self) -> None:
        # CE-001 says "Estimated cost: 4850.00". The amount must be findable with or without cents, and by its
        # digits alone, in every store (PostgreSQL's text parser used to keep 4850.00 as ONE token).
        for query, expected in (("4850", "CE-001"), ("4850.00", "CE-001"), ("18500", "CE-004"), ("22300.00", "CE-005"),
                                ("which change event costs 22300.00", "CE-005"), ("12640", "CE-002")):
            with self.subTest(query=query):
                top = [h.chunk.record_id for h in self.store.keyword_search(query, limit=3)]
                self.assertIn(expected, top)

    def test_nonsense_and_common_words_alone_find_nothing(self) -> None:
        for query in ("recipe for chocolate cake", "who won the football game last night", "the of and"):
            with self.subTest(query=query):
                self.assertEqual(self.store.keyword_search(query, limit=5), [])

    def test_keyword_search_honours_filters_and_orders_best_first(self) -> None:
        hits = self.store.keyword_search("paint", limit=10, filters={"record_type": "punch_item"})
        self.assertTrue(hits)
        self.assertTrue(all(h.chunk.record_type == "punch_item" for h in hits))
        scores = [h.score for h in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_filters_ignore_case(self) -> None:
        exact = self.store.vector_search(query_vector("paint"), limit=50, filters={"responsible_party": "Sample Painting"})
        loose = self.store.vector_search(query_vector("paint"), limit=50, filters={"responsible_party": "sample PAINTING"})
        self.assertTrue(exact)
        self.assertEqual({h.chunk.key for h in exact}, {h.chunk.key for h in loose})

    def test_a_filter_with_no_match_gives_no_results(self) -> None:
        self.assertEqual(self.store.vector_search(query_vector("beam"), limit=5, filters={"status": "nonsense"}), [])
        self.assertEqual(self.store.keyword_search("beam", limit=5, filters={"status": "nonsense"}), [])

    def test_hostile_filter_keys_and_values_are_only_data(self) -> None:
        nasty = {"a'b\"; DROP TABLE chunks; --": "x' OR '1'='1"}
        self.assertEqual(self.store.vector_search(query_vector("beam"), limit=5, filters=nasty), [])
        self.assertEqual(self.store.keyword_search("beam", limit=5, filters=nasty), [])
        self.assertEqual(self.store.count(), 88)  # nothing was dropped

    def test_hostile_search_text_is_only_data(self) -> None:
        for text in ("'; DROP TABLE chunks; --", "rfi & !(", "\\x00 ' | & : * ( )", "a" * 5000):
            with self.subTest(text=text[:30]):
                self.store.keyword_search(text, limit=5)  # must not raise
        self.assertEqual(self.store.count(), 88)


    # ---- aggregates: exact numbers computed by the store, checked against the CSVs --------------------------
    def test_the_largest_crew_and_the_days_it_was_reached(self) -> None:
        crews = csv_values("daily_reports.csv", "report_date", "manpower_total")
        peak = max(crews.values())
        result = self.store.aggregate("daily_report", "manpower_total", "max")
        self.assertEqual(result.value, peak)
        self.assertEqual(set(result.record_ids), {f"DR-{day}" for day, n in crews.items() if n == peak})
        self.assertEqual(result.count, len(crews))
        self.assertEqual(result.more_ids, 0)

    def test_min_sum_avg_and_count_agree_with_the_csv(self) -> None:
        crews = list(csv_values("daily_reports.csv", "report_date", "manpower_total").values())
        self.assertEqual(self.store.aggregate("daily_report", "manpower_total", "min").value, min(crews))
        self.assertEqual(self.store.aggregate("daily_report", "manpower_total", "sum").value, sum(crews))
        self.assertEqual(self.store.aggregate("daily_report", "manpower_total", "count").value, len(crews))
        average = self.store.aggregate("daily_report", "manpower_total", "avg").value
        self.assertEqual(average, (Decimal(sum(crews)) / len(crews)).quantize(Decimal("0.01")))

    def test_only_sums_and_counts_and_averages_carry_no_record_ids(self) -> None:
        for operation in ("sum", "avg", "count"):
            with self.subTest(operation=operation):
                self.assertEqual(self.store.aggregate("daily_report", "manpower_total", operation).record_ids, ())

    def test_superseded_submittal_revisions_are_not_counted(self) -> None:
        rows = [r for r in load_csv(SAMPLE / "submittals.csv")]
        newest: dict[str, dict[str, str]] = {}
        for row in rows:
            if row["submittal_number"] not in newest or int(row["revision"]) > int(newest[row["submittal_number"]]["revision"]):
                newest[row["submittal_number"]] = row
        lead_times = {k: Decimal(v["lead_time_days"]) for k, v in newest.items() if v["lead_time_days"].strip()}
        result = self.store.aggregate("submittal", "lead_time_days", "max")
        self.assertEqual(result.value, max(lead_times.values()))
        self.assertEqual(set(result.record_ids), {k for k, n in lead_times.items() if n == max(lead_times.values())})
        self.assertEqual(self.store.aggregate("submittal", "lead_time_days", "count").value, len(lead_times))

    def test_a_filter_narrows_what_is_aggregated_and_ignores_case(self) -> None:
        costs = csv_values("change_events.csv", "event_number", "estimated_cost", keep=lambda r: r["status"] == "open")
        for status in ("open", "OPEN"):
            with self.subTest(status=status):
                result = self.store.aggregate("change_event", "estimated_cost", "sum", {"status": status})
                self.assertEqual(result.value, sum(costs.values()))
                self.assertEqual(result.count, len(costs))

    def test_nothing_to_aggregate_gives_no_value_and_a_zero_count(self) -> None:
        for operation in ("max", "min", "sum", "avg"):
            with self.subTest(operation=operation):
                result = self.store.aggregate("change_event", "estimated_cost", operation, {"status": "no-such-status"})
                self.assertIsNone(result.value)
                self.assertEqual((result.count, result.record_ids), (0, ()))
        self.assertEqual(self.store.aggregate("change_event", "estimated_cost", "count", {"status": "no-such-status"}).value, 0)

    def test_an_unknown_record_type_field_or_operation_is_refused_before_anything_runs(self) -> None:
        bad = [("invoice", "manpower_total", "max"), ("daily_report", "weather", "max"), ("daily_report", "manpower_total", "median"),
               ("daily_report", "manpower_total; DROP TABLE chunks; --", "max"), ("punch_item", "manpower_total", "max"),
               ("daily_report", "manpower_total", "max); DROP TABLE chunks; --")]
        for record_type, field, operation in bad:
            with self.subTest(record_type=record_type, field=field, operation=operation):
                with self.assertRaises(ValueError):
                    self.store.aggregate(record_type, field, operation)
        self.assertEqual(self.store.count(), 14 + 11 + 7 + 32 + 24)  # nothing was dropped

    def test_the_operations_are_the_five_the_tool_offers(self) -> None:
        self.assertEqual(AGGREGATE_OPERATIONS, ("max", "min", "sum", "avg", "count"))


def csv_values(name: str, id_column: str, field: str, keep: Callable[[dict[str, str]], bool] = lambda row: True) -> dict[str, Decimal]:
    """record id -> number, read straight from a CSV (an independent derivation of what aggregate must report)."""
    rows = load_csv(SAMPLE / name)
    return {row[id_column]: Decimal(row[field]) for row in rows if row[field].strip() and keep(row)}


def punch(record_id: str, room: str, trade: str, **meta: str) -> Chunk:
    return Chunk("punch_item", record_id, f"Punch item {record_id}. Room {room}.", {"is_latest": "yes", **meta},
                 relations=(("located_in", "room", room), ("responsible_party", "trade", trade)))


class CustomChunkContract(MixinBase):
    """Graph rules that need purpose-built data. Run against every store, with a store made from the chunks."""

    def make_store_from(self, chunks: list[Chunk]) -> Store:
        raise NotImplementedError

    def test_a_superseded_revision_creates_no_stale_connection(self) -> None:
        trade_old = (("responsible_party", "trade", "Old Co"),)
        trade_new = (("responsible_party", "trade", "New Co"),)
        # The newest revision is listed FIRST, so a lookup that keeps the last row would return the wrong one.
        chunks = [
            Chunk("submittal", "S-900", "rev one", {"revision": "1", "is_latest": "yes"}, relations=trade_new),
            Chunk("submittal", "S-900", "rev zero", {"revision": "0", "is_latest": "no"}, relations=trade_old),
            Chunk("punch_item", "PL-1", "an Old Co item", {"is_latest": "yes"}, relations=trade_old),
            Chunk("punch_item", "PL-2", "a New Co item", {"is_latest": "yes"}, relations=trade_new),
        ]
        store = self.make_store_from(chunks)
        self.assertEqual(store.related("PL-1"), [])  # Old Co no longer has the submittal
        self.assertEqual([(r.chunk.record_id, r.chunk.metadata["revision"]) for r in store.related("PL-2")], [("S-900", "1")])
        self.assertEqual([r.chunk.record_id for r in store.related("S-900")], ["PL-2"])

    def test_a_different_record_type_beats_a_newer_id_of_the_same_type(self) -> None:
        # PL-1 sorts BEFORE S-8 and S-9 by id, so only the "other types first" rule can put it on top.
        acme = (("responsible_party", "trade", "Acme"),)
        chunks = [Chunk("submittal", f"S-{n}", f"submittal {n}", {"is_latest": "yes", "revision": "0"}, relations=acme) for n in (5, 8, 9)]
        chunks.append(Chunk("punch_item", "PL-1", "a punch item", {"is_latest": "yes"}, relations=acme))
        store = self.make_store_from(chunks)
        self.assertEqual([r.chunk.record_id for r in store.related("S-5", per_entity=1)], ["PL-1"])
        self.assertEqual([r.chunk.record_id for r in store.related("S-5", per_entity=2)], ["PL-1", "S-9"])  # then the newest

    def test_the_per_entity_cap_applies_to_each_entity_separately(self) -> None:
        sheets = lambda *names: tuple(("references_sheet", "drawing_sheet", n) for n in names)  # noqa: E731
        chunks = [
            Chunk("rfi", "RFI-A", "a", {"is_latest": "yes"}, relations=sheets("A-101", "A-102")),
            Chunk("rfi", "RFI-B", "b", {"is_latest": "yes"}, relations=sheets("A-101")),
            Chunk("rfi", "RFI-C", "c", {"is_latest": "yes"}, relations=sheets("A-102")),
        ]
        found = {r.chunk.record_id for r in self.make_store_from(chunks).related("RFI-A", per_entity=1)}
        self.assertEqual(found, {"RFI-B", "RFI-C"})  # one from EACH sheet, not one in total

    def test_an_entity_shared_by_a_huge_number_of_records_is_not_listed(self) -> None:
        from rag.records.graph import HUB_DEGREE

        chunks = [punch(f"PL-{n:04d}", "Site", f"Trade {n}") for n in range(HUB_DEGREE + 5)]
        result = self.make_store_from(chunks).related("PL-0000")
        self.assertEqual([r.chunk.record_id for r in result], [])  # "everyone on site" tells the user nothing
        self.assertEqual(result.omitted["shares room: Site"], HUB_DEGREE + 4)

    def test_the_hub_limit_itself_is_still_listed_and_one_more_is_not(self) -> None:
        from rag.records.graph import HUB_DEGREE

        def others_in_the_room(count: int) -> RelatedRecords:
            chunks = [punch(f"PL-{n:04d}", "Site", f"Trade {n}") for n in range(count + 1)]  # PL-0000 is the seed
            return self.make_store_from(chunks).related("PL-0000", per_entity=5)

        at_limit = others_in_the_room(HUB_DEGREE)
        self.assertEqual(len(at_limit), 5)  # listed, up to the cap
        self.assertEqual(at_limit.omitted["shares room: Site"], HUB_DEGREE - 5)
        over_limit = others_in_the_room(HUB_DEGREE + 1)
        self.assertEqual(len(over_limit), 0)  # a hub: nothing is listed
        self.assertEqual(over_limit.omitted["shares room: Site"], HUB_DEGREE + 1)

    def test_a_record_that_reaches_an_entity_by_two_relations_counts_once(self) -> None:
        twice = (("references_sheet", "drawing_sheet", "A-101"), ("shows_sheet", "drawing_sheet", "A-101"))
        chunks = [Chunk("rfi", "RFI-A", "a", {"is_latest": "yes"}, relations=(("references_sheet", "drawing_sheet", "A-101"),)),
                  Chunk("rfi", "RFI-B", "b", {"is_latest": "yes"}, relations=twice)]
        result = self.make_store_from(chunks).related("RFI-A", per_entity=1)
        self.assertEqual([r.chunk.record_id for r in result], ["RFI-B"])
        self.assertEqual(result.omitted, {})  # RFI-B is one record, however many ways it points at the sheet

    def test_aggregate_lists_at_most_ten_tied_records_and_says_how_many_more(self) -> None:
        chunks = [Chunk("daily_report", f"DR-2026-01-{d:02d}", "d", {"is_latest": "yes", "manpower_total": "30"}) for d in range(1, 13)]
        chunks.append(Chunk("daily_report", "DR-2026-02-01", "d", {"is_latest": "yes", "manpower_total": "10"}))
        result = self.make_store_from(chunks).aggregate("daily_report", "manpower_total", "max")
        self.assertEqual(MAX_AGGREGATE_IDS, 10)
        self.assertEqual(result.record_ids, tuple(f"DR-2026-01-{d:02d}" for d in range(1, 11)))  # alphabetical, 10 of them
        self.assertEqual(result.more_ids, 2)
        self.assertEqual(result.count, 13)

    def test_an_old_revision_never_counts_even_when_it_is_listed_after_the_current_one(self) -> None:
        chunks = [Chunk("submittal", "S-1", "rev 1", {"is_latest": "yes", "revision": "1", "lead_time_days": "10"}),
                  Chunk("submittal", "S-1", "rev 0", {"is_latest": "no", "revision": "0", "lead_time_days": "99"})]
        store = self.make_store_from(chunks)
        result = store.aggregate("submittal", "lead_time_days", "max")
        self.assertEqual((result.value, result.record_ids, result.count), (Decimal("10"), ("S-1",), 1))
        self.assertEqual(store.aggregate("submittal", "lead_time_days", "sum").value, Decimal("10"))

    def test_words_inside_emails_file_names_and_paths_are_searchable(self) -> None:
        chunks = [Chunk("rfi", "RFI-1", "Contact john.smith@acme.com about sheet A-201.pdf and the file C:/plans/level2.dwg.",
                        {"is_latest": "yes"}),
                  Chunk("rfi", "RFI-2", "Nothing relevant here.", {"is_latest": "yes"})]
        store = self.make_store_from(chunks)
        for query in ("acme", "john", "smith", "pdf", "level2", "dwg", "plans"):
            with self.subTest(query=query):
                self.assertEqual([h.chunk.record_id for h in store.keyword_search(query, limit=5)], ["RFI-1"])

    def test_two_chunks_for_the_same_record_and_revision_are_refused_by_every_store(self) -> None:
        # PostgreSQL refuses them (one row per record and revision); the memory store used to keep both, so the same
        # data was accepted in one place and rejected in the other.
        twice = [Chunk("daily_report", "DR-2026-08-05", "first", {"is_latest": "yes"}),
                 Chunk("daily_report", "DR-2026-08-05", "second report the same day", {"is_latest": "yes"})]
        with self.assertRaises(StoreError):
            self.make_store_from(twice)

    def test_money_is_added_exactly_not_as_floating_point(self) -> None:
        chunks = [Chunk("change_event", f"CE-{n}", "e", {"is_latest": "yes", "estimated_cost": cost})
                  for n, cost in enumerate(("0.10", "0.20", "1000000.07"), start=1)]
        result = self.make_store_from(chunks).aggregate("change_event", "estimated_cost", "sum")
        self.assertEqual(result.value, Decimal("1000000.37"))

    def test_negative_numbers_and_records_without_a_value(self) -> None:
        chunks = [Chunk("change_event", "CE-1", "e", {"is_latest": "yes", "estimated_cost": "-500.00"}),
                  Chunk("change_event", "CE-2", "e", {"is_latest": "yes", "estimated_cost": "250.00"}),
                  Chunk("change_event", "CE-3", "e", {"is_latest": "yes"})]  # no cost recorded
        store = self.make_store_from(chunks)
        minimum = store.aggregate("change_event", "estimated_cost", "min")
        self.assertEqual((minimum.value, minimum.record_ids), (Decimal("-500.00"), ("CE-1",)))
        self.assertEqual(store.aggregate("change_event", "estimated_cost", "count").value, 2)  # CE-3 has no value to count
        self.assertEqual(store.aggregate("change_event", "estimated_cost", "avg").value, Decimal("-125.00"))

    def test_the_average_is_rounded_to_cents(self) -> None:
        chunks = [Chunk("daily_report", f"DR-2026-01-0{d}", "d", {"is_latest": "yes", "manpower_total": n})
                  for d, n in ((1, "10"), (2, "10"), (3, "11"))]
        self.assertEqual(self.make_store_from(chunks).aggregate("daily_report", "manpower_total", "avg").value, Decimal("10.33"))

    def test_a_value_that_is_not_a_number_in_the_metadata_is_ignored(self) -> None:
        chunks = [Chunk("daily_report", "DR-2026-01-01", "d", {"is_latest": "yes", "manpower_total": "12"}),
                  Chunk("daily_report", "DR-2026-01-02", "d", {"is_latest": "yes", "manpower_total": "lots"})]
        result = self.make_store_from(chunks).aggregate("daily_report", "manpower_total", "max")
        self.assertEqual((result.value, result.count), (Decimal("12"), 1))

    def test_linked_only_returns_records_not_entities_that_happen_to_share_a_name(self) -> None:
        # A record whose id equals an entity's key must not be mistaken for that entity.
        chunks = [punch("PL-1", "Room 1", "Sample Painting"),
                  Chunk("punch_item", "sample painting", "an odd record named like a trade", {"is_latest": "yes"})]
        self.assertEqual(self.make_store_from(chunks).linked("PL-1"), [])


class MemoryStoreContract(StoreContract, unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.store = build_index(SAMPLE, hashing_embed, "hashing")


class MemoryCustomChunkTests(CustomChunkContract, unittest.TestCase):
    def make_store_from(self, chunks: list[Chunk]) -> Store:
        return make_index(chunks, hashing_embed, "hashing")


@requires_database
class PostgresCustomChunkTests(CustomChunkContract, unittest.TestCase):
    def make_store_from(self, chunks: list[Chunk]) -> Store:
        schema = temporary_schema()
        conn, _ = schema.__enter__()
        self.addCleanup(schema.__exit__, None, None, None)
        apply_migrations(conn, embedding_dim=DIM)
        store = PostgresStore(conn, "custom")
        store.sync(chunks, hashing_embed, "hashing")
        return store


@requires_database
class PostgresStoreContract(StoreContract, unittest.TestCase):
    _schema: Any  # the open temporary_schema() context, closed in tearDownClass

    @classmethod
    def setUpClass(cls) -> None:
        cls._schema = temporary_schema()
        conn, _ = cls._schema.__enter__()
        apply_migrations(conn, embedding_dim=DIM)
        store = PostgresStore(conn, "library-renovation")
        store.sync(load_chunks(SAMPLE), hashing_embed, "hashing")
        cls.store = store

    @classmethod
    def tearDownClass(cls) -> None:
        cls._schema.__exit__(None, None, None)


if __name__ == "__main__":
    unittest.main()
