import json
import tempfile
import unittest
from pathlib import Path

from rag.providers.embed import hashing_embed
from rag.records.chunker import Chunk, row_to_chunk
from rag.records.graph import (
    ENTITY_KINDS,
    EXPANSION_EXCLUDED_RELATIONS,
    HUB_DEGREE,
    Graph,
    build_graph,
    entity_key,
    extract_relations,
    normalise_spec,
)
from rag.storage.store import build_index, chunks_fingerprint, load_chunks, load_index, save_index

SAMPLE = Path("data/records/samples/library-renovation")


class ExtractRelationsTests(unittest.TestCase):
    def test_an_rfi_links_to_its_sheets_spec_sections_and_the_party_holding_it(self) -> None:
        row = {"rfi_number": "RFI-003", "references": "A-201, spec 09 21 16, S-103", "ball_in_court": "Architect"}
        self.assertEqual(set(extract_relations("rfi", row)), {
            ("references_sheet", "drawing_sheet", "A-201"),
            ("references_sheet", "drawing_sheet", "S-103"),
            ("references_spec", "spec_section", "09 21 16"),
            ("ball_in_court", "party", "Architect"),
        })

    def test_a_submittal_links_to_its_spec_section_its_trade_and_the_party_holding_it(self) -> None:
        row = {"submittal_number": "S-001", "spec_section": "05 50 00", "responsible_party": "Sample Metals", "ball_in_court": "Architect"}
        self.assertEqual(set(extract_relations("submittal", row)), {
            ("in_spec_section", "spec_section", "05 50 00"),
            ("responsible_party", "trade", "Sample Metals"),
            ("ball_in_court", "party", "Architect"),
        })

    def test_a_punch_item_links_to_its_room_and_its_trade(self) -> None:
        row = {"item_number": "PL-001", "location": "Room 102", "responsible_party": "Sample Painting"}
        self.assertEqual(set(extract_relations("punch_item", row)), {
            ("located_in", "room", "Room 102"),
            ("responsible_party", "trade", "Sample Painting"),
        })

    def test_a_change_event_links_to_its_change_order(self) -> None:
        row = {"event_number": "CE-001", "related_rfi": "RFI-003", "change_order_number": "CO-001"}
        self.assertEqual(set(extract_relations("change_event", row)), {("issued_as", "change_order", "CO-001")})
        # related_rfi is a link between two records, handled separately from entities

    def test_a_daily_report_has_no_entity_links(self) -> None:
        self.assertEqual(extract_relations("daily_report", {"report_date": "2026-08-03", "work_performed": "Framing"}), ())

    def test_blank_fields_make_no_relation_and_names_are_cleaned_up(self) -> None:
        self.assertEqual(extract_relations("punch_item", {"location": "  ", "responsible_party": ""}), ())
        cleaned = extract_relations("punch_item", {"location": "Room   102 ", "responsible_party": "  Sample\tPainting"})
        self.assertEqual(set(cleaned), {("located_in", "room", "Room 102"), ("responsible_party", "trade", "Sample Painting")})

    def test_sheet_numbers_are_read_in_any_case_and_deduplicated(self) -> None:
        relations = extract_relations("rfi", {"references": "a-201; A-201, s-103"})
        self.assertEqual(sorted(r[2] for r in relations if r[0] == "references_sheet"), ["A-201", "S-103"])

    def test_only_real_entity_kinds_are_produced(self) -> None:
        for record_type, row in (("rfi", {"references": "A-1 spec 01 02 03", "ball_in_court": "X"}),
                                 ("submittal", {"spec_section": "1", "responsible_party": "Y"}),
                                 ("punch_item", {"location": "Z", "responsible_party": "Y"}),
                                 ("change_event", {"change_order_number": "CO-9"})):
            for _, kind, _ in extract_relations(record_type, row):
                self.assertIn(kind, ENTITY_KINDS)

    def test_entity_keys_ignore_case_and_spacing(self) -> None:
        self.assertEqual(entity_key("Sample  Painting "), entity_key("sample painting"))
        self.assertEqual(entity_key("Sample Painting"), "sample painting")

    def test_big_hub_relations_are_not_used_for_expansion(self) -> None:
        self.assertIn("ball_in_court", EXPANSION_EXCLUDED_RELATIONS)  # "everything the Architect holds" is not a neighbour


class SpecSectionTests(unittest.TestCase):
    def test_spec_sections_are_normalised_whatever_way_they_are_written(self) -> None:
        for written in ("09 21 16", "092116", "09-21-16", "09 21 16 - Gypsum Board", " 09  21 16 "):
            with self.subTest(written=written):
                self.assertEqual(normalise_spec(written), "09 21 16")
        self.assertIsNone(normalise_spec("Division 9"))

    def test_a_submittal_and_an_rfi_meet_at_the_same_spec_section_however_it_is_written(self) -> None:
        submittal = extract_relations("submittal", {"spec_section": "092116"})
        rfi = extract_relations("rfi", {"references": "see spec 09-21-16 and A-201"})
        self.assertIn(("in_spec_section", "spec_section", "09 21 16"), submittal)
        self.assertIn(("references_spec", "spec_section", "09 21 16"), rfi)

    def test_free_text_does_not_turn_a_date_into_a_spec_section(self) -> None:
        self.assertEqual([r for r in extract_relations("rfi", {"references": "issued 2026-09-20"}) if r[1] == "spec_section"], [])

    def test_formats_that_are_not_recognised_yet_are_listed_here_so_the_limit_is_known(self) -> None:
        # Real drawing sets use many numbering styles. Only the ones below are understood today.
        for unsupported in ("A2.01", "A201", "M-1", "E-1.1"):
            with self.subTest(sheet=unsupported):
                self.assertEqual([r for r in extract_relations("rfi", {"references": unsupported}) if r[1] == "drawing_sheet"], [])


class ChunkRelationTests(unittest.TestCase):
    def test_a_chunk_carries_the_relations_of_its_row(self) -> None:
        row = {"project_id": "P", "item_number": "PL-001", "location": "Room 102", "description": "d",
               "responsible_party": "Sample Painting", "created_date": "x", "status": "open"}
        chunk = row_to_chunk("punch_item", row)
        self.assertIn(("located_in", "room", "Room 102"), chunk.relations)

    def test_the_fingerprint_changes_when_only_the_relations_change(self) -> None:
        a = Chunk("punch_item", "PL-1", "same", {"x": "1"}, relations=(("located_in", "room", "Room 1"),))
        b = Chunk("punch_item", "PL-1", "same", {"x": "1"}, relations=(("located_in", "room", "Room 2"),))
        self.assertNotEqual(chunks_fingerprint([a]), chunks_fingerprint([b]))

    def test_relations_survive_saving_and_loading_the_index_cache(self) -> None:
        index = build_index(SAMPLE, hashing_embed, "hashing")
        with tempfile.TemporaryDirectory() as tmp:
            save_index(index, Path(tmp) / "i.json")
            loaded = load_index(Path(tmp) / "i.json")
        before = {c.key: c.relations for c in index.chunks}
        after = {c.key: c.relations for c in loaded.chunks}
        self.assertEqual(before, after)
        self.assertTrue(all(isinstance(r, tuple) for rels in after.values() for r in rels))
        json.dumps(after["PL-001", "0"])  # plain data, so it can be written as JSON


class SheetExtractionTests(unittest.TestCase):
    def sheets(self, references: str) -> list[str]:
        return [r[2] for r in extract_relations("rfi", {"references": references}) if r[0] == "references_sheet"]

    def test_real_sheet_numbers_are_found(self) -> None:
        self.assertEqual(self.sheets("A-201, spec 09 21 16"), ["A-201"])
        self.assertEqual(self.sheets("A-201.pdf, M-401 and S-103"), ["A-201", "M-401", "S-103"])

    def test_a_record_id_mentioned_in_the_references_is_not_a_drawing_sheet(self) -> None:
        # Two-letter prefixes such as CE-001 and PL-007 fit the sheet pattern but name records.
        self.assertEqual(self.sheets("see CE-001 and PL-007, also DR-2026 and CO-001, sheet A-201"), ["A-201"])


class BuildGraphTests(unittest.TestCase):
    chunks: list[Chunk]  # set in setUpClass
    graph: Graph

    @classmethod
    def setUpClass(cls) -> None:
        cls.chunks = load_chunks(SAMPLE)
        cls.graph = build_graph(cls.chunks)

    def test_there_is_one_node_per_record_and_one_per_entity(self) -> None:
        records = {(n.kind, n.key) for n in self.graph.nodes if n.kind not in ENTITY_KINDS}
        self.assertEqual(len(records), 86)  # 88 chunks; S-001 and S-004 have two revisions each
        painting = [n for n in self.graph.nodes if n.kind == "trade" and n.key == "sample painting"]
        self.assertEqual(len(painting), 1)
        self.assertEqual(painting[0].label, "Sample Painting")

    def test_entity_names_that_differ_only_in_case_or_spacing_become_one_node(self) -> None:
        a = row_to_chunk("punch_item", {"project_id": "P", "item_number": "PL-1", "location": "Room 1", "description": "d",
                                         "responsible_party": "Sample Painting", "created_date": "x", "status": "open"})
        b = row_to_chunk("punch_item", {"project_id": "P", "item_number": "PL-2", "location": "room  1", "description": "d",
                                         "responsible_party": "sample   painting", "created_date": "x", "status": "open"})
        graph = build_graph([a, b])
        self.assertEqual(sum(1 for n in graph.nodes if n.kind == "trade"), 1)
        self.assertEqual(sum(1 for n in graph.nodes if n.kind == "room"), 1)

    def test_related_rfi_makes_a_record_to_record_edge_only_when_both_exist(self) -> None:
        pairs = {(e.src_key, e.dst_key) for e in self.graph.edges if e.relation == "related_rfi"}
        self.assertEqual(pairs, {("CE-001", "RFI-003"), ("CE-003", "RFI-004"), ("CE-004", "RFI-011"), ("CE-007", "RFI-013")})
        orphan = Chunk("change_event", "CE-900", "t", {"related_rfi": "RFI-404"})
        self.assertEqual([e for e in build_graph([orphan]).edges if e.relation == "related_rfi"], [])

    def test_the_rfi_a_change_event_names_matches_whatever_the_case(self) -> None:
        rfi = Chunk("rfi", "RFI-003", "t", {})
        event = Chunk("change_event", "CE-001", "t", {"related_rfi": " rfi-003 "})
        self.assertEqual([(e.src_key, e.dst_key) for e in build_graph([rfi, event]).edges], [("CE-001", "RFI-003")])

    def test_a_punch_item_points_at_its_room_and_trade(self) -> None:
        edges = {(e.relation, e.dst_kind, e.dst_key) for e in self.graph.edges if e.src_key == "PL-001"}
        self.assertEqual(edges, {("located_in", "room", "room 102"), ("responsible_party", "trade", "sample painting")})

    def test_edges_are_not_duplicated_for_a_submittal_with_two_revisions(self) -> None:
        s001 = [e for e in self.graph.edges if e.src_key == "S-001" and e.relation == "in_spec_section"]
        self.assertEqual(len(s001), 1)

    def test_a_superseded_revision_contributes_no_edges_and_no_stale_entities(self) -> None:
        old = Chunk("submittal", "S-900", "rev 0", {"revision": "0", "is_latest": "no"}, relations=(("responsible_party", "trade", "Old Co"),))
        new = Chunk("submittal", "S-900", "rev 1", {"revision": "1", "is_latest": "yes"}, relations=(("responsible_party", "trade", "New Co"),))
        graph = build_graph([old, new])
        self.assertEqual([(e.relation, e.dst_key) for e in graph.edges], [("responsible_party", "new co")])
        self.assertNotIn("old co", {n.key for n in graph.nodes})  # no orphan entity left from the old revision
        self.assertIn(("submittal", "S-900"), {(n.kind, n.key) for n in graph.nodes})  # the record itself stays

    def test_edge_order_is_fully_determined(self) -> None:
        a = Chunk("rfi", "X-1", "t", {}, relations=(("references_sheet", "drawing_sheet", "A-101"), ("references_sheet", "drawing_sheet", "A-102")))
        self.assertEqual(build_graph([a]).edges, build_graph([a]).edges)

    def test_a_hub_is_a_known_size(self) -> None:
        self.assertGreaterEqual(HUB_DEGREE, 100)  # big enough that a normal trade is not a hub

    def test_every_edge_ends_at_a_node_that_exists(self) -> None:
        nodes = {(n.kind, n.key) for n in self.graph.nodes}
        for edge in self.graph.edges:
            self.assertIn((edge.src_kind, edge.src_key), nodes)
            self.assertIn((edge.dst_kind, edge.dst_key), nodes)


if __name__ == "__main__":
    unittest.main()
