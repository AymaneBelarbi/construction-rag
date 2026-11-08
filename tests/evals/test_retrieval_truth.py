"""Every retrieval label is derived from specific CSV columns and compared with the label in
evals/retrieval_cases.py, so a label cannot drift from the data or be copied from the text being searched."""
import unittest
from pathlib import Path

from evals.retrieval_cases import CASES
from rag.records.loader import load_csv
from tests.support.db_support import skip_unless

SAMPLE = Path("data/records/samples/library-renovation")


def rows(name: str) -> list[dict[str, str]]:
    return load_csv(SAMPLE / name)


def derive() -> dict[str, set[str]]:
    rfis, events, subs = rows("rfis.csv"), rows("change_events.csv"), rows("submittals.csv")
    reports, punch = rows("daily_reports.csv"), rows("punch_items.csv")

    def rfi_ids(word: str) -> set[str]:
        return {r["rfi_number"] for r in rfis if word in r["subject"].lower()}

    def event_ids(word: str) -> set[str]:
        return {e["event_number"] for e in events if word in e["title"].lower()}

    def submittal_ids(word: str) -> set[str]:
        return {s["submittal_number"] for s in subs if word in s["title"].lower()}

    pl001 = next(p for p in punch if p["item_number"] == "PL-001")
    s004 = next(s for s in subs if s["submittal_number"] == "S-004")
    return {
        "id-rfi": {"RFI-011"}, "id-daily": {"DR-2026-09-10"}, "id-submittal": {"S-004"},
        "id-change-event": {"CE-005"}, "id-punch": {"PL-017"},
        "amount-with-cents": {e["event_number"] for e in events if e["estimated_cost"] == "4850.00"},
        "amount-digits-only": {e["event_number"] for e in events if e["estimated_cost"] and float(e["estimated_cost"]) == 18500},
        "topic-conduit": rfi_ids("conduit") | event_ids("conduit"),
        "topic-asbestos": event_ids("asbestos"),
        "topic-rooftop": rfi_ids("rooftop unit") | event_ids("rooftop unit") | submittal_ids("rooftop unit"),
        "sem-hazard": event_ids("asbestos"),
        "topic-inspection": {f"DR-{r['report_date']}" for r in reports if "passed" in r["inspections"].lower() and "ceiling" in r["inspections"].lower()},
        "topic-rain": {f"DR-{r['report_date']}" for r in reports if "rain" in r["weather"].lower()},
        "sem-sprinkler": rfi_ids("sprinkler") | event_ids("sprinkler"),
        "topic-skylight": rfi_ids("skylight"),
        "partial-leveling": rfi_ids("leveling") | event_ids("leveling"),
        "partial-hardware": rfi_ids("door hardware") | submittal_ids("door hardware"),
        "sem-firewall": rfi_ids("fire rating"),
        "sem-bookcases": rfi_ids("shelving"),
        "sem-heaters": rfi_ids("radiators"),
        "sem-sockets": event_ids("data outlets"),
        "filter-punch": {p["item_number"] for p in punch if p["status"] == "ready_for_review" and "screw pops" in p["description"].lower()},
        "filter-open-rfi": {r["rfi_number"] for r in rfis if r["status"] == "open" and "ceiling" in r["subject"].lower()},
        "neg-gibberish": set(), "neg-recipe": set(), "neg-sport": set(),
        "adj-owner-budget": set(), "adj-over-budget": set(), "adj-architect-staff": set(),
        "graph-trade": {p["item_number"] for p in punch if p["responsible_party"] == pl001["responsible_party"]} - {"PL-001"}
                       | {s["submittal_number"] for s in subs if s["responsible_party"] == pl001["responsible_party"]},
        "graph-room": {p["item_number"] for p in punch if p["location"] == pl001["location"]} - {"PL-001"},
        "graph-link": {e["event_number"] for e in events if e["related_rfi"] == "RFI-011"},
        "graph-flooring": {p["item_number"] for p in punch if p["responsible_party"] == s004["responsible_party"]},
    }


@skip_unless(SAMPLE.exists(), "clone the records repo into data/records first (see README)")
class RetrievalLabelsMatchTheData(unittest.TestCase):
    truth: dict[str, set[str]]  # set in setUpClass

    @classmethod
    def setUpClass(cls) -> None:
        cls.truth = derive()

    def test_every_case_has_a_derivation_and_nothing_else_does(self) -> None:
        self.assertEqual(set(self.truth), {case.id for case in CASES})

    def test_each_label_equals_what_the_data_says(self) -> None:
        for case in CASES:
            with self.subTest(case=case.id):
                self.assertEqual(set(case.relevant), self.truth[case.id])

    def test_case_ids_are_unique_and_every_kind_is_used(self) -> None:
        ids = [case.id for case in CASES]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual({c.kind for c in CASES}, {"exact", "topic", "partial", "semantic", "filtered", "negative", "adjacent", "graph"})

    def test_graph_cases_name_a_seed_and_others_do_not(self) -> None:
        for case in CASES:
            self.assertEqual(case.seed is not None, case.kind == "graph", case.id)

    def test_the_negative_questions_really_are_about_nothing_in_the_records(self) -> None:
        everything = " ".join(" ".join(r.values()) for name in ("rfis.csv", "submittals.csv", "change_events.csv", "daily_reports.csv", "punch_items.csv") for r in rows(name)).lower()
        for case in (c for c in CASES if c.kind == "negative"):
            words = [w for w in case.query.lower().split() if w not in ("for", "who", "the", "last", "won")]
            with self.subTest(case=case.id):
                self.assertTrue(all(w not in everything for w in words), words)

    def test_a_semantic_query_shares_no_content_word_with_the_records_it_should_find(self) -> None:
        # What makes a case "semantic" is that it is a paraphrase. A shared word (even "ceiling") lets keyword search
        # find the record, and the case would quietly stop measuring meaning. Words of 4 letters or more count.
        import re

        stopwords = {"that", "with", "were", "what", "which", "where", "when", "does", "have", "this", "from", "much", "long", "should", "must"}
        files = {"rfi": "rfis.csv", "change_event": "change_events.csv", "submittal": "submittals.csv",
                 "daily_report": "daily_reports.csv", "punch_item": "punch_items.csv"}
        text_of: dict[str, str] = {}
        for name in files.values():
            for row in rows(name):
                key = next((row[c] for c in ("rfi_number", "event_number", "submittal_number", "item_number") if c in row), None)
                key = key or f"DR-{row['report_date']}"
                text_of[key] = text_of.get(key, "") + " " + " ".join(row.values()).lower()
        for case in (c for c in CASES if c.kind == "semantic"):
            query_words = set(re.findall(r"[a-z]{4,}", case.query.lower())) - stopwords
            for record_id in case.relevant:
                shared = query_words & set(re.findall(r"[a-z]{4,}", text_of[record_id]))
                with self.subTest(case=case.id, record=record_id):
                    self.assertEqual(shared, set())

    def test_semantic_cases_are_the_paraphrases_and_the_lexical_ones_are_topic_cases(self) -> None:
        self.assertEqual({c.id for c in CASES if c.kind == "semantic"},
                         {"sem-hazard", "sem-sprinkler", "sem-firewall", "sem-bookcases", "sem-heaters", "sem-sockets"})


if __name__ == "__main__":
    unittest.main()
