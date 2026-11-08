"""Check the golden questions against the data, in both directions.

For each question the TRUE answer is derived from the CSVs and the cpr checker (never copied from the
question set), then these tests assert that the question's own spec accepts it:
  - answer_ids / must_cite equal the derived IDs,
  - every fact group matches at least one true statement,
  - no forbidden pattern matches a true statement.
So if someone edits a question's expectation (or the data changes), a test fails. A meta-test makes
sure no question can be added without a truth entry.

The tests run the local checker (python -m cpr, a subprocess). They never touch the network."""
import re
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evals.questions import AS_OF, INJECTION_PROJECT, INJECTION_QUESTIONS, QUESTIONS, SAMPLE_PROJECT
from rag.agent import make_cpr_runner
from rag.providers.embed import hashing_embed
from rag.records.loader import load_csv
from rag.storage.store import build_index
from tests.support.db_support import skip_unless

RECORDS = Path("data/records")
SAMPLE = RECORDS / SAMPLE_PROJECT
ALL_QUESTIONS = {q.id: q for q in (*QUESTIONS, *INJECTION_QUESTIONS)}


@dataclass
class Truth:
    ids: set[str] | None = None  # the exact set the IDS line must hold (list questions), else None
    cites: set[str] = field(default_factory=set)  # records an answer must cite
    statements: list[str] = field(default_factory=list)  # true sentences, written from the data


def matches_any(patterns: tuple[str, ...], text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def rows(folder: Path, name: str) -> list[dict[str, str]]:
    return load_csv(folder / name)


def checker_lines(output: str, label: str) -> str:
    match = re.search(rf"^\s*{re.escape(label)}\s+(.+)$", output, re.MULTILINE)
    assert match, f"checker output has no '{label}' line"
    return match.group(1)


def derive_golden() -> dict[str, Truth]:
    checker = make_cpr_runner(RECORDS, SAMPLE_PROJECT)(AS_OF)
    rfis, events = rows(SAMPLE, "rfis.csv"), rows(SAMPLE, "change_events.csv")
    reports, punch = rows(SAMPLE, "daily_reports.csv"), rows(SAMPLE, "punch_items.csv")
    submittals = rows(SAMPLE, "submittals.csv")

    overdue = dict(re.findall(r"(RFI-\d+) \((\d+) days\)", checker_lines(checker, "most overdue")))
    at_risk = dict(re.findall(r"(S-\d+) \((\d+) days\)", checker_lines(checker, "procurement at risk")))
    rfi004 = next(r for r in rfis if r["rfi_number"] == "RFI-004")
    linked = [e for e in events if e["related_rfi"] == "RFI-011"]
    s001_latest = max((r for r in submittals if r["submittal_number"] == "S-001"), key=lambda r: int(r["revision"]))
    exposure_cost = float(checker_lines(checker, "open exposure cost"))
    exposure_days = int(checker_lines(checker, "open exposure days"))
    ready = {r["item_number"] for r in punch if r["status"] == "ready_for_review"}
    asbestos = [e for e in events if "asbestos" in e["title"].lower()]
    inspection = next(r for r in reports if "reading room" in r["inspections"].lower())
    rain = [r for r in reports if "rain" in r["weather"].lower()]
    tied = [e for e in events if e["status"] == "open" and e["related_rfi"] in overdue]
    architect = [r for r in rfis if r["status"] == "open" and r["ball_in_court"] == "Architect"]
    newest: dict[str, dict[str, str]] = {}
    for r in submittals:
        if r["submittal_number"] not in newest or int(r["revision"]) > int(newest[r["submittal_number"]]["revision"]):
            newest[r["submittal_number"]] = r
    longest = max((r for r in newest.values() if r["lead_time_days"]), key=lambda r: int(r["lead_time_days"]))
    event_days = sum(int(e["estimated_days"]) for e in events if e["estimated_days"])
    counts = {r["report_date"]: int(r["manpower_total"]) for r in reports if r["manpower_total"]}
    peak = max(counts.values())
    peak_dates = sorted(d for d, n in counts.items() if n == peak)

    return {
        "lookup-rfi": Truth(cites={"RFI-004"}, statements=[f"RFI-004 is {rfi004['status']} and its cost impact is {rfi004['cost_impact']}."]),
        "link-ce": Truth(cites={e["event_number"] for e in linked},
                         statements=[f"{e['event_number']} is estimated at ${float(e['estimated_cost']):,.2f}." for e in linked]),
        "superseded": Truth(cites={"S-001"}, statements=[f"S-001 is currently {s001_latest['status'].replace('_', ' ')}."]),
        "overdue-rfi": Truth(ids=set(overdue), statements=[f"{r} is {d} days overdue." for r, d in overdue.items()]),
        "exposure": Truth(statements=[f"Open exposure is ${exposure_cost:,.2f} and {exposure_days} days."]),
        "sub-risk": Truth(ids=set(at_risk), statements=[f"{s} is {d} days past its approval date." for s, d in at_risk.items()]),
        "punch-rfr": Truth(ids=ready, statements=[f"{i} is ready for review." for i in sorted(ready)]),
        "hazmat": Truth(ids={e["event_number"] for e in asbestos}, statements=[e["title"] for e in asbestos]),
        "inspection": Truth(cites={f"DR-{inspection['report_date']}"}, statements=[inspection["inspections"]]),
        "rain-days": Truth(ids={f"DR-{r['report_date']}" for r in rain}, statements=[r["weather"] for r in rain]),
        "multihop": Truth(ids={e["event_number"] for e in tied},
                          statements=[f"Together they add up to ${sum(float(e['estimated_cost']) for e in tied):,.0f} "
                                      f"and {sum(int(e['estimated_days']) for e in tied)} days."]),
        "architect": Truth(ids={r["rfi_number"] for r in architect}, statements=[f"{r['rfi_number']} is with the architect." for r in architect]),
        "peak-manpower": Truth(statements=[f"The peak was {peak} workers on {d}." for d in peak_dates]),
        "longest-lead": Truth(statements=[f"{longest['submittal_number']} has the longest lead time, {longest['lead_time_days']} days."]),
        "total-event-days": Truth(statements=[f"All the change events add up to {event_days} days."]),
        # Abstentions: the truth is that the record does not exist; the statements are honest ways to say so.
        "no-such-ce": Truth(statements=["I found no record of CE-009 in the logs."]),
        "false-premise": Truth(statements=["RFI-099 isn't in the log, so I can't verify that."]),
        "out-of-scope": Truth(statements=["The project logs don't name a fire marshal."]),
    }


def derive_injection() -> dict[str, Truth]:
    checker = make_cpr_runner(RECORDS, INJECTION_PROJECT.resolve())(AS_OF)
    overdue = dict(re.findall(r"(RFI-\d+) \((\d+) days\)", checker_lines(checker, "most overdue")))
    rfi002 = next(r for r in rows(INJECTION_PROJECT, "rfis.csv") if r["rfi_number"] == "RFI-002")
    pl001 = next(r for r in rows(INJECTION_PROJECT, "punch_items.csv") if r["item_number"] == "PL-001")
    dr = next(r for r in rows(INJECTION_PROJECT, "daily_reports.csv") if r["report_date"] == "2026-08-03")
    return {
        "inj-overdue": Truth(ids=set(overdue), statements=[f"{r} is {d} days overdue." for r, d in overdue.items()]),
        "inj-status": Truth(cites={"RFI-002"}, statements=[f"RFI-002 is {rfi002['status']}."]),
        "inj-link": Truth(cites={"PL-001"}, statements=[pl001["description"].split(".")[0] + "."]),
        "inj-leak": Truth(cites={"DR-2026-08-03"}, statements=[dr["work_performed"].split(".")[0] + "."]),
    }


@skip_unless(SAMPLE.exists(), "clone the records repo into data/records first (see README)")
class GoldenAnswersMatchTheData(unittest.TestCase):
    truth: dict[str, Any]  # set in setUpClass

    @classmethod
    def setUpClass(cls) -> None:
        cls.truth = {**derive_golden(), **derive_injection()}

    def test_every_question_has_a_truth_entry_and_nothing_else_does(self) -> None:
        self.assertEqual(set(self.truth), set(ALL_QUESTIONS))

    def test_list_questions_expect_exactly_the_derived_ids(self) -> None:
        for qid, q in ALL_QUESTIONS.items():
            with self.subTest(question=qid):
                expected = self.truth[qid].ids
                self.assertEqual(set(q.answer_ids) if q.answer_ids is not None else None, expected)

    def test_required_citations_equal_the_derived_ones(self) -> None:
        for qid, q in ALL_QUESTIONS.items():
            with self.subTest(question=qid):
                self.assertEqual(set(q.must_cite), self.truth[qid].cites)

    def test_every_fact_group_accepts_a_true_statement(self) -> None:
        for qid, q in ALL_QUESTIONS.items():
            for group in q.facts:
                with self.subTest(question=qid, fact=group[0][:50]):
                    self.assertTrue(any(matches_any(group, s) for s in self.truth[qid].statements),
                                    f"none of {self.truth[qid].statements} matches {group}")

    def test_no_forbidden_pattern_matches_a_true_statement(self) -> None:
        for qid, q in ALL_QUESTIONS.items():
            for pattern in q.forbidden:
                for statement in self.truth[qid].statements:
                    with self.subTest(question=qid, pattern=pattern[:40]):
                        self.assertIsNone(re.search(pattern, statement, re.IGNORECASE), statement)

    def test_every_question_with_expectations_has_something_to_check_against(self) -> None:
        for qid, q in ALL_QUESTIONS.items():
            with self.subTest(question=qid):
                self.assertTrue(q.answer_ids or q.must_cite or q.facts, "a question with no expectations proves nothing")

    def test_every_expected_record_exists(self) -> None:
        known = {c.record_id for c in build_index(SAMPLE, hashing_embed, "h").chunks}
        known |= {c.record_id for c in build_index(INJECTION_PROJECT, hashing_embed, "h").chunks}
        for qid, q in ALL_QUESTIONS.items():
            for record_id in (*(q.answer_ids or ()), *q.must_cite):
                with self.subTest(question=qid, record=record_id):
                    self.assertIn(record_id, known)

    def test_the_abstain_questions_really_have_no_answer_in_the_data(self) -> None:
        index = build_index(SAMPLE, hashing_embed, "h")
        known = {c.record_id for c in index.chunks}
        self.assertNotIn("CE-009", known)
        self.assertNotIn("RFI-099", known)
        self.assertNotIn("fire marshal", " ".join(c.text for c in index.chunks).lower())

    def test_the_peak_is_a_tie_so_the_date_fact_must_accept_each_tied_date(self) -> None:
        counts = {r["report_date"]: int(r["manpower_total"]) for r in rows(SAMPLE, "daily_reports.csv") if r["manpower_total"]}
        tied = [d for d, n in counts.items() if n == max(counts.values())]
        self.assertGreater(len(tied), 1)
        date_group = ALL_QUESTIONS["peak-manpower"].facts[1]
        for date in tied:
            self.assertTrue(matches_any(date_group, date))


if __name__ == "__main__":
    unittest.main()
