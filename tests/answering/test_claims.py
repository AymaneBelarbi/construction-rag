import time
import unittest

from rag.answering.claims import find_unsupported_claims

RECORDS = {
    "CE-004": "Change event CE-004 (open). Title: Relocated rooftop unit support steel. Estimated cost: 18500.00. "
              "Estimated days: 5. Event date: 2026-08-20. Notice required by: 2026-09-03.",
    "CE-001": "Change event CE-001 (approved). Title: Abandoned conduit removal. Estimated cost: 4850.00. Estimated days: 0.",
    "CE-007": "Change event CE-007 (open). Title: Delay awaiting sprinkler main relocation answer. Estimated cost: 0. Estimated days: 3.",
    "DR-2026-08-05": "Daily report 2026-08-05. Weather: Dry. Manpower: 28. Author: Superintendent.",
    "RFI-010": "RFI RFI-010 (open). Subject: Acoustic panel mounting at reading room ceiling. Date needed: 2026-08-26.",
}
COMPUTED = "most overdue RFI-010 (26 days), RFI-011 (20 days)\nopen exposure cost 48720.00\nopen exposure days 15\nresubmitted percent 22.2"
CONTEXT = "Which RFIs are overdue? 2026-09-20"


def check(answer: str, records: dict[str, str] | None = None, computed: str = COMPUTED, context: str = CONTEXT) -> list[str]:
    return find_unsupported_claims(answer, RECORDS if records is None else records, computed, context)


class SupportedClaimsTests(unittest.TestCase):
    def test_a_figure_that_is_in_the_cited_record_is_supported(self) -> None:
        self.assertEqual(check("CE-004 is estimated at $18,500.00 [CE-004]."), [])
        self.assertEqual(check("It adds 5 days [CE-004]."), [])
        self.assertEqual(check("The peak was 28 workers [DR-2026-08-05]."), [])

    def test_money_matches_whatever_way_it_is_written(self) -> None:
        for written in ("$18,500.00", "$18,500", "$18500", "$18.5K", "$18.5k", "$ 18,500"):
            with self.subTest(written=written):
                self.assertEqual(check(f"It costs {written} [CE-004]."), [])

    def test_a_date_in_the_cited_record_is_supported(self) -> None:
        self.assertEqual(check("Notice was required by 2026-09-03 [CE-004]."), [])

    def test_report_numbers_cannot_be_supported_by_record_citation(self) -> None:
        self.assertEqual(len(check("RFI-010 is 26 days overdue [RFI-010].")), 1)
        self.assertEqual(len(check("Open exposure is $48,720.00 and 15 days [CE-004].")), 2)
        self.assertEqual(len(check("22.2 percent were resubmitted [RFI-010].")), 1)

    def test_user_figures_do_not_become_evidence(self) -> None:
        self.assertEqual(len(check("It took 12 days [CE-004].", context="it took 12 days")), 1)
        self.assertTrue(check("CE-004 costs $48,720 [CE-004].", context="Is CE-004 $48,720?"))
        self.assertEqual(find_unsupported_claims("As of 2026-09-20 it is open [CE-004].", RECORDS,
                                                 as_of="2026-09-20"), [])

    def test_zero_is_checked_like_other_values(self) -> None:
        self.assertEqual(check("CE-007 costs $0 [CE-007]."), [])
        self.assertEqual(len(check("CE-007 has 0 days [CE-007].")), 0)  # numeric match is not field-level proof
        self.assertEqual(len(check("CE-007 costs $0 [CE-007].", {"CE-007": "Estimated cost: not recorded."}, "", "")), 1)


class UnsupportedClaimsTests(unittest.TestCase):
    def test_a_wrong_figure_for_a_cited_record_is_reported_with_its_citation(self) -> None:
        found = check("CE-004 is estimated at $20,000 [CE-004].")
        self.assertEqual(len(found), 1)
        self.assertIn("$20,000", found[0])
        self.assertIn("CE-004", found[0])

    def test_a_figure_taken_from_another_record_than_the_one_cited_is_reported(self) -> None:
        # 28 workers is in DR-2026-08-05, not in the change event the sentence cites.
        found = check("CE-004 needed 28 workers [CE-004].")
        self.assertEqual(len(found), 1)
        self.assertIn("28 workers", found[0])

    def test_a_wrong_date_and_a_wrong_quantity_are_reported(self) -> None:
        self.assertEqual(len(check("Notice was required by 2026-09-04 [CE-004].")), 1)
        self.assertEqual(len(check("It adds 6 days [CE-004].")), 1)
        self.assertEqual(len(check("It adds 5 days [CE-007].")), 1)  # CE-007 is 3 days; 5 belongs to CE-004

    def test_every_wrong_figure_in_a_sentence_is_reported_once(self) -> None:
        found = check("It costs $20,000 and adds 9 days [CE-004]. Again $20,000 [CE-004].")
        self.assertEqual(len(found), 2)

    def test_a_figure_with_no_citation_must_exist_somewhere_in_the_evidence(self) -> None:
        self.assertEqual(check("The roof steel costs $18,500."), [])  # in CE-004, though not cited here
        found = check("The roof steel costs $19,000.")
        self.assertEqual(len(found), 1)
        self.assertIn("no citation", found[0])

    def test_a_citation_of_an_unknown_record_does_not_hide_an_invented_figure(self) -> None:
        # RFI-999 was never returned by a tool; its (missing) text supports nothing.
        self.assertEqual(len(check("It costs $77,000 [RFI-999].")), 1)

    def test_several_cited_records_together_can_support_the_figures(self) -> None:
        self.assertEqual(check("CE-004 adds 5 days and CE-007 adds 3 days [CE-004] [CE-007]."), [])
        self.assertEqual(check("They add 5 days and 3 days [CE-004, CE-007]."), [])


class CitationPlacementTests(unittest.TestCase):
    """Models write the citation AFTER the full stop: 'It costs $5. [CE-004]'. It belongs to the sentence before it."""

    def test_a_citation_after_the_full_stop_belongs_to_the_sentence_it_follows(self) -> None:
        # The real answer that was falsely flagged in the first live run (Haiku 5.5, lookup-rfi, run 2).
        records = {
            "RFI-004": "RFI RFI-004 (answered). Subject: Floor leveling compound depth. Answer: Response attached. Cost impact: possible. "
                       "Schedule impact: none.",
            "CE-003": "Change event CE-003 (submitted). Title: Extra floor leveling at stacks area. Estimated cost: 7920.00. "
                      "Estimated days: 1. Event date: 2026-07-28. Notice required by: 2026-08-11. Notice given: 2026-08-14.",
        }
        answer = ('**Cost impact:** The RFI is marked "Cost impact: possible," and its schedule impact is "none." [RFI-004] '
                  'A linked change event, CE-003, has an estimated cost of $7,920 and an estimated 1 day. [CE-003] '
                  'Notice was due by 2026-08-11, but it was given on 2026-08-14. [CE-003]')
        self.assertEqual(check(answer, records, computed="", context=""), [])

    def test_the_figure_is_still_checked_against_the_citation_that_follows_its_own_sentence(self) -> None:
        records = {"RFI-004": "RFI RFI-004 (answered). Cost impact: possible.", "CE-003": "Estimated cost: 7920.00. Estimated days: 1."}
        answer = 'Its schedule impact is "none." [RFI-004] A linked event costs $7,920. [RFI-004]'  # cited to the wrong record
        found = check(answer, records, computed="", context="")
        self.assertEqual(len(found), 1)
        self.assertIn("cited RFI-004", found[0])

    def test_a_figure_cannot_borrow_support_from_the_citation_of_the_next_or_previous_sentence(self) -> None:
        records = {"RFI-004": "RFI RFI-004 (answered). Cost impact: possible.", "CE-003": "Estimated cost: 7920.00. Notice given: 2026-08-14."}
        answer = "The notice was given on 2026-08-14. [CE-003] A linked event costs $7,920. [RFI-004]"
        found = check(answer, records, computed="", context="")
        self.assertEqual(len(found), 1)  # $7,920 is CE-003's, and this sentence cites RFI-004
        self.assertIn("$7,920 (cited RFI-004)", found[0])

    def test_two_citations_in_a_row_and_a_citation_inside_a_sentence_do_not_split_it(self) -> None:
        self.assertEqual(check("CE-004 adds 5 days and costs $18,500.00. [CE-004] [RFI-010]"), [])
        self.assertEqual(check("As [CE-004] shows, it adds 5 days and costs $18,500.00."), [])


class HostileAnswerSpeedTests(unittest.TestCase):
    def test_an_answer_built_to_be_slow_is_checked_in_well_under_a_second(self) -> None:
        payloads = {"unclosed brackets": "[" * 20_000, "nested brackets": "[[" * 10_000 + "]" * 5_000,
                    "a 50,000 digit amount": "$" + "9" * 50_000 + " [CE-004].", "many citations": "It adds 5 days [CE-004] " * 4_000,
                    "many sentences": "It adds 5 days. " * 20_000, "many table rows": "| a [CE-004] | 5 days |\n" * 10_000}
        for label, text in payloads.items():
            with self.subTest(payload=label):
                started = time.perf_counter()
                check(text)
                self.assertLess(time.perf_counter() - started, 1.5)


class ReviewFindingsTests(unittest.TestCase):
    """Gaps a reviewer proved in the first version."""

    def test_amounts_written_with_million_billion_or_thousand_are_understood(self) -> None:
        records = {"CE-004": "Estimated cost: 1500000.00.", "CE-001": "Estimated cost: 2000000000.00. Other: 12000.00."}
        self.assertEqual(check("It costs $1.5 million [CE-004].", records, "", ""), [])
        self.assertEqual(check("It costs $1.5 Million [CE-004].", records, "", ""), [])
        self.assertEqual(check("It costs $2 billion [CE-001].", records, "", ""), [])
        self.assertEqual(check("It costs $12 thousand [CE-001].", records, "", ""), [])
        self.assertEqual(len(check("It costs $1.6 million [CE-004].", records, "", "")), 1)

    def test_a_citation_at_the_start_of_a_line_governs_the_sentence_after_it(self) -> None:
        # CE-004 costs 18,500; 4,850 is CE-001's. The wrong figure must not escape because the citation comes first.
        self.assertEqual(len(check("- [CE-004] Cost is $4,850.")), 1)
        self.assertEqual(len(check("* [CE-004] Cost is $4,850. It also adds 5 days.")), 1)
        self.assertEqual(len(check("1. [CE-004] Cost is $4,850.")), 1)
        self.assertEqual(len(check("[CE-004] Cost is $4,850.")), 1)
        self.assertEqual(check("- [CE-004] Cost is $18,500.00 and adds 5 days."), [])
        self.assertEqual(check("- [CE-004] [RFI-010] Cost is $18,500.00."), [])

    def test_a_sentence_with_two_citations_checks_each_figure_against_its_own_citation(self) -> None:
        wrong = check("CE-004 costs $18,500 [CE-004] and CE-001 costs $18,500 [CE-001].")
        self.assertEqual(len(wrong), 1)
        self.assertIn("cited CE-001", wrong[0])
        self.assertEqual(check("CE-004 costs $18,500 [CE-004] and CE-001 costs $4,850 [CE-001]."), [])
        self.assertEqual(check("CE-004 adds 5 days [CE-004] and CE-007 adds 3 days [CE-007]."), [])

    def test_a_trailing_figure_after_the_last_citation_belongs_to_that_citation(self) -> None:
        self.assertEqual(len(check("CE-004 [CE-004] and CE-007 [CE-007] together add 8 days.")), 1)  # 8 is nobody's

    def test_more_ways_of_writing_a_quantity_are_checked(self) -> None:
        for wrong in ("It adds 30 calendar days [CE-004].", "It adds 30 working days [CE-004].", "It costs 2,000,000 dollars [CE-004].",
                      "It costs USD 2000000 [CE-004].", "It costs $2,000,000 USD [CE-004].", "It adds 20-30 days [CE-004].",
                      "It adds 20 to 30 days [CE-004]."):
            with self.subTest(wrong=wrong):
                self.assertNotEqual(check(wrong), [], wrong)
        self.assertEqual(check("It adds 5 calendar days [CE-004]."), [])
        self.assertEqual(check("It costs 18,500 dollars [CE-004]."), [])
        self.assertEqual(check("It costs USD 18500 [CE-004]."), [])

    def test_a_range_is_supported_only_when_both_ends_are(self) -> None:
        records = {"CE-004": "Estimated days: 5. Notice days: 12."}
        self.assertEqual(check("It adds 5-12 days [CE-004].", records, "", ""), [])
        self.assertEqual(len(check("It adds 5-13 days [CE-004].", records, "", "")), 1)

    def test_dates_written_out_are_compared_with_the_isos_in_the_record(self) -> None:
        for right in ("Notice was required by September 3, 2026 [CE-004].", "by Sep 3, 2026 [CE-004]", "by 3 September 2026 [CE-004]",
                      "by 3rd Sept. 2026 [CE-004]"):
            with self.subTest(right=right):
                self.assertEqual(check(right), [], right)
        for wrong in ("Notice was required by September 4, 2026 [CE-004].", "by 4 September 2026 [CE-004]"):
            with self.subTest(wrong=wrong):
                self.assertEqual(len(check(wrong)), 1, wrong)

    def test_every_month_name_and_abbreviation_maps_to_its_own_number(self) -> None:
        names = [("January", "Jan"), ("February", "Feb"), ("March", "Mar"), ("April", "Apr"), ("May", "May"), ("June", "Jun"),
                 ("July", "Jul"), ("August", "Aug"), ("September", "Sep"), ("October", "Oct"), ("November", "Nov"), ("December", "Dec")]
        for number, (full, short) in enumerate(names, start=1):
            records = {"CE-004": f"Event date: 2026-{number:02d}-15."}
            other = f"2026-{(number % 12) + 1:02d}-15"
            with self.subTest(month=full):
                self.assertEqual(check(f"It was on {full} 15, 2026 [CE-004].", records, "", ""), [])
                self.assertEqual(check(f"It was on 15 {short} 2026 [CE-004].", records, "", ""), [])
                wrong = check(f"It was on {full} 15, 2026 [CE-004].", {"CE-004": f"Event date: {other}."}, "", "")
                self.assertEqual(len(wrong), 1)

    def test_a_month_and_day_without_a_year_is_not_checked(self) -> None:
        self.assertEqual(check("Notice was due on September 4 [CE-004]."), [])  # no year, so no safe comparison


class IdsAndTablesTests(unittest.TestCase):
    def test_the_digits_of_an_id_in_a_record_never_make_a_figure_look_supported(self) -> None:
        # CE-004's text mentions RFI-011, which must not make "11 days" a figure that CE-004 states.
        records = {"CE-004": "Change event CE-004. Related RFI: RFI-011. Origin: design_change."}
        self.assertEqual(len(check("It adds 11 days [CE-004].", records, computed="")), 1)

    def test_the_date_inside_an_id_in_the_answer_is_not_a_date_claim(self) -> None:
        self.assertEqual(check("See DR-2026-08-05 for the crew [CE-004]."), [])

    def test_a_table_row_is_not_split_into_sentences_so_its_figure_stays_with_its_citation(self) -> None:
        # 5 days exists (in CE-004), so it would pass as an uncited sentence, but this row cites CE-007, which says 3.
        row = "| Sprinkler delay [CE-007]. Open. | 5 days |"
        found = check(row)
        self.assertEqual(len(found), 1)
        self.assertIn("cited CE-007", found[0])

    def test_the_line_under_a_table_header_and_an_ids_line_have_nothing_to_report(self) -> None:
        self.assertEqual(check("|---|---|\nIDS: CE-004, DR-2026-08-05"), [])


class WhatIsCheckedTests(unittest.TestCase):
    def test_digits_inside_record_ids_and_their_dates_are_not_figures(self) -> None:
        self.assertEqual(check("See RFI-010 and DR-2026-08-05 and S-004 revision 1 [RFI-010]."), [])

    def test_plain_counts_and_labels_without_a_unit_are_not_checked(self) -> None:
        self.assertEqual(check("Stair 2 on level 3 has 4 items [RFI-010]."), [])

    def test_a_number_that_is_part_of_a_longer_number_does_not_match(self) -> None:
        self.assertEqual(len(check("It costs $118,500 [CE-004].")), 1)  # not 18,500
        self.assertEqual(len(check("It adds 55 days [CE-004].")), 1)  # not 5

    def test_each_table_row_is_one_claim_with_its_own_citations(self) -> None:
        table = ("| Item | Cost | Days |\n|---|---|---|\n"
                 "| Roof steel [CE-004] | $18,500.00 | 5 days |\n"
                 "| Sprinkler delay [CE-007] | $0.00 | 3 days |\n")
        self.assertEqual(check(table), [])
        wrong = table.replace("3 days", "9 days")
        self.assertEqual(len(check(wrong)), 1)

    def test_a_long_table_row_with_the_citation_in_an_early_cell_still_belongs_to_the_row(self) -> None:
        row = "| CE-004 | Relocated rooftop unit support steel. Open. Notice due. [CE-004] | 2026-08-20 | 5 days | $18,500.00 |"
        self.assertEqual(check(row), [])
        self.assertEqual(len(check(row.replace("5 days", "7 days"))), 1)

    def test_a_sentence_ending_in_a_period_does_not_steal_the_next_sentences_citation(self) -> None:
        text = "It adds 5 days. It costs $18,500 [CE-004]."
        # "5 days" has no citation of its own, but it exists in the evidence, so it passes; the cost is cited and right.
        self.assertEqual(check(text), [])
        wrong = "It adds 8 days. It costs $18,500 [CE-004]."
        self.assertEqual(len(check(wrong)), 1)
        self.assertIn("no citation", check(wrong)[0])

    def test_an_answer_with_no_figures_has_nothing_to_check(self) -> None:
        self.assertEqual(check("The conduit was removed under CE-004 [CE-004]."), [])
        self.assertEqual(check(""), [])

    def test_the_report_is_capped_and_hostile_input_is_cheap(self) -> None:
        many = " ".join(f"It costs ${n}00,000 [CE-004]." for n in range(1, 40))
        self.assertLessEqual(len(check(many)), 10)
        self.assertEqual(len(check("It costs $" + "9" * 5000 + " [CE-004].")), 1)  # no hang, no crash
        check("[" * 10000 + "]" * 10000)


if __name__ == "__main__":
    unittest.main()
