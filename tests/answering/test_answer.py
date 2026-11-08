import unittest

from rag.answering.answer import (
    Answer,
    answer_from_hits,
    extract_text,
    find_bracketed_ids,
    find_cited_ids,
    format_context,
    make_answer,
    refusal_answer,
    uncited_passages,
    ungrounded_ids,
)
from rag.records.chunker import Chunk
from rag.retrieval.retrieve import Hit
from tests.support.fakes import ScriptedChat, text_response

CHUNK = Chunk("rfi", "RFI-003", "RFI RFI-003 (closed). Subject: Abandoned conduit.", {"status": "closed"})
HIT = Hit(CHUNK, 0.9)
# This record only MENTIONS RFI-003 in its text; RFI-003 itself is not among the evidence.
MENTIONS_ONLY = Hit(
    Chunk("change_event", "CE-001", "Change event CE-001 (approved). Related RFI: RFI-003.", {}), 0.9
)


class AnswerHelperTests(unittest.TestCase):
    def test_find_cited_ids(self) -> None:
        found = find_cited_ids("See [RFI-003], CE-001 and S-001; also DR-2026-08-03.")
        self.assertEqual(found, ["RFI-003", "CE-001", "S-001", "DR-2026-08-03"])

    def test_ungrounded_ids_are_those_not_in_the_allowed_set(self) -> None:
        allowed = {"RFI-003", "CE-001"}
        self.assertEqual(ungrounded_ids("Cited RFI-003 and RFI-099", allowed), ["RFI-099"])

    def test_ungrounded_check_ignores_case(self) -> None:
        self.assertEqual(ungrounded_ids("rfi-003", {"RFI-003"}), [])

    def test_unfamiliar_bracketed_id_is_checked_even_with_a_valid_citation(self) -> None:
        answer = make_answer("Approved [RFI-003] and [ABC-123].", ["RFI-003"])
        self.assertEqual(answer.ungrounded, ["ABC-123"])

    def test_unfamiliar_returned_id_can_be_cited(self) -> None:
        answer = make_answer("Approved [ABC-123].", ["ABC-123"])
        self.assertEqual(answer.ungrounded, [])
        self.assertFalse(answer.uncited)

    def test_alphabetic_record_ids_are_checked(self) -> None:
        self.assertEqual(make_answer("Approved [RECORD].", ["RECORD"]).ungrounded, [])
        self.assertEqual(make_answer("Approved [OTHER].", ["RECORD"]).ungrounded, ["OTHER"])

    def test_malformed_citation_is_rejected(self) -> None:
        for text in ("Approved [RFI-003", "Approved [RFI 003]", "Approved [RFI-003,]."):
            with self.subTest(text=text):
                self.assertIn("malformed citation", make_answer(text, ["RFI-003"]).ungrounded)

    def test_plain_mention_of_a_not_found_id_is_allowed(self) -> None:
        text = "There is no record of RFI-099 in the logs."
        self.assertEqual(ungrounded_ids(text, {"RFI-003"}, {"RFI-099"}), [])

    def test_bracketed_citation_of_a_not_found_id_is_still_flagged(self) -> None:
        text = "It was approved [RFI-099]."
        self.assertEqual(ungrounded_ids(text, {"RFI-003"}, {"RFI-099"}), ["RFI-099"])

    def test_plain_mention_of_an_id_nobody_looked_up_is_still_flagged(self) -> None:
        self.assertEqual(ungrounded_ids("RFI-099 was closed.", {"RFI-003"}, {"RFI-050"}), ["RFI-099"])

    def test_a_drawing_sheet_reference_from_the_evidence_may_be_mentioned_in_plain_text(self) -> None:
        # RFI-009 has "References: S-109" (a structural drawing sheet). S-109 is not a submittal.
        evidence = "[RFI-009] RFI RFI-009 (open). References: S-109."
        text = "RFI-009 refers to sheet S-109 [RFI-009]."
        self.assertEqual(ungrounded_ids(text, {"RFI-009"}, evidence_text=evidence), [])

    def test_a_sheet_reference_is_still_flagged_when_bracketed_as_if_it_were_a_record(self) -> None:
        evidence = "[RFI-009] References: S-109."
        self.assertEqual(ungrounded_ids("Approved [S-109].", {"RFI-009"}, evidence_text=evidence), ["S-109"])

    def test_an_s_id_that_is_nowhere_in_the_evidence_is_still_flagged(self) -> None:
        evidence = "[RFI-009] References: S-109."
        self.assertEqual(ungrounded_ids("See S-777.", {"RFI-009"}, evidence_text=evidence), ["S-777"])

    def test_other_prefixes_get_no_sheet_reference_exemption(self) -> None:
        # Only S- collides with drawing numbering. A plain claim about an RFI seen only by name stays flagged.
        evidence = "[CE-001] Change event CE-001. Related RFI: RFI-003."
        self.assertEqual(ungrounded_ids("RFI-003 was closed.", {"CE-001"}, evidence_text=evidence), ["RFI-003"])

    def test_find_bracketed_ids_only_returns_ids_inside_brackets(self) -> None:
        text = "See [RFI-010, RFI-011] and [S-005] but not RFI-003."
        self.assertEqual(find_bracketed_ids(text), ["RFI-010", "RFI-011", "S-005"])

    def test_change_order_numbers_are_not_checked_because_they_are_not_records(self) -> None:
        self.assertEqual(ungrounded_ids("Approved as CO-001.", set()), [])

    def test_format_context_lists_each_record_with_its_id(self) -> None:
        context = format_context([HIT])
        self.assertIn("[RFI-003]", context)
        self.assertIn("Abandoned conduit", context)

    def test_extract_text_joins_text_blocks_only(self) -> None:
        response = {"content": [{"type": "text", "text": "A"}, {"type": "tool_use", "id": "x"}, {"type": "text", "text": "B"}]}
        self.assertEqual(extract_text(response), "A\nB")

    def test_answer_flags_default_to_false(self) -> None:
        answer = Answer("text", [], [])
        self.assertFalse(answer.truncated)
        self.assertFalse(answer.uncited)
        self.assertFalse(answer.refused)

    def test_a_refusal_is_a_fixed_sentence_that_names_what_was_asked_for(self) -> None:
        plain = refusal_answer([], tool_calls=2)
        self.assertTrue(plain.refused)
        self.assertEqual(plain.tool_calls, 2)
        self.assertIn("could not find", plain.text)
        named = refusal_answer(["RFI-099", "CE-009"])
        self.assertIn("CE-009, RFI-099", named.text)  # sorted, so the sentence is stable
        self.assertEqual(named.ungrounded, [])
        self.assertFalse(named.uncited)

    def test_the_three_situations_get_three_different_honest_messages(self) -> None:
        failed = refusal_answer([], tool_calls=1, search_failed=True)
        nothing_looked_up = refusal_answer([], tool_calls=0)
        nothing_found = refusal_answer([], tool_calls=1)
        self.assertTrue(failed.search_failed)
        self.assertIn("a tool failed", failed.text)
        self.assertFalse(nothing_looked_up.search_failed)
        self.assertIn("did not look anything up", nothing_looked_up.text)
        self.assertIn("could not find", nothing_found.text)
        self.assertEqual(len({failed.text, nothing_looked_up.text, nothing_found.text}), 3)

    def test_ids_may_be_given_as_a_one_shot_iterator(self) -> None:
        answer = refusal_answer((i for i in ["RFI-099"]), tool_calls=1)
        self.assertIn("RFI-099", answer.text)


class UnsupportedFieldTests(unittest.TestCase):
    def test_an_answer_starts_with_no_unsupported_claims(self) -> None:
        self.assertEqual(make_answer("RFI-003 is closed [RFI-003].", ["RFI-003"]).unsupported, [])
        self.assertEqual(refusal_answer(["RFI-099"], tool_calls=1).unsupported, [])


class AnswerFromHitsTests(unittest.TestCase):
    def test_default_verifier_withholds_a_semantically_contradicted_answer(self) -> None:
        chat = ScriptedChat([text_response("RFI-003 is open [RFI-003].")])

        def reject(system: str, messages: list[dict], tools: list[dict]) -> dict:
            return text_response('{"verdicts":[{"id":0,"status":"contradicted"}]}')

        answer = answer_from_hits("What is its status?", [HIT], chat, verifier_chat=reject)
        self.assertTrue(answer.verification_failed)
        self.assertEqual(answer.verification_status, "rejected")

    def test_unsupported_figure_is_withheld_from_direct_caller(self) -> None:
        chat = ScriptedChat([text_response("It cost $20,000 [RFI-003].")])
        answer = answer_from_hits("What did it cost?", [HIT], chat)
        self.assertTrue(answer.verification_failed)
        self.assertNotIn("$20,000", answer.text)

    def test_grounded_answer_has_no_warnings(self) -> None:
        chat = ScriptedChat([text_response("The conduit RFI is [RFI-003].")])
        answer = answer_from_hits("What about the conduit?", [HIT], chat)
        self.assertEqual(answer.cited_ids, ["RFI-003"])
        self.assertEqual(answer.ungrounded, [])
        self.assertFalse(answer.uncited)
        self.assertEqual(answer.evidence[0].records[0].text, CHUNK.text)
        self.assertIn("RFI-003", chat.calls[0]["messages"][0]["content"])
        self.assertEqual(chat.calls[0]["tools"], [])

    def test_invented_id_is_flagged(self) -> None:
        chat = ScriptedChat([text_response("See RFI-003 and RFI-099.")])
        answer = answer_from_hits("q", [HIT], chat)
        self.assertEqual(answer.ungrounded, ["RFI-099"])

    def test_id_that_is_only_mentioned_inside_another_record_is_flagged(self) -> None:
        chat = ScriptedChat([text_response("RFI-003 was closed with cost impact.")])
        answer = answer_from_hits("q", [MENTIONS_ONLY], chat)
        self.assertEqual(answer.ungrounded, ["RFI-003"])

    def test_answer_with_records_but_no_citation_is_marked_uncited(self) -> None:
        chat = ScriptedChat([text_response("RFI-003 was closed last month.")])
        answer = answer_from_hits("q", [HIT], chat)
        self.assertTrue(answer.uncited)
        self.assertTrue(answer.verification_failed)
        self.assertNotIn("closed last month", answer.text)

    def test_uncited_extra_sentence_is_withheld(self) -> None:
        # On its own line it continues nothing, so it needs its own citation.
        chat = ScriptedChat([text_response("The RFI is open [RFI-003].\nThe attachment confirms it.")])
        answer = answer_from_hits("q", [HIT], chat)
        self.assertTrue(answer.verification_failed)
        self.assertTrue(answer.citation_gaps)

    def test_a_withheld_answer_says_which_checks_failed(self) -> None:
        answer = Answer("text", [], ["RFI-099"], unsupported=["$5 (cited CE-1)"], citation_gaps=["a", "b"],
                        verification_status="rejected")
        answer.withhold_unverified(enforce_verification=True)
        self.assertEqual(answer.withheld_because, ["cites records no tool returned (1)", "claims without a citation (2)",
                                                   "figures the cited records do not state (1)", "verifier rejected"])

    def test_shadow_verification_is_not_a_reason_to_withhold(self) -> None:
        answer = Answer("RFI-1 is open [RFI-1].", ["RFI-1"], [], verification_status="rejected")
        answer.withhold_unverified(enforce_verification=False)
        self.assertEqual(answer.withheld_because, [])
        self.assertFalse(answer.verification_failed)

    def test_no_hits_skips_the_llm(self) -> None:
        chat = ScriptedChat([])
        answer = answer_from_hits("q", [], chat)
        self.assertIn("No matching records", answer.text)
        self.assertEqual(chat.calls, [])
        self.assertFalse(answer.uncited)


class UncitedPassagesTests(unittest.TestCase):
    """Section headings carry no facts, so they need no citation. Found in the 2026-11-08 production eval: a fully cited
    answer was withheld only because of the headings **Open issue**, **Resolved** and **Field activity**."""

    def test_markdown_section_headings_need_no_citation(self) -> None:
        for heading in ("**Open issue**", "## Open issue", "### Items with no ball in court recorded", "- **Resolved**",
                        "__Field activity__"):
            with self.subTest(heading=heading):
                self.assertEqual(uncited_passages(f"{heading}\n- RFI-010 is open [RFI-010]."), [])

    def test_a_heading_may_name_the_record_it_is_about(self) -> None:
        # Seen in the second production run: a record ID labels the section, it is not a figure.
        for heading in ("**Cost (CE-004)**", "**Structural support (RFI-011)**", "## S-004 flooring samples",
                        "**RFIs and submittals with the ball in Sample Electric's court**"):
            with self.subTest(heading=heading):
                self.assertEqual(uncited_passages(heading), [])

    def test_a_bold_sentence_is_still_a_claim(self) -> None:
        self.assertEqual(uncited_passages("**RFI-010 is overdue.**"), ["**RFI-010 is overdue.**"])

    def test_a_heading_with_a_figure_is_still_a_claim(self) -> None:
        # A number in a heading states a fact (an amount, a count, a date) and must be cited like any other.
        for heading in ("**Approved total $17,490**", "## 5 overdue RFIs", "**Peak crew on 2026-08-05**",
                        "**CE-004 adds 5 days**"):
            with self.subTest(heading=heading):
                self.assertEqual(uncited_passages(heading), [heading])

    def test_a_sentence_continuing_a_cited_one_on_the_same_line_needs_no_citation(self) -> None:
        # Found in the 2026-11-08 production eval: "It came from an unforeseen condition." follows "[CE-005]".
        self.assertEqual(uncited_passages("- CE-005 is priced [CE-005]. It came from an unforeseen condition."), [])

    def test_a_continuing_sentence_with_a_figure_still_needs_its_own_citation(self) -> None:
        # Figures in a sentence without a citation are checked against all evidence at once (claims.py), which is weak.
        self.assertEqual(uncited_passages("RFI-010 is open [RFI-010]. It was needed by 2026-08-25."),
                         ["It was needed by 2026-08-25."])
        self.assertEqual(uncited_passages("CE-004 is open [CE-004]. It adds five days, about 5 days."),
                         ["It adds five days, about 5 days."])
        # A count written as a word is still a figure.
        self.assertEqual(uncited_passages("Four change events trace back to an RFI. The list is complete [CALC-1]."),
                         ["Four change events trace back to an RFI."])
        # ...but "one" and "both" used as pronouns are not counts.
        self.assertEqual(uncited_passages("CE-006 was rejected [CE-006]. If that's the one you meant, I can't say why."), [])
        self.assertEqual(uncited_passages("S-004 and S-010 are with the Architect [S-004] [S-010]. Both are in review."), [])

    def test_an_uncited_sentence_on_its_own_line_still_needs_a_citation(self) -> None:
        self.assertEqual(uncited_passages("RFI-010 is open [RFI-010].\nThe architect is slow to respond."),
                         ["The architect is slow to respond."])

    def test_saying_a_record_does_not_contain_something_requires_a_citation(self) -> None:
        for sentence in ("The logs don't show the asbestos change event as rejected.", "No answered date is recorded.",
                         "I couldn't find an original contract sum in the project records I searched.",
                         "The record gives no reason for the rejection.", "The log doesn't say who signed off on the fix."):
            with self.subTest(sentence=sentence):
                self.assertEqual(uncited_passages(sentence), [sentence])

    def test_an_absence_statement_with_a_figure_or_a_plain_negation_still_needs_a_citation(self) -> None:
        for sentence in ("The logs don't show the $5,000 payment.", "No notice was recorded by 2026-09-03.",
                         "RFI-010 is not answered.", "The submittal was not approved."):
            with self.subTest(sentence=sentence):
                self.assertEqual(uncited_passages(sentence), [sentence])

    def test_an_uncited_missing_information_claim_is_withheld(self) -> None:
        answer = make_answer("I couldn't find an original contract sum in the project records I searched.", ["CE-001"])
        self.assertTrue(answer.uncited)
        self.assertTrue(answer.citation_gaps)
        answer.withhold_unverified(enforce_verification=False)
        self.assertTrue(answer.verification_failed)

    def test_uncited_missing_field_on_a_partly_cited_line_is_flagged(self) -> None:
        text = "CE-003 is submitted [CE-003]. The notice date is not recorded."
        self.assertEqual(uncited_passages(text), ["The notice date is not recorded."])

    def test_an_answer_with_an_uncited_claim_is_still_uncited(self) -> None:
        self.assertTrue(make_answer("RFI-003 was closed last month.", ["RFI-003"]).uncited)

    def test_a_short_lead_in_ending_in_a_colon_needs_no_citation_unless_it_states_a_figure(self) -> None:
        # A heading that states a count is a claim.
        self.assertEqual(uncited_passages("Open items:\n- RFI-010 is open [RFI-010]."), [])
        for lead_in in ("The Architect holds four open RFIs:", "Overdue since 2026-09-01:"):
            with self.subTest(lead_in=lead_in):
                self.assertEqual(uncited_passages(f"{lead_in}\n- RFI-010 is open [RFI-010]."), [lead_in])

    def test_a_long_bold_line_is_not_a_heading(self) -> None:
        line = "**The architect has not answered the panel mounting question for the reading room ceiling yet, it seems**"
        self.assertEqual(uncited_passages(line), [line])


if __name__ == "__main__":
    unittest.main()
