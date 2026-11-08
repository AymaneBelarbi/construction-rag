import json
import unittest

from rag.answering.verification import VERIFIER_PROMPT, verify_answer
from rag.infra.resilience import TransientError
from tests.support.fakes import text_response


def _payload(messages: list[dict]) -> dict:
    """The JSON the verifier receives, taken out of its untrusted-data wrapper."""
    content = messages[0]["content"]
    return json.loads(content[content.index("{"):content.rindex("}") + 1])


class VerifierTests(unittest.TestCase):
    def test_only_cited_record_is_given_to_verifier(self) -> None:
        def verifier(system: str, messages: list[dict], tools: list[dict]) -> dict:
            content = messages[0]["content"]
            self.assertIn("open", content)
            self.assertNotIn("SECRET OTHER RECORD", content)
            return text_response(json.dumps({"verdicts": [{"id": 0, "status": "contradicted"}]}))

        self.assertEqual(verify_answer("Closed [RFI-1].", {"RFI-1": "Status: open", "RFI-2": "SECRET OTHER RECORD"}, "", verifier),
                         "rejected")

    def test_missing_duplicate_or_unknown_verdicts_fail_closed(self) -> None:
        responses = ["{}", '{"verdicts": []}', '{"verdicts": [{"id": 1, "status": "supported"}]}',
                     '{"verdicts": [{"id": 0, "status": "maybe"}]}']
        for response in responses:
            with self.subTest(response=response):
                def reply(system: str, messages: list[dict], tools: list[dict], value: str = response) -> dict:
                    return text_response(value)

                self.assertEqual(verify_answer("Open [RFI-1].", {"RFI-1": "open"}, "", reply), "unavailable")

    def test_network_failure_fails_closed(self) -> None:
        def down(system: str, messages: list[dict], tools: list[dict]) -> dict:
            raise TransientError("offline")

        self.assertEqual(verify_answer("Open [RFI-1].", {"RFI-1": "open"}, "", down), "unavailable")

    def test_every_passage_requires_exactly_one_verdict(self) -> None:
        text = "Open [RFI-1].\nClosed [RFI-2]."
        records = {"RFI-1": "Open", "RFI-2": "Closed"}

        def duplicate(system: str, messages: list[dict], tools: list[dict]) -> dict:
            return text_response('{"verdicts":[{"id":0,"status":"supported"},{"id":0,"status":"supported"}]}')

        self.assertEqual(verify_answer(text, records, "", duplicate), "unavailable")

    def test_report_line_for_another_id_is_not_evidence(self) -> None:
        def inspect(system: str, messages: list[dict], tools: list[dict]) -> dict:
            self.assertNotIn("RFI-10", messages[0]["content"])
            return text_response('{"verdicts":[{"id":0,"status":"not_established"}]}')

        self.assertEqual(verify_answer("Closed [RFI-1].", {}, "RFI-10 was closed", inspect), "rejected")

    def test_a_passage_without_citation_is_checked_against_the_records_the_answer_cites(self) -> None:
        # A continuation can use the evidence cited by the answer.
        def inspect(system: str, messages: list[dict], tools: list[dict]) -> dict:
            payload = _payload(messages)
            self.assertEqual(payload["passages"][1]["evidence_ids"], ["CE-005"])
            self.assertIn("Origin: unforeseen_condition", payload["records"]["CE-005"])
            return text_response('{"verdicts":[{"id":0,"status":"supported"},{"id":1,"status":"supported"}]}')

        text = "CE-005 is priced [CE-005]. It came from an unforeseen condition."
        self.assertEqual(verify_answer(text, {"CE-005": "Status: priced. Origin: unforeseen_condition"}, "", inspect), "supported")

    def test_a_cited_passage_is_checked_only_against_its_own_citations(self) -> None:
        def inspect(system: str, messages: list[dict], tools: list[dict]) -> dict:
            passages = _payload(messages)["passages"]
            self.assertEqual([p["evidence_ids"] for p in passages], [["RFI-1"], ["RFI-2"]])
            return text_response('{"verdicts":[{"id":0,"status":"supported"},{"id":1,"status":"supported"}]}')

        self.assertEqual(verify_answer("Open [RFI-1].\nClosed [RFI-2].", {"RFI-1": "Open", "RFI-2": "Closed"}, "", inspect),
                         "supported")

    def test_a_reply_in_a_code_fence_is_read(self) -> None:
        def fenced(system: str, messages: list[dict], tools: list[dict]) -> dict:
            return text_response('```json\n{"verdicts": [{"id": 0, "status": "supported"}]}\n```')

        self.assertEqual(verify_answer("Open [RFI-1].", {"RFI-1": "open"}, "", fenced), "supported")

    def test_text_around_a_code_fence_still_fails_closed(self) -> None:
        def chatty(system: str, messages: list[dict], tools: list[dict]) -> dict:
            return text_response('Here you go:\n```json\n{"verdicts": [{"id": 0, "status": "supported"}]}\n```')

        self.assertEqual(verify_answer("Open [RFI-1].", {"RFI-1": "open"}, "", chatty), "unavailable")

    def test_the_prompt_accepts_a_correct_statement_that_something_is_not_recorded(self) -> None:
        # The live check is the production eval; this pins the rule so it is not dropped by accident.
        self.assertIn("does not record", VERIFIER_PROMPT)

    def test_uncited_missing_field_is_rejected_before_calling_the_verifier(self) -> None:
        def should_not_run(system: str, messages: list[dict], tools: list[dict]) -> dict:
            raise AssertionError("an uncited missing-field claim has no source to check")

        self.assertEqual(verify_answer("The records do not show a notice date.",
                                       {"CE-003": "Notice given: 2026-08-14."}, "", should_not_run), "rejected")
        self.assertEqual(verify_answer("The records lack a notice date.",
                                       {"CE-003": "Notice given: 2026-08-14."}, "", should_not_run), "rejected")

    def test_cited_missing_field_is_checked_against_that_record(self) -> None:
        def inspect(system: str, messages: list[dict], tools: list[dict]) -> dict:
            payload = _payload(messages)
            self.assertEqual(payload["passages"][0]["evidence_ids"], ["CE-003"])
            self.assertIn("2026-08-14", payload["records"]["CE-003"])
            return text_response('{"verdicts":[{"id":0,"status":"contradicted"}]}')

        self.assertEqual(verify_answer("The notice date is not recorded [CE-003].",
                                       {"CE-003": "Notice given: 2026-08-14."}, "", inspect), "rejected")

    def test_calculation_must_be_cited_to_reach_verifier(self) -> None:
        def inspect(system: str, messages: list[dict], tools: list[dict]) -> dict:
            self.assertNotIn("secret unrelated calculation", messages[0]["content"])
            return text_response('{"verdicts":[{"id":0,"status":"not_established"}]}')

        self.assertEqual(verify_answer("Cost is $99 [CE-1].", {"CE-1": "Cost: 1"},
                                       "secret unrelated calculation: 99", inspect), "rejected")
