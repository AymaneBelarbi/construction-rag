import unittest

from evals.export_azure import export_rows


class AzureExportTests(unittest.TestCase):
    def test_completed_trace_maps_to_foundry_input_without_inventing_ground_truth(self) -> None:
        rows = [{"id": "q1", "query": "What is open?", "text": "RFI-1 is open.", "context": "RFI-1: open",
                 "dataset_version": "v1", "error": None}]
        self.assertEqual(export_rows(rows), [{"case_id": "q1", "query": "What is open?", "response": "RFI-1 is open.",
                                              "context": "RFI-1: open", "dataset_version": "v1"}])

    def test_missing_capture_cannot_silently_export_an_empty_context(self) -> None:
        with self.assertRaisesRegex(ValueError, "--capture-evidence"):
            export_rows([{"id": "q1", "text": "answer", "error": None}])

    def test_failed_runs_do_not_become_evaluation_examples(self) -> None:
        with self.assertRaisesRegex(ValueError, "No completed"):
            export_rows([{"id": "q1", "error": "provider down"}])

    def test_source_checked_reference_and_date_are_exported(self) -> None:
        rows = [{"id": "q1", "query": "When?", "text": "Unknown", "context": "Date not recorded",
                 "reference_answer": "Date not recorded", "as_of": "2026-09-20", "error": None}]
        exported = export_rows(rows)[0]
        self.assertEqual(exported["ground_truth"], "Date not recorded")
        self.assertEqual(exported["as_of"], "2026-09-20")
