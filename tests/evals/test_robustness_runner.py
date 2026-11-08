import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from evals.run_robustness import EXPANDED_DATASET_SHA256, main
from rag.answering.answer import Answer
from rag.answering.evidence import ToolEvidence
from tests.support.db_support import skip_unless

SAMPLE = Path("data/records/samples/library-renovation")


@skip_unless(SAMPLE.exists(), "clone the pinned records repo into data/records first")
class RobustnessRunnerTests(unittest.TestCase):
    def test_expanded_suite_is_frozen_and_has_traceable_records(self) -> None:
        content = Path("evals/data/robustness_v2.json").read_bytes()
        self.assertEqual(hashlib.sha256(content).hexdigest(), EXPANDED_DATASET_SHA256)
        suite = json.loads(content)
        self.assertEqual(len(suite["cases"]), 30)
        self.assertEqual(len({case["id"] for case in suite["cases"]}), 30)
        known: set[str] = set()
        for filename, column in [("rfis.csv", "rfi_number"), ("submittals.csv", "submittal_number"),
                                 ("change_events.csv", "event_number"), ("punch_items.csv", "item_number")]:
            with (SAMPLE / filename).open(newline="", encoding="utf-8") as stream:
                known.update(row[column] for row in csv.DictReader(stream))
        with (SAMPLE / "daily_reports.csv").open(newline="", encoding="utf-8") as stream:
            known.update("DR-" + row["report_date"] for row in csv.DictReader(stream))
        for case in suite["cases"]:
            with self.subTest(case=case["id"]):
                self.assertTrue(case["query"] and case["reference_answer"] and case["source_fields"]
                                and case["failure_condition"])
                self.assertTrue(set(case["source_ids"]) <= known or case["id"] == "no-such-change")
        self.assertNotIn("CE-099", known)

    def test_runner_captures_source_checked_references_without_claiming_a_score(self) -> None:
        evidence = ToolEvidence("get_record", "v", "[RFI-009] status open", False)
        answer = Answer("A model answer.", ["RFI-009"], [], evidence=(evidence,))
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch("evals.run_robustness.load_env"), \
                mock.patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}), \
                mock.patch("evals.run_robustness.make_anthropic_chat", return_value=lambda *args: {}), \
                mock.patch("evals.run_robustness.run_agent", return_value=answer):
            output = Path(directory) / "review.jsonl"
            self.assertEqual(main_args(["--store", "memory", "--output", str(output)]), 0)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[0]["context"], evidence.text)
        self.assertTrue(rows[0]["reference_answer"])
        self.assertIn("trace", rows[0])
        self.assertIn("withheld_because", rows[0])
        self.assertNotIn("passed", rows[0])

    def test_trace_capture_can_be_explicitly_disabled(self) -> None:
        answer = Answer("Example.", [], [])
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch("evals.run_robustness.load_env"), \
                mock.patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}), \
                mock.patch("evals.run_robustness.make_anthropic_chat", return_value=lambda *args: {}), \
                mock.patch("evals.run_robustness.run_agent", return_value=answer):
            output = Path(directory) / "review.jsonl"
            self.assertEqual(main_args(["--store", "memory", "--no-capture-trace", "--output", str(output)]), 0)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        self.assertNotIn("trace", rows[0])

    def test_expanded_suite_can_be_captured_without_scoring(self) -> None:
        answer = Answer("Example [RFI-009].", ["RFI-009"], [])
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch("evals.run_robustness.load_env"), \
                mock.patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}), \
                mock.patch("evals.run_robustness.make_anthropic_chat", return_value=lambda *args: {}), \
                mock.patch("evals.run_robustness.run_agent", return_value=answer):
            output = Path(directory) / "review.jsonl"
            self.assertEqual(main_args(["--store", "memory", "--suite", "v2", "--output", str(output)]), 0)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 30)
        self.assertEqual(rows[0]["suite_sha256"], EXPANDED_DATASET_SHA256)
        self.assertTrue(rows[0]["failure_condition"])
        self.assertNotIn("passed", rows[0])

    def test_changed_dataset_fails_before_calling_the_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch("evals.run_robustness.DATASET_SHA256", "changed"), \
                mock.patch("evals.run_robustness.make_anthropic_chat") as chat:
            with self.assertRaises(SystemExit) as caught:
                main_args(["--store", "memory", "--output", str(Path(directory) / "review.jsonl")])
        self.assertEqual(caught.exception.code, 2)
        chat.assert_not_called()


def main_args(args: list[str]) -> int:
    with mock.patch("sys.argv", ["evals.run_robustness", *args]):
        return main()
