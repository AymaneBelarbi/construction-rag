import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from evals.rescore import main, rescore

GOOD = {"id": "lookup-rfi", "repeat": 0, "passed": True, "error": None, "ungrounded": [], "truncated": False,
        "text": "RFI-004 is answered, with a possible cost impact [RFI-004]."}
WRONG = {**GOOD, "repeat": 1, "passed": True, "text": "It is closed."}  # was recorded as passing; today's scorer disagrees
FAILED_THEN_FIXED = {**GOOD, "repeat": 2, "passed": False}  # recorded as a failure; today's scorer accepts the answer
ERRORED = {"id": "lookup-rfi", "repeat": 3, "passed": False, "error": "TransientError: down", "text": ""}


class RescoreTests(unittest.TestCase):
    def test_it_scores_every_valid_row_again_and_reports_the_verdicts_that_changed(self) -> None:
        result = rescore([GOOD, WRONG, FAILED_THEN_FIXED, ERRORED], "golden")
        self.assertEqual((result.total, result.passed), (3, 2))  # the errored run was never an answer
        self.assertEqual(result.changed, [("lookup-rfi", 1, True, False), ("lookup-rfi", 2, False, True)])

    def test_the_command_prints_the_counts_and_exits_1_when_an_answer_fails(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "claude-x-golden-1.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in (GOOD, WRONG)), encoding="utf-8")
            out = io.StringIO()
            with redirect_stdout(out):
                code = main([str(path)])
        self.assertEqual(code, 1)
        self.assertIn("1/2", out.getvalue())
        self.assertIn("changed", out.getvalue())

    def test_the_suite_is_taken_from_the_file_name(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "claude-x-nothing-1.jsonl"
            path.write_text(json.dumps(GOOD), encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit):
                    main([str(path)])


class CommittedBaselinesTests(unittest.TestCase):
    """Recorded baselines must retain their original scores."""

    def test_every_committed_set_of_answers_scores_as_recorded_by_the_current_scorer(self) -> None:
        files = sorted(Path("evals/baselines").glob("*.jsonl"))
        self.assertGreaterEqual(len(files), 4)  # golden and injection for both models
        for path in files:
            with self.subTest(file=path.name):
                with redirect_stdout(io.StringIO()):
                    code = main([str(path)])
                self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
