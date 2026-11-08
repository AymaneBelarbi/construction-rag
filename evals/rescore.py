"""Score saved live answers again with today's scorer. Free, offline, no API.

    python -m evals.rescore evals/baselines/claude-sonnet-5-5-golden-20261107-231803.jsonl

The file is one line per run, written by evals/run_eval.py. Rescoring shows whether a change to a scoring pattern
changes the recorded answers' verdicts. It cannot redo the claim check: that needs the evidence the tools returned, which a
saved run does not keep; the live run's own `unsupported` field is the record of it.
"""
import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evals.questions import SUITES
from evals.scoring import score_answer
from rag.answering.answer import Answer, find_bracketed_ids


@dataclass(frozen=True)
class Rescored:
    total: int
    passed: int
    changed: list[tuple[str, int, bool, bool]]  # (question id, repeat, verdict when recorded, verdict now)


def rescore(rows: list[dict[str, Any]], suite: str) -> Rescored:
    questions = {question.id: question for question in SUITES[suite].questions}
    total = passed = 0
    changed: list[tuple[str, int, bool, bool]] = []
    for row in rows:
        if row.get("error"):
            continue  # the run never produced an answer
        answer = Answer(row["text"], find_bracketed_ids(row["text"]), row.get("ungrounded", []), truncated=row.get("truncated", False))
        now = score_answer(questions[row["id"]], answer).passed
        total += 1
        passed += now
        if now != row["passed"]:
            changed.append((row["id"], row["repeat"], row["passed"], now))
    return Rescored(total, passed, changed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score saved live answers again with the current scorer.")
    parser.add_argument("file", help="a .jsonl file written by evals/run_eval.py")
    args = parser.parse_args(argv)
    path = Path(args.file)
    suite = next((name for name in SUITES if f"-{name}-" in path.name), None)
    if suite is None:
        parser.error(f"cannot tell the suite from the file name {path.name!r} (expected -golden- or -injection-)")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = rescore(rows, suite)
    print(f"{path.name}: {result.passed}/{result.total} answers pass the current {suite} scorer")
    for question_id, repeat, was, now in result.changed:
        print(f"  changed: {question_id} run {repeat}: recorded {'pass' if was else 'fail'}, now {'pass' if now else 'fail'}")
    return 0 if result.passed == result.total else 1


if __name__ == "__main__":
    sys.exit(main())
