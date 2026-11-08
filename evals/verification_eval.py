"""Paid verifier challenge set. Run explicitly: python -m evals.verification_eval --model MODEL."""
import argparse
import json
import os
import time
from pathlib import Path

from rag.answering.verification import verify_answer
from rag.infra.telemetry import RunStats, metered_chat
from rag.providers.llm import load_env, make_anthropic_chat


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure verifier misses and false rejections on held-out examples.")
    parser.add_argument("--model", default="claude-sonnet-5-5")
    args = parser.parse_args()
    load_env()
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        parser.error("ANTHROPIC_API_KEY is required")
    cases = json.loads((Path(__file__).parent / "data" / "verification_holdout.json").read_text(encoding="utf-8"))
    stats = RunStats(model=args.model)
    chat = metered_chat(make_anthropic_chat(key, model=args.model), stats)
    false_accepts = false_rejections = unavailable = 0
    timings = []
    started = time.monotonic()
    for case in cases:
        case_started = time.monotonic()
        verdict = verify_answer(case["answer"], case["records"], case["computed"], chat)
        timings.append(time.monotonic() - case_started)
        expected = case["expected"]
        false_accepts += verdict == "supported" and expected == "rejected"
        false_rejections += verdict == "rejected" and expected == "supported"
        unavailable += verdict == "unavailable"
        print(f"{case['id']}: expected={expected} actual={verdict}")
    negative = sum(case["expected"] == "rejected" for case in cases)
    positive = len(cases) - negative
    print(json.dumps({"cases": len(cases), "unsupported_cases": negative, "supported_cases": positive,
                      "false_accepts": false_accepts, "false_accept_rate": false_accepts / negative if negative else None,
                      "false_rejections": false_rejections, "false_rejection_rate": false_rejections / positive if positive else None,
                      "unavailable": unavailable, "seconds": round(time.monotonic() - started, 2),
                      "mean_seconds_per_case": round(sum(timings) / len(timings), 2) if timings else None,
                      "estimated_cost_usd": stats.cost_usd()}, indent=2))
    return 1 if false_accepts or false_rejections or unavailable else 0


if __name__ == "__main__":
    raise SystemExit(main())
