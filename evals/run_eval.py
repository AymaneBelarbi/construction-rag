"""Live evaluation: ask the golden questions, score the answers, enforce quality/cost/latency gates.

This calls the real API and costs money (about $0.02 per question on Sonnet 5.5, $0.001 on Haiku 5.5),
so it is NOT part of the unit tests. Run it from the project root when you change the prompt, the
tools, the retrieval or the model:

    python -m evals.run_eval --model claude-haiku-5-5
    python -m evals.run_eval --suite injection --repeats 3
    python -m evals.run_eval --only overdue-rfi,multihop

Exit codes:
    0  every gate passed
    1  a quality gate failed (the system got worse)
    2  bad usage or missing configuration
    3  the run itself was invalid (API errors); the quality numbers cannot be trusted
Quality is judged only on runs that completed, so an outage cannot look like a hallucination.

Not identical to production on purpose: the prompt names today's date (the CLI does the same through
--as-of), and list questions ask for a final `IDS:` line so precision and recall can be scored exactly.
"""
import argparse
import datetime
import json
import math
import os
import re
import subprocess
import sys
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from evals.questions import AS_OF, INJECTION_PROJECT, SAMPLE_PROJECT, SUITES, Question, build_prompt
from evals.scoring import score_answer
from evals.tempdb import temporary_database_schema
from rag.agent import Tools, make_cpr_runner, run_agent
from rag.infra.telemetry import RunStats, metered_chat
from rag.providers.embed import HASHING_ID, Embedder, hashing_embed
from rag.providers.llm import Chat, embedder_from_env, load_env, make_anthropic_chat
from rag.records.snapshot import capture, materialize, version
from rag.retrieval.retrieve import Retriever
from rag.storage.pgstore import PostgresStore
from rag.storage.store import Index, load_chunks, load_or_build
from rag.storage.store_api import Store

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"  # git-ignored: holds full answer texts for auditing
EXIT_OK, EXIT_QUALITY, EXIT_USAGE, EXIT_INVALID = 0, 1, 2, 3


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile: the smallest value that at least `fraction` of the values do not exceed.
    With fewer than 20 values the 95th percentile is simply the slowest run."""
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[rank - 1]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Quality numbers come only from runs that completed; failed runs are counted separately."""
    valid = [row for row in rows if not row.get("error")]
    summary: dict[str, Any] = {"runs": len(rows), "valid_runs": len(valid), "errors": len(rows) - len(valid)}
    if not valid:
        return summary

    # A question counts as passing when MORE THAN HALF of its repeats pass, which tolerates one unlucky
    # sample out of three but not a question that is wrong most of the time.
    by_question: dict[str, list[bool]] = defaultdict(list)
    for row in valid:
        by_question[row["id"]].append(row["passed"])
    stable = {qid: sum(results) * 2 > len(results) for qid, results in by_question.items()}

    walls = [row["stats"]["wall_seconds"] for row in valid]
    costs = [row["stats"]["cost_usd"] for row in valid]
    summary.update(
        run_pass_rate=sum(row["passed"] for row in valid) / len(valid),
        question_pass_rate=sum(stable.values()) / len(stable),
        failing_questions=sorted(qid for qid, ok in stable.items() if not ok),
        grounded_rate=sum(row["grounded"] for row in valid) / len(valid),
        claims_supported_rate=sum(not row.get("unsupported") for row in valid) / len(valid),
        p50_wall_seconds=percentile(walls, 0.5),
        p95_wall_seconds=percentile(walls, 0.95),
        mean_cost_usd=(sum(costs) / len(costs)) if None not in costs else None,
        mean_llm_calls=sum(row["stats"]["llm_calls"] for row in valid) / len(valid),
        mean_input_tokens=sum(row["stats"]["input_tokens"] for row in valid) / len(valid),
        mean_output_tokens=sum(row["stats"]["output_tokens"] for row in valid) / len(valid),
    )
    return summary


def thresholds_for(config: dict[str, Any], suite_name: str) -> dict[str, float]:
    """The general thresholds with this suite's own settings laid over them (thresholds.json, key "suites").
    The injection suite is stricter: one repeat that obeys an attack is a finding, whatever the other repeats did."""
    general = {key: value for key, value in config.items() if key != "suites"}
    return {**general, **config.get("suites", {}).get(suite_name, {})}


def run_problems(summary: dict[str, Any], thresholds: dict[str, float]) -> list[str]:
    """Reasons the run is invalid (so its quality numbers should not be read at all)."""
    problems: list[str] = []
    if summary["valid_runs"] == 0:
        problems.append("no run completed")
    if summary["errors"] > thresholds["max_errors"]:
        problems.append(f"{summary['errors']} run(s) failed with errors (allowed: {thresholds['max_errors']})")
    return problems


def quality_failures(summary: dict[str, Any], thresholds: dict[str, float]) -> list[str]:
    """One message per quality gate that failed. Only meaningful when run_problems() is empty."""
    failures: list[str] = []
    if summary["question_pass_rate"] < thresholds["min_question_pass_rate"]:
        failures.append(
            f"question pass rate {summary['question_pass_rate']:.0%} is below {thresholds['min_question_pass_rate']:.0%}"
            f" (failing: {', '.join(summary['failing_questions'])})"
        )
    if "min_run_pass_rate" in thresholds and summary["run_pass_rate"] < thresholds["min_run_pass_rate"]:
        failures.append(f"run pass rate {summary['run_pass_rate']:.0%} is below {thresholds['min_run_pass_rate']:.0%}")
    if "min_claims_supported_rate" in thresholds and summary["claims_supported_rate"] < thresholds["min_claims_supported_rate"]:
        failures.append(f"claims supported {summary['claims_supported_rate']:.0%} is below "
                        f"{thresholds['min_claims_supported_rate']:.0%} (figures the cited records do not state)")
    if summary["grounded_rate"] < thresholds["min_grounded_rate"]:
        failures.append(f"grounded rate {summary['grounded_rate']:.0%} is below {thresholds['min_grounded_rate']:.0%}")
    if summary["p95_wall_seconds"] > thresholds["max_p95_wall_seconds"]:
        failures.append(f"p95 latency {summary['p95_wall_seconds']:.1f}s is above {thresholds['max_p95_wall_seconds']}s")
    cost = summary["mean_cost_usd"]
    if cost is None:
        failures.append("cost is unknown for this model (add it to PRICE_PER_MTOK), so the cost gate cannot be checked")
    elif cost > thresholds["max_mean_cost_usd"]:
        failures.append(f"mean cost ${cost:.4f} is above ${thresholds['max_mean_cost_usd']}")
    return failures


def judge(summary: dict[str, Any], config: dict[str, Any], suite_name: str) -> tuple[list[str], list[str]]:
    """(reasons the run is invalid, quality gates that failed) using the thresholds of this suite."""
    thresholds = thresholds_for(config, suite_name)
    problems = run_problems(summary, thresholds)
    return problems, ([] if problems else quality_failures(summary, thresholds))


def git_state(folder: Path | None = None) -> str:
    """Short commit id of a git checkout, with '+dirty' when it has uncommitted changes."""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=folder, capture_output=True,
                                text=True, timeout=10, check=True).stdout.strip()
        changes = subprocess.run(["git", "status", "--porcelain"], cwd=folder, capture_output=True,
                                 text=True, timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return (commit or "unknown") + ("+dirty" if changes else "")


def run_one(
    question: Question,
    repeat: int,
    retriever: Retriever,
    index: Store,
    chat_factory: Callable[[], Chat],
    runner: Callable[[str], str],
    model: str,
    verifier_mode: str = "off",
    dataset_version: str | None = None,
    capture_evidence: bool = False,
    source_files: dict[str, str] | None = None,
) -> dict[str, Any]:
    stats = RunStats(model=model)
    tools = Tools(retriever, index, runner, AS_OF, stats=stats, dataset_version=dataset_version, source_files=source_files)
    row: dict[str, Any] = {"id": question.id, "kind": question.kind, "repeat": repeat}
    try:
        # A fresh chat (and so a fresh circuit breaker) per run: one outage must not poison the rest.
        answer = run_agent(build_prompt(question), tools, metered_chat(chat_factory(), stats), verifier_mode=verifier_mode)
    except Exception as problem:  # deliberate: one broken run is recorded, not allowed to lose the others
        row.update(passed=False, grounded=False, error=f"{type(problem).__name__}: {problem}", text="")
    else:
        score = score_answer(question, answer)
        row.update(passed=score.passed, grounded=score.grounded, ids_line_found=score.ids_line_found,
                   missing_ids=list(score.missing_ids), extra_ids=list(score.extra_ids),
                   missing_cites=list(score.missing_cites), missing_facts=list(score.missing_facts),
                   forbidden_hits=list(score.forbidden_hits), ungrounded=answer.ungrounded, unsupported=answer.unsupported,
                   truncated=answer.truncated, verification_failed=answer.verification_failed,
                   verification_status=answer.verification_status, text=answer.text,
                   error=answer.status if answer.status in ("retrieval_failed", "budget_exhausted") else None)
        row["answer_status"] = answer.status
        if capture_evidence:
            row["query"] = question.text
            row["context"] = "\n\n".join(item.text for item in answer.evidence)
            row["dataset_version"] = answer.dataset_version
    row["stats"] = stats.to_dict()
    return row


def slug(text: str) -> str:
    """Safe for use in a file name."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)


@contextmanager
def open_store(kind: str, project: Path, embed: Embedder, embedder_id: str, cache: Path) -> Iterator[Store]:
    """The store the agent searches. "temp" (the default) is what ships: a throw-away PostgreSQL schema filled by
    PostgresStore.sync, so retrieval, filters and the graph run through the production SQL. "memory" is the offline
    stand-in."""
    if kind == "memory":
        yield load_or_build(project, embed, embedder_id, cache)
        return
    with temporary_database_schema(len(embed(["probe"])[0])) as conn:
        store = PostgresStore(conn, "eval")
        files = capture(project)
        with materialize(files) as folder:
            store.sync(load_chunks(folder), embed, embedder_id, source_files=files)
        yield store


def run_suite(suite_name: str, args: argparse.Namespace, api_key: str, config: dict[str, Any],
              stamp: str) -> int:
    suite = SUITES[suite_name]
    thresholds = thresholds_for(config, suite_name)  # recorded with the results
    wanted = {q for q in args.only.split(",") if q}
    questions = [q for q in suite.questions if not wanted or q.id in wanted]
    if not questions:
        print(f"[{suite_name}] no question matches --only {args.only!r}.", file=sys.stderr)
        return EXIT_USAGE

    records_root = Path(args.records)
    project_for_chunks = records_root / SAMPLE_PROJECT if suite_name == "golden" else INJECTION_PROJECT
    project_for_checker = SAMPLE_PROJECT if suite_name == "golden" else INJECTION_PROJECT.resolve()
    embed, embedder_id = embedder_from_env() if args.real_embeddings else (hashing_embed, HASHING_ID)
    RESULTS_DIR.mkdir(exist_ok=True)

    def chat_factory() -> Chat:
        return make_anthropic_chat(api_key, model=args.model)

    rows_file = RESULTS_DIR / f"{slug(args.model)}-{suite_name}-{stamp}.jsonl"
    rows: list[dict[str, Any]] = []
    # The memory store keeps its own index cache, so it can never overwrite the one the application uses.
    cache = RESULTS_DIR / f"index-{suite_name}-{slug(embedder_id)}.json"
    with open_store(args.store, project_for_chunks, embed, embedder_id, cache) as store:
        if isinstance(store, PostgresStore):
            snapshot = store.source_snapshot()
            if snapshot is None:
                raise RuntimeError("Evaluation store has no source snapshot")
            dataset_version, source_files = snapshot
        elif isinstance(store, Index) and store.source_files is not None:
            source_files = store.source_files
            dataset_version = version(source_files)
        else:
            raise RuntimeError("Evaluation store has no source snapshot")
        runner = make_cpr_runner(records_root, project_for_checker, source_files)
        retriever = Retriever(store, embed)
        with rows_file.open("w", encoding="utf-8") as sink:  # one line per finished run, so a crash loses nothing
            for repeat in range(args.repeats):
                for question in questions:
                    row = run_one(question, repeat, retriever, store, chat_factory, runner, args.model,
                                  args.verifier_mode, dataset_version, args.capture_evidence, source_files)
                    rows.append(row)
                    sink.write(json.dumps(row) + "\n")
                    sink.flush()
                    flag = "ERROR" if row["error"] else ("PASS" if row["passed"] else "FAIL")
                    print(f"[{suite_name}] [{flag}] {question.id:<14} run{repeat} {row['stats']['wall_seconds']:5.1f}s "
                          f"calls={row['stats']['llm_calls']} cost={row['stats']['cost_usd']}", flush=True)

    summary = summarize(rows)
    provenance = {
        "suite": suite_name, "model": args.model, "repeats": args.repeats, "only": sorted(wanted),
        "as_of": AS_OF, "embedder": embedder_id, "store": args.store, "thresholds": thresholds,
        "dataset_version": dataset_version,
        "verifier_mode": args.verifier_mode,
        "capture_evidence": args.capture_evidence,
        "code_commit": git_state(), "records_commit": git_state(records_root),
        "python": sys.version.split()[0],
    }
    summary_file = rows_file.with_suffix(".summary.json")
    summary_file.write_text(json.dumps({"provenance": provenance, "summary": summary}, indent=2), encoding="utf-8")

    print(f"\n=== SUMMARY [{suite_name}] ===")
    for key, value in summary.items():
        print(f"{key:>20}: {round(value, 4) if isinstance(value, float) else value}")
    print(f"answers: {rows_file}\nprovenance and summary: {summary_file}")

    problems, failures = judge(summary, config, suite_name)
    for problem in problems:
        print(f"RUN INVALID: {problem}", file=sys.stderr)
    if problems:
        return EXIT_INVALID
    for failure in failures:
        print(f"GATE FAILED: {failure}", file=sys.stderr)
    return EXIT_QUALITY if failures else EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the golden questions against the live agent.")
    parser.add_argument("--suite", choices=[*SUITES, "all"], default="golden")
    parser.add_argument("--model", default="claude-sonnet-5-5")
    parser.add_argument("--verifier-mode", choices=["off", "shadow", "enforce"], default="off")
    parser.add_argument("--capture-evidence", action="store_true",
                        help="include question and retrieved tool content in local evaluation results (sensitive)")
    parser.add_argument("--repeats", type=int, default=3, help="runs per question; answers vary between runs")
    parser.add_argument("--only", default="", help="comma-separated question ids")
    parser.add_argument("--records", default="data/records")
    parser.add_argument("--thresholds", default=str(HERE / "data" / "thresholds.json"))
    parser.add_argument("--store", choices=["temp", "memory"], default="temp",
                        help="temp: a throw-away PostgreSQL schema, the production path (default). memory: the offline stand-in")
    parser.add_argument("--real-embeddings", action="store_true",
                        help="use EMBED_* from .env instead of the offline hashing embedder")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    load_env()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY is not set (see .env.example).", file=sys.stderr)
        return EXIT_USAGE
    thresholds = json.loads(Path(args.thresholds).read_text(encoding="utf-8"))
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    names = list(SUITES) if args.suite == "all" else [args.suite]
    return max(run_suite(name, args, api_key, thresholds, stamp) for name in names)


if __name__ == "__main__":
    sys.exit(main())
