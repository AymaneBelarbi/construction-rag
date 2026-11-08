"""Run the versioned, source-checked cases. This calls paid APIs and is never run by CI."""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from evals.run_eval import open_store
from rag.agent import Tools, make_cpr_runner, run_agent
from rag.infra.telemetry import RunStats, metered_chat
from rag.providers.embed import HASHING_ID, hashing_embed, min_vector_score_for
from rag.providers.llm import embedder_from_env, load_env, make_anthropic_chat
from rag.records.snapshot import capture, version
from rag.retrieval.retrieve import Retriever
from rag.storage.pgstore import PostgresStore
from rag.storage.store import Index

DATASET_SHA256 = "99ccc466e43ab41898f28d2bb7b0664e523406e9394cdd19272256ede1ff1693"
EXPANDED_DATASET_SHA256 = "414764a3273b583201b050e8aca220eaecac39938be4147f28fccc376ef70cbc"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run source-checked robustness cases for human or Foundry evaluation.")
    parser.add_argument("--output", type=Path, required=True, help="new local JSONL file containing project data")
    parser.add_argument("--suite", choices=["v1", "v2"], default="v1", help="frozen development question set")
    parser.add_argument("--records", type=Path, default=Path("data/records"))
    parser.add_argument("--model", default="claude-haiku-5-5")
    parser.add_argument("--store", choices=["temp", "memory"], default="temp")
    parser.add_argument("--real-embeddings", action="store_true")
    parser.add_argument("--verifier-mode", choices=["off", "shadow", "enforce"], default="enforce")
    parser.add_argument("--capture-trace", action=argparse.BooleanOptionalAction, default=True,
                        help="include sensitive candidate and tool details in the protected output (default: on)")
    args = parser.parse_args()
    dataset_bytes = (Path(__file__).parent / "data" / f"robustness_{args.suite}.json").read_bytes()
    expected_hash = DATASET_SHA256 if args.suite == "v1" else EXPANDED_DATASET_SHA256
    if hashlib.sha256(dataset_bytes).hexdigest() != expected_hash:
        parser.error("Frozen evaluation set changed; create a new version rather than altering version 1")
    dataset = json.loads(dataset_bytes)
    project = args.records / dataset["project"]
    files = capture(project)
    if version(files) != dataset["dataset_version"]:
        parser.error("Project CSVs do not match the frozen evaluation dataset")
    try:
        commit = subprocess.run(["git", "-C", str(args.records), "rev-parse", "HEAD"], text=True,
                                capture_output=True, check=True, timeout=10).stdout.strip()
        changes = subprocess.run(["git", "-C", str(args.records), "status", "--porcelain", "--", "cpr", "schema"],
                                 text=True, capture_output=True, check=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        parser.error("Cannot establish the checker repository version")
    if commit != dataset["records_commit"] or changes:
        parser.error("Checker code or schema do not match the pinned evaluation version")
    load_env()
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        parser.error("ANTHROPIC_API_KEY is required")
    if args.real_embeddings and not os.environ.get("EMBED_API_KEY"):
        parser.error("EMBED_API_KEY is required for real embeddings")
    embed, embedder_id = embedder_from_env() if args.real_embeddings else (hashing_embed, HASHING_ID)
    # Output is exclusive and owner-readable because it includes the original project evidence.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    cache = args.output.with_suffix(".index.json")
    failures = 0
    with os.fdopen(fd, "w", encoding="utf-8") as sink, open_store(args.store, project, embed, embedder_id, cache) as store:
        if isinstance(store, PostgresStore):
            snapshot = store.source_snapshot()
            if snapshot is None or snapshot[0] != dataset["dataset_version"]:
                raise RuntimeError("Indexed data differs from the frozen evaluation dataset")
            files = snapshot[1]
        elif isinstance(store, Index) and store.source_files is not None:
            files = store.source_files
            if version(files) != dataset["dataset_version"]:
                raise RuntimeError("Indexed data differs from the frozen evaluation dataset")
        else:
            raise RuntimeError("Evaluation store has no source snapshot")
        retriever = Retriever(store, embed, min_vector_score=min_vector_score_for(embedder_id, os.environ))
        checker = make_cpr_runner(args.records, Path(dataset["project"]), files)
        for case in dataset["cases"]:
            stats = RunStats(model=args.model)
            trace: list[dict[str, Any]] | None = [] if args.capture_trace else None
            tools = Tools(retriever, store, checker, dataset["as_of"], stats=stats,
                          dataset_version=dataset["dataset_version"], source_files=files, trace=trace)
            chat = metered_chat(make_anthropic_chat(key, model=args.model), stats)
            row: dict[str, Any] = {"id": case["id"], "query": case["query"], "reference_answer": case["reference_answer"],
                                   "suite": args.suite, "suite_sha256": expected_hash, "category": case["category"],
                                   "source_fields": case["source_fields"], "failure_condition": case["failure_condition"],
                                   "dataset_version": dataset["dataset_version"], "model": args.model,
                                   "as_of": dataset["as_of"], "verifier_mode": args.verifier_mode,
                                   "embedder": embedder_id, "expected_source_ids": case["source_ids"], "error": None}
            try:
                answer = run_agent(f"{case['query']}\n\n(Today's date for this question is {dataset['as_of']}.)",
                                   tools, chat, verifier_mode=args.verifier_mode)
                row.update(text=answer.text, context="\n\n".join(e.text for e in answer.evidence),
                           answer_status=answer.status, verification_status=answer.verification_status,
                           withheld_because=answer.withheld_because,
                           cited_ids=answer.cited_ids,
                           retrieved_records=[{"record_id": record.record_id, "revision": record.revision,
                                               "evidence_id": record.evidence_id, "complete": record.complete}
                                              for item in answer.evidence for record in item.records])
                if answer.status in ("retrieval_failed", "budget_exhausted"):
                    row["error"] = answer.status
                    failures += 1
            except Exception as problem:
                row.update(text="", context="\n\n".join(e.text for e in tools.evidence), error=type(problem).__name__)
                failures += 1
            row["stats"] = stats.to_dict()
            if trace is not None:
                row["trace"] = trace
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            sink.flush()
            print(f"{case['id']}: {row.get('answer_status', 'error')}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
