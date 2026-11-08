"""Convert explicitly captured evaluation traces into a Foundry-compatible JSONL dataset."""
import argparse
import json
import os
from pathlib import Path
from typing import Any


def export_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    result = []
    for row in rows:
        if row.get("error"):
            continue
        if any(not isinstance(row.get(key), str) for key in ("query", "text", "context")):
            raise ValueError("Results must be created with --capture-evidence before exporting")
        item = {"query": row["query"], "response": row["text"], "context": row["context"],
                "case_id": str(row["id"]), "dataset_version": str(row.get("dataset_version") or "unknown")}
        if isinstance(row.get("reference_answer"), str):
            item["ground_truth"] = row["reference_answer"]
        if isinstance(row.get("as_of"), str):
            item["as_of"] = row["as_of"]
        result.append(item)
    if not result:
        raise ValueError("No completed evaluation rows to export")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Export captured evaluation results for Microsoft Foundry.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path, help="new JSONL file; may contain sensitive project data")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    exported = export_rows(rows)
    # Do not overwrite a previous dataset; restrict this file because it contains project text.
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as sink:
        for row in exported:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Exported {len(exported)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
