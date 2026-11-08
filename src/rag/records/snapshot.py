"""Capture the exact project CSVs used by ingestion and the deterministic checker."""
import hashlib
import json
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

FILES = ("rfis.csv", "submittals.csv", "change_events.csv", "daily_reports.csv", "punch_items.csv")


def capture(folder: Path) -> dict[str, str]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Project folder not found: {folder}")
    return {name: (folder / name).read_bytes().decode("utf-8") for name in FILES if (folder / name).is_file()}


def version(files: Mapping[str, str]) -> str:
    payload = json.dumps(dict(files), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@contextmanager
def materialize(files: Mapping[str, str]) -> Iterator[Path]:
    if not files or any(name not in FILES or not isinstance(text, str) for name, text in files.items()):
        raise ValueError("Invalid or empty project snapshot")
    with tempfile.TemporaryDirectory(prefix="rag-snapshot-") as directory:
        folder = Path(directory)
        for name, text in files.items():
            (folder / name).write_bytes(text.encode("utf-8"))
        yield folder
