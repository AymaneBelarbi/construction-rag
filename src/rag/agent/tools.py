"""Tool execution: run what the model asked for, keep the evidence it produced, enforce the per-question limits.

Failures come back as text starting with ERROR: so the model can read them and adapt. Only records that a tool
really returned (`seen_ids`) may be cited in the final answer.
"""
import csv
import datetime
import hashlib
import io
import logging
import os
import subprocess
import sys
import time
from collections.abc import Callable
from contextlib import nullcontext
from decimal import Decimal
from pathlib import Path
from typing import Any

import psycopg

from rag.agent.schemas import KNOWN_TOOL_NAMES
from rag.answering.answer import format_context, format_record
from rag.answering.evidence import RecordEvidence, ToolEvidence
from rag.infra.resilience import CircuitOpenError, PermanentError, TransientError, remaining_timeout
from rag.infra.telemetry import RunStats, log_event
from rag.records.chunker import Chunk, find_record_ids, mark_latest_revisions, row_to_chunk
from rag.records.snapshot import materialize, version
from rag.retrieval.retrieve import Retriever
from rag.security.injection import looks_like_injection, strip_hidden_and_control
from rag.storage.store_api import Store, StoreError, chunk_matches

ERROR_PREFIX = "ERROR:"
# Limits on what one question may cost, so a fooled or confused model cannot multiply paid calls or flood its own context.
MAX_TOOL_CALLS = 12  # tool calls per question
MAX_RESULT_CHARS = 20_000  # one tool result (about 5,000 tokens) is cut after this many characters
MAX_LIST_RECORDS = 100
# The only environment variables the checker subprocess receives. It runs code from a cloned repository, so it must
# not see the API keys and the database password that load_env put into this process.
CHECKER_ENV_NAMES = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR")


def is_tool_error(result: str) -> bool:
    """Tools report failure as text starting with ERROR:, so the model can read it and adapt."""
    return result.startswith(ERROR_PREFIX)


# The checker's own meaning of "open" (cpr/checks.py: OPEN_RFI, OPEN_SUBMITTAL, OPEN_EVENT, OPEN_PUNCH). Its punch
# count "open 14" includes items ready for review, which a reader takes for status "open" unless told.
CHECKER_DEFINITIONS = (
    "Definitions used by this report: open RFIs have status open; open submittals are not_submitted, in_review or "
    "revise_and_resubmit; open change exposure counts events that are open, priced or submitted; open punch items "
    "have status open or ready_for_review; closed and void items are excluded."
)


def make_cpr_runner(records_root: Path, project_folder: Path, source_files: dict[str, str] | None = None) -> Callable[[str], str]:
    """Run `python -m cpr check <folder> --as-of <date>` inside the records repo.
    project_folder is relative to records_root. The date comes from the user and
    is validated in Tools before it gets here."""

    def run(as_of: str) -> str:
        with materialize(source_files) if source_files is not None else nullcontext(records_root / project_folder) as folder:
            if not folder.is_dir() or not any(folder.glob("*.csv")):
                return f"ERROR: no project records found in {project_folder}, so nothing was checked"
            checked = str(folder.resolve())
            result = subprocess.run(
                [sys.executable, "-m", "cpr", "check", checked, "--as-of", as_of],
                cwd=records_root, capture_output=True, text=True, timeout=remaining_timeout(30),
                env={name: os.environ[name] for name in CHECKER_ENV_NAMES if name in os.environ},
            )
        # The report names the folder it checked, which for a snapshot is a temporary path on this machine. The model
        # quotes the report, so the user would see it: name the project instead.
        output = (result.stdout + result.stderr).strip().replace(checked, str(project_folder))
        # Exit code 1 only means "the data has errors" and the report is complete. A crash (a missing module, a bad
        # folder) also exits 1 or 2 but prints no report, which always starts with this line.
        if result.returncode not in (0, 1) or not output.startswith("Project logs in"):
            return f"{ERROR_PREFIX} the checker failed (exit {result.returncode}): {output}"
        return f"{output}\n\n{CHECKER_DEFINITIONS}"

    return run


def _text_arg(args: dict[str, Any], key: str) -> str:
    value = args[key]  # a missing key raises KeyError, reported to the model by Tools.call
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _optional_text_arg(args: dict[str, Any], key: str) -> str | None:
    value = args.get(key)
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _top_k(args: dict[str, Any]) -> int:
    raw = args.get("top_k")
    if raw is None:
        return 5
    try:
        return max(1, min(int(raw), 10))
    except (TypeError, ValueError):
        raise ValueError("top_k must be a number") from None


def _figure(field: str, value: Decimal | None) -> str:
    """A number as the model should quote it: dollars with cents, others without a trailing .00."""
    if value is None:
        return "no value"
    if field == "estimated_cost":
        return f"${value:,.2f}"
    return f"{value:,.0f}" if value == value.to_integral_value() else f"{value:,.2f}"


class Tools:
    def __init__(
        self,
        retriever: Retriever,
        store: Store,
        run_checks_fn: Callable[[str], str],
        as_of: str,
        stats: RunStats | None = None,
        dataset_version: str | None = None,
        source_files: dict[str, str] | None = None,
        trace: list[dict[str, Any]] | None = None,
    ) -> None:
        datetime.date.fromisoformat(as_of)  # raises ValueError for a bad date, before anything runs
        self.retriever = retriever
        self.store = store
        self.run_checks_fn = run_checks_fn
        self.as_of = as_of  # fixed by the user: the model cannot choose the date
        self.stats = stats  # optional: when given, every tool call is timed and counted
        self.dataset_version = dataset_version
        self.source_files = dict(source_files) if source_files is not None else None
        self.trace = trace  # opt-in: contains potentially sensitive project and model data
        # IDs of records that tools really returned. Only these may be cited in an answer.
        self.seen_ids: set[str] = set()
        # IDs the model asked for and the tools said do not exist. The model may say so in plain text.
        self.not_found_ids: set[str] = set()
        self.evidence: list[ToolEvidence] = []
        self._pending_chunks: list[Chunk] = []
        self._pending_computed = ""
        self.tool_calls = 0
        self.tool_failures = 0  # calls that failed because something broke (database, checker), not because of bad arguments

    def call(self, name: str, args: dict[str, Any]) -> str:
        """Run a tool and record how long it took. Never raises for tool failures."""
        started = time.perf_counter()
        self._pending_chunks = []
        self._pending_computed = ""
        original = self._dispatch(name, args)
        computed_id = None
        if self._pending_computed and not is_tool_error(original):
            identity = f"{self.dataset_version}\0{self.as_of}\0{name}\0{self._pending_computed}"
            computed_id = "CALC-" + hashlib.sha256(identity.encode()).hexdigest()[:16].upper()
            original = f"[{computed_id}] Computed result:\n{original}"
        if name == "list_records" and len(original) > MAX_RESULT_CHARS:
            original = "ERROR: exact listing exceeds the result size limit; narrow the filters."
            self._pending_computed = ""
            self._pending_chunks = []
        truncated = len(original) > MAX_RESULT_CHARS
        visible = original[:MAX_RESULT_CHARS] if truncated else original
        result = (f"{visible}\n[result cut: {len(original) - MAX_RESULT_CHARS} more characters not shown]"
                  if truncated else visible)
        seconds = time.perf_counter() - started
        is_error = is_tool_error(result)
        computed_ids: set[str] = set()
        if not is_error and self._pending_computed and not truncated:
            try:
                computed_ids = {i for i in find_record_ids(self._pending_computed) if self.store.get(i)}
            except (OSError, psycopg.Error, StoreError) as problem:
                self.tool_failures += 1
                result = f"ERROR: could not validate computed record IDs ({type(problem).__name__})"
                is_error = True
        if not is_error:
            records = []
            cursor = 0
            for chunk in self._pending_chunks:
                marker = f"[{chunk.record_id}] "
                start = original.find(marker + chunk.text, cursor)
                if start < 0:
                    continue
                content_start = start + len(marker)
                end = content_start + len(chunk.text)
                cursor = end
                if content_start >= len(visible):
                    continue
                complete = end <= len(visible)
                records.append(RecordEvidence(chunk.record_id.upper(), chunk.record_type, chunk.key[1], self.dataset_version,
                                              visible[content_start:min(end, len(visible))], complete))
                if complete:
                    self.seen_ids.add(chunk.record_id.upper())
                    if chunk.record_type == "submittal":
                        alias = f"{chunk.record_id}.REV-{chunk.key[1]}".upper()
                        if f"Revision citation: [{alias}]" in visible[end:]:
                            self.seen_ids.add(alias)
            computed = self._pending_computed if not truncated else ""
            if computed:
                self.seen_ids.update(computed_ids)
                if computed_id is not None:
                    self.seen_ids.add(computed_id)
            self.evidence.append(ToolEvidence(name, self.dataset_version, result, truncated, tuple(records), computed,
                                              computed_id if computed else None))
        if self.stats is not None:
            self.stats.add_tool_call(seconds, is_error)
        if self.trace is not None:
            evidence = self.evidence[-1] if not is_error else None
            self.trace.append({"event": "tool_call", "name": name, "requested_arguments": args,
                               "error": is_error, "truncated": truncated,
                               "evidence_id": evidence.evidence_id if evidence else None,
                               "records": [{"evidence_id": record.evidence_id, "record_id": record.record_id,
                                            "revision": record.revision, "complete": record.complete}
                                           for record in evidence.records] if evidence else []})
        # The tool name comes from the model, so it is only logged when it is one of ours. Anything
        # else could carry words from the question or the records into the log.
        log_event("tool_call", tool=name if name in KNOWN_TOOL_NAMES else "unknown",
                  seconds=round(seconds, 6), error=is_error, result_chars=len(result))
        return result

    @property
    def has_evidence(self) -> bool:
        """True when some tool really returned project data: at least one record, or a checker report.
        'No matching records' and errors are not evidence."""
        return bool(self.seen_ids) or any(item.computed for item in self.evidence)

    @property
    def computed_text(self) -> str:
        """What code calculated (checker report, aggregate results): any citation may use these figures."""
        return "\n".join(item.computed for item in self.evidence if item.computed)

    def _remember(self, chunks: list[Chunk]) -> None:
        """Stage records until the actual, possibly truncated tool result is known."""
        self._pending_chunks.extend(chunks)

    @property
    def record_texts(self) -> dict[str, str]:
        texts: dict[str, list[str]] = {}
        for item in self.evidence:
            if item.computed_id and item.computed:
                texts.setdefault(item.computed_id, []).append(item.computed)
            for record in item.records:
                if record.complete and record.text not in texts.setdefault(record.record_id, []):
                    texts[record.record_id].append(record.text)
                    if record.record_type == "submittal":
                        texts.setdefault(f"{record.record_id}.REV-{record.revision}".upper(), []).append(record.text)
        return {key: "\n".join(values) for key, values in texts.items()}

    @property
    def evidence_text(self) -> str:
        """Everything the tools returned so far, as one string."""
        return "\n".join(item.text for item in self.evidence)

    def _dispatch(self, name: str, args: dict[str, Any]) -> str:
        """Run a tool. Failures come back as text so the LLM can read the error and adapt."""
        self.tool_calls += 1
        if self.tool_calls > MAX_TOOL_CALLS:
            return f"{ERROR_PREFIX} too many tool calls for one question (the limit is {MAX_TOOL_CALLS}). Answer with what you have."
        handlers = {
            "list_records": self._list_records,
            "search_records": self._search_records,
            "get_record": self._get_record,
            "related_records": self._related_records,
            "run_checks": self._run_checks,
            "aggregate_records": self._aggregate_records,
        }
        if name not in handlers:
            return f"ERROR: unknown tool {name!r}. Available: {', '.join(handlers)}"
        try:
            return handlers[name](args)
        except KeyError as missing:
            return f"ERROR: missing argument {missing}"
        except ValueError as problem:
            return f"ERROR: {problem}"
        except (OSError, subprocess.SubprocessError, psycopg.Error, StoreError,
                TransientError, PermanentError, CircuitOpenError) as problem:
            self.tool_failures += 1
            return f"ERROR: the tool failed ({type(problem).__name__}: {problem})"

    @staticmethod
    def _report_suspicious(chunks: list[Chunk]) -> None:
        """Log (by id only) when a record flagged as instruction-like is about to reach the model."""
        flagged = sorted({c.record_id for c in chunks if c.metadata.get("injection_suspected")})
        if flagged:
            log_event("suspicious_record_returned", level=logging.WARNING, record_ids=flagged)

    def _remember_missing(self, record_id: str) -> None:
        """Keep the ID the model asked for, to name it in a refusal, but only when it really is a record ID: the
        model may pass any text, and the refusal is written by code and shown to the user."""
        candidate = record_id.strip().upper()
        if len(candidate) <= 40 and find_record_ids(candidate) == [candidate]:
            self.not_found_ids.add(candidate)

    def _list_records(self, args: dict[str, Any]) -> str:
        record_type = _text_arg(args, "record_type")
        if record_type not in ("rfi", "submittal", "change_event", "daily_report", "punch_item"):
            raise ValueError("unknown record type")
        filters = {"record_type": record_type}
        for key in ("status", "ball_in_court", "responsible_party"):
            value = _optional_text_arg(args, key)
            if value:
                filters[key] = value
        filenames = {"rfi": "rfis.csv", "submittal": "submittals.csv", "change_event": "change_events.csv",
                     "daily_report": "daily_reports.csv", "punch_item": "punch_items.csv"}
        if (self.source_files is None or self.dataset_version is None or
                version(self.source_files) != self.dataset_version or filenames[record_type] not in self.source_files):
            raise ValueError("exact listing requires the indexed source snapshot")
        reader = csv.DictReader(io.StringIO(self.source_files[filenames[record_type]].lstrip("\ufeff"), newline=""))
        if not set(filters).difference({"record_type"}).issubset(set(reader.fieldnames or [])):
            raise ValueError("filter is not available in the source")
        source = mark_latest_revisions([row_to_chunk(record_type, row) for row in reader])
        expected = sorted((chunk.record_id, chunk.key[1]) for chunk in source
                          if chunk.metadata.get("is_latest", "yes") == "yes" and chunk_matches(chunk, filters))
        matches = self.store.list_records(filters, MAX_LIST_RECORDS)
        if [(chunk.record_id, chunk.key[1]) for chunk in matches] != expected[:MAX_LIST_RECORDS + 1]:
            raise ValueError("index does not match source snapshot; re-ingest the project")
        complete = len(matches) <= MAX_LIST_RECORDS
        records = matches[:MAX_LIST_RECORDS]
        self._remember(records)
        scope = ", ".join(f"{key}={value}" for key, value in filters.items())
        ids = ", ".join(chunk.record_id for chunk in records) or "none"
        self._pending_computed = (f"Exact current-record query ({scope}): "
                                  f"{'complete' if complete else 'incomplete: additional matches omitted'}. "
                                  f"Returned {len(records)} records: {ids}.")
        return self._pending_computed + ("\n\n" + "\n".join(format_record(c) for c in records) if records else "")

    def _search_records(self, args: dict[str, Any]) -> str:
        query = _text_arg(args, "query")
        filters = {}
        for key in ("record_type", "status"):
            value = _optional_text_arg(args, key)
            if value:
                filters[key] = value
        if not args.get("include_superseded"):
            filters["is_latest"] = "yes"
        hits = self.retriever.search(query, top_k=_top_k(args), filters=filters)

        # Tell the model when it asked for an ID that does not exist, instead of silently
        # returning look-alikes. Change order numbers (CO-) are not records, so they are skipped.
        missing = [i for i in find_record_ids(query) if not i.startswith("CO-") and not self.store.get(i)]
        note = "".join(f"No record with ID {i}.\n" for i in missing)
        self.not_found_ids.update(missing)

        self._remember([hit.chunk for hit in hits])
        self._report_suspicious([hit.chunk for hit in hits])
        return note + (format_context(hits) if hits else "No matching records.")

    def _get_record(self, args: dict[str, Any]) -> str:
        record_id = _text_arg(args, "record_id")
        chunks = self.store.get(record_id)
        if not chunks:
            self._remember_missing(record_id)
            return f"No record with ID {record_id}."
        linked = self.store.linked(record_id)
        self._remember(chunks + linked)
        self._report_suspicious(chunks + linked)
        lines = [format_record(c) for c in chunks]
        if linked:
            lines.append("Linked records:")
            lines.extend(format_record(c) for c in linked)
        return "\n".join(lines)

    def _related_records(self, args: dict[str, Any]) -> str:
        record_id = _text_arg(args, "record_id")
        if not self.store.get(record_id):
            self._remember_missing(record_id)
            return f"No record with ID {record_id}."
        related = self.store.related(record_id)
        if not related:
            if related.omitted:  # connected only through groups that were too large to list
                counts = "; ".join(f"{count} ({reason})" for reason, count in related.omitted.items())
                return f"No records listed for {record_id}. Not shown: {counts}."
            return f"No connected records for {record_id}."
        self._remember([item.chunk for item in related])
        self._report_suspicious([item.chunk for item in related])
        text = "\n\n".join(f"{format_record(item.chunk)}\n(connected via {item.via})" for item in related)
        if related.omitted:
            counts = "; ".join(f"{count} more ({reason})" for reason, count in related.omitted.items())
            text += f"\n\nNot shown: {counts}."
        return text

    def _aggregate_records(self, args: dict[str, Any]) -> str:
        record_type, field, operation = _text_arg(args, "record_type"), _text_arg(args, "field"), _text_arg(args, "operation")
        status = _optional_text_arg(args, "status")
        result = self.store.aggregate(record_type, field, operation, {"status": status} if status else None)
        scope = f"{record_type} records" + (f" with status {status}" if status else "")
        if result.count == 0:
            return f"No records: none of the {scope} has a value for {field}."
        if operation == "count":
            # Only records with a value are counted. Say so, or the number is read as how many records match.
            text = f"{result.count} of the {scope} have a value for {field} (records without one are not counted)."
        else:
            text = f"{operation} of {field} over the {result.count} {scope} that have a value for it: {_figure(field, result.value)}."
        if result.record_ids:
            more = f" and {result.more_ids} more" if result.more_ids else ""
            text += f" Reached by {', '.join(result.record_ids)}{more}."
        self._pending_computed = text
        if result.record_ids:
            records = [c for record_id in result.record_ids for c in self.store.get(record_id)
                       if c.metadata.get("is_latest", "yes") == "yes"]
            if records:
                self._remember(records)
                text += "\n\n" + "\n".join(format_record(c) for c in records)
        return text

    def _run_checks(self, args: dict[str, Any]) -> str:
        output = self.run_checks_fn(self.as_of)  # any argument the model sent is ignored
        if is_tool_error(output):
            self.tool_failures += 1
            return output
        # The report echoes text from the CSVs ("per RFI-777" in a party name), so only IDs of records that
        # really exist become citable. An empty report is not evidence, and a later empty run does not undo a good one.
        # The report prints text from the CSVs (a party name, a subject), so it gets the same treatment as a record.
        output, hidden = strip_hidden_and_control(output)
        flags = looks_like_injection(output) + (["hidden_characters"] if hidden else [])
        if flags:
            log_event("suspicious_checker_output", level=logging.WARNING, patterns=flags)
        self._pending_computed = output if output.strip() else ""
        return output
