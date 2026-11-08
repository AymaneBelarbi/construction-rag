"""INGEST, step 2: turn one CSV row into one text chunk.

One row = one chunk. Records are short, so splitting them further would only
separate a question from its answer. Every chunk starts with the record ID
and type so that both keyword search and the LLM can find and cite it.
"""
import logging
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from rag.infra.telemetry import log_event
from rag.records.graph import extract_relations
from rag.security.injection import looks_like_injection, strip_hidden_and_control


@dataclass(frozen=True)
class Chunk:
    """A dataclass is a class whose __init__ is generated from the field list.
    frozen=True means the fields cannot be changed after creation."""

    record_type: str  # "rfi", "submittal", "change_event", "daily_report", "punch_item"
    record_id: str  # what the answer cites, for example "RFI-003"
    text: str  # what gets embedded and shown to the LLM
    metadata: dict[str, str]  # small fields used for filtering
    relations: tuple[tuple[str, str, str], ...] = ()  # (relation, entity kind, name): see graph.py

    @property
    def key(self) -> tuple[str, str]:
        """Identifies one stored row: the record id plus its revision (submittals have several)."""
        return (self.record_id, str(_revision_number(self)))


def _sentences(pairs: list[tuple[str, str]]) -> str:
    """Join 'Label: value.' pieces, leaving out blank values."""
    return " ".join(f"{label}: {value.strip()}." for label, value in pairs if value.strip())


def _recorded(row: dict[str, str], field: str) -> str:
    """Distinguish an empty field in the source from a field absent from the input."""
    value = row.get(field, "").strip()
    return value or ("not recorded" if field in row else "")


def rfi_to_text(row: dict[str, str]) -> str:
    header = f"RFI {row['rfi_number']} ({row['status']})."
    body = _sentences([
        ("Subject", row.get("subject", "")),
        ("Question", row.get("question", "")),
        ("Proposed answer", row.get("proposed_answer", "")),
        ("References", row.get("references", "")),
        ("Created", row.get("created_date", "")),
        ("Date needed", _recorded(row, "date_needed")),
        ("Reason needed", row.get("reason_needed", "")),
        ("Ball in court", row.get("ball_in_court", "")),
        ("Answered", _recorded(row, "answered_date")),
        ("Answer", _recorded(row, "answer")),
        ("Cost impact", row.get("cost_impact", "")),
        ("Schedule impact", row.get("schedule_impact", "")),
        ("Raised by", row.get("created_by", "")),
    ])
    return f"{header} {body}"


def submittal_to_text(row: dict[str, str]) -> str:
    header = f"Submittal {row['submittal_number']} revision {row['revision']} ({row['status']})."
    body = _sentences([
        ("Title", row.get("title", "")),
        ("Spec section", row.get("spec_section", "")),
        ("Type", row.get("submittal_type", "")),
        ("Submitted", _recorded(row, "submitted_date")),
        ("Review due", row.get("review_due_date", "")),
        ("Returned", _recorded(row, "returned_date")),
        ("Required on site", row.get("required_on_site", "")),
        ("Lead time days", row.get("lead_time_days", "")),
        ("Ball in court", row.get("ball_in_court", "")),
        ("Responsible party", row.get("responsible_party", "")),
    ])
    return f"{header} {body}"


def change_event_to_text(row: dict[str, str]) -> str:
    header = f"Change event {row['event_number']} ({row['status']})."
    body = _sentences([
        ("Title", row.get("title", "")),
        ("Origin", row.get("origin", "")),
        ("Event date", row.get("event_date", "")),
        ("Notice required by", row.get("notice_required_by", "")),
        ("Notice given", _recorded(row, "notice_date")),
        ("Estimated cost", row.get("estimated_cost", "")),
        ("Estimated days", row.get("estimated_days", "")),
        ("Related RFI", row.get("related_rfi", "")),
        ("Change order", row.get("change_order_number", "")),
    ])
    return f"{header} {body}"


def daily_report_to_text(row: dict[str, str]) -> str:
    header = f"Daily report {row['report_date']}."
    body = _sentences([
        ("Weather", row.get("weather", "")),
        ("Work performed", row.get("work_performed", "")),
        ("Manpower", row.get("manpower_total", "")),
        ("Delays", row.get("delays", "")),
        ("Inspections", row.get("inspections", "")),
        ("Safety", row.get("safety_notes", "")),
        ("Author", row.get("author", "")),
    ])
    return f"{header} {body}"


def punch_item_to_text(row: dict[str, str]) -> str:
    header = f"Punch item {row['item_number']} ({row['status']})."
    body = _sentences([
        ("Location", row.get("location", "")),
        ("Description", row.get("description", "")),
        ("Responsible party", row.get("responsible_party", "")),
        ("Created", row.get("created_date", "")),
        ("Due", row.get("due_date", "")),
        ("Closed", _recorded(row, "closed_date")),
        ("Verified by", _recorded(row, "verified_by")),
    ])
    return f"{header} {body}"


# record type -> (function that writes the text, column that holds the ID)
_TEXT_FUNCTIONS: dict[str, Callable[[dict[str, str]], str]] = {
    "rfi": rfi_to_text,
    "submittal": submittal_to_text,
    "change_event": change_event_to_text,
    "daily_report": daily_report_to_text,
    "punch_item": punch_item_to_text,
}
_ID_COLUMN = {
    "rfi": "rfi_number",
    "submittal": "submittal_number",
    "change_event": "event_number",
    "punch_item": "item_number",
}
RECORD_TYPES = tuple(_TEXT_FUNCTIONS)
_FILTER_FIELDS = ("status", "related_rfi", "responsible_party", "ball_in_court")
# Numbers a question may need to total, compare or count (the largest crew, the longest lead time). They are kept as
# text in the chunk metadata, in plain decimal form, and Store.aggregate computes over them exactly.
NUMERIC_FIELDS: dict[str, tuple[str, ...]] = {
    "daily_report": ("manpower_total",),
    "submittal": ("lead_time_days",),
    "change_event": ("estimated_cost", "estimated_days"),
}
# A comma is only a thousands separator (1,250 or 1,250,000.50): "1,50" is a decimal comma elsewhere, and reading it as 150 would
# be a silent hundredfold error in a cost, so it is dropped.
_PLAIN_NUMBER = re.compile(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


# Matches IDs such as RFI-003, CE-001, CO-001, PL-007, S-004, DR-2026-08-03.
_ID_PATTERN = re.compile(r"\b(?:RFI|CE|CO|PL|S|DR)-\d+(?:-\d+)*\b", re.IGNORECASE)


def find_record_ids(text: str) -> list[str]:
    """Every record ID mentioned in the text, upper-cased, without repeats, in order."""
    found: list[str] = []
    for match in _ID_PATTERN.findall(text):
        if match.upper() not in found:
            found.append(match.upper())
    return found


_VALID_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


def record_id_of(record_type: str, row: dict[str, str]) -> str:
    if record_type == "daily_report":
        record_id = f"DR-{row['report_date']}"  # daily reports have no number, so the date is the ID
    else:
        record_id = row[_ID_COLUMN[record_type]]
    record_id = record_id.strip()
    # The ID is printed in citations like [RFI-003] and written to logs, so text that merely sits in the ID column
    # ("PL-001] [RFI-777") must not get through. The message does not repeat the bad value.
    if not _VALID_ID.fullmatch(record_id):
        raise ValueError(f"A {record_type} row has an unusable ID: use letters, digits, '.', '_' and '-' (up to 64 characters)")
    return record_id


MAX_FIELD_CHARS = 20_000  # one cell of a project log is a sentence or a paragraph; more is an export gone wrong or an attack


def _clean_row(record_type: str, row: Mapping[Any, object]) -> tuple[dict[str, str], bool]:
    """The row as plain text: missing cells become "", cells beyond the header are dropped, control and hidden
    characters are removed, and a cell over MAX_FIELD_CHARS is refused (the message does not repeat it). The flag says
    hidden text was found."""
    clean: dict[str, str] = {}
    hidden = False
    for key, value in row.items():
        text = "" if value is None else value
        if not isinstance(text, str):
            continue  # cells beyond the header: csv.DictReader keeps them as a list under the key None
        if len(text) > MAX_FIELD_CHARS:
            raise ValueError(f"A {record_type} row has a field longer than {MAX_FIELD_CHARS} characters")
        text, found = strip_hidden_and_control(text)
        hidden = hidden or found
        clean[key] = text
    return clean, hidden


def plain_number(value: object) -> str | None:
    """"$1,250.50" -> "1250.50"; None for blank text or anything that is not a plain decimal number (n/a, NaN, 1e999)."""
    if not isinstance(value, str):
        return None
    cleaned = value.strip().replace("$", "").strip()
    return cleaned.replace(",", "") if _PLAIN_NUMBER.fullmatch(cleaned) else None


def _revision_number(chunk: Chunk) -> int:
    try:
        return int(chunk.metadata.get("revision", "0"))
    except ValueError:
        return 0


def mark_latest_revisions(chunks: list[Chunk]) -> list[Chunk]:
    """A resubmittal keeps its number and raises the revision, so S-001 has one chunk per
    revision. Mark the older ones as superseded so search does not present them as current, and say in the newest
    one's text that it is current: the text is the evidence the answer verifier reads."""
    newest: dict[str, int] = {}
    for chunk in chunks:
        if chunk.record_type == "submittal":
            newest[chunk.record_id] = max(newest.get(chunk.record_id, 0), _revision_number(chunk))
    marked: list[Chunk] = []
    for chunk in chunks:
        if chunk.record_type == "submittal" and _revision_number(chunk) < newest[chunk.record_id]:
            chunk = Chunk(
                chunk.record_type,
                chunk.record_id,
                chunk.text + " This revision is superseded by a later revision.",
                {**chunk.metadata, "is_latest": "no"},
                chunk.relations,
            )
        elif chunk.record_type == "submittal":
            chunk = Chunk(chunk.record_type, chunk.record_id, chunk.text + " This is the current revision.",
                          chunk.metadata, chunk.relations)
        marked.append(chunk)
    return marked


def row_to_chunk(record_type: str, row: Mapping[Any, object]) -> Chunk:
    if record_type not in _TEXT_FUNCTIONS:
        raise ValueError(f"Unknown record type: {record_type!r}")
    row, hidden = _clean_row(record_type, row)
    try:
        return _build_chunk(record_type, row, hidden)
    except KeyError as missing:  # a required column is not in the file's header
        raise ValueError(f"A {record_type} row has no {missing.args[0]!r} column") from None


def _build_chunk(record_type: str, row: dict[str, str], hidden: bool) -> Chunk:
    metadata = {"record_type": record_type, "project_id": row.get("project_id", ""), "is_latest": "yes"}
    if record_type == "submittal":
        metadata["revision"] = row.get("revision", "0")
    for field in _FILTER_FIELDS:
        if row.get(field, "").strip():
            metadata[field] = row[field].strip()
    for field in NUMERIC_FIELDS.get(record_type, ()):
        number = plain_number(row.get(field))
        if number is not None:
            metadata[field] = number
    record_id = record_id_of(record_type, row)
    text = _TEXT_FUNCTIONS[record_type](row)
    suspicious = looks_like_injection(text)
    if hidden:
        suspicious.append("hidden_characters")  # the characters were removed above, so the scan of the text cannot see them
    if suspicious:
        # The record is kept (it may be real project data), but it is flagged for a human to look at.
        # Only the record id and the names of the matching heuristics are logged, never the text.
        metadata["injection_suspected"] = "yes"
        log_event("suspicious_record", level=logging.WARNING, record_id=record_id,
                  record_type=record_type, patterns=suspicious)
    return Chunk(record_type=record_type, record_id=record_id, text=text, metadata=metadata,
                 relations=extract_relations(record_type, row))
