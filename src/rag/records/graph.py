"""KNOWLEDGE GRAPH: how records connect to each other and to the things they are about.

A node is either a RECORD (an RFI, a submittal, a punch item ...) or an ENTITY that records mention: a
trade, a room, a spec section, a drawing sheet, a change order, a party holding the ball. An edge is
typed and points from a record to what it mentions:

    PL-001 --located_in--> room 102          PL-001 --responsible_party--> sample painting
    S-010  --responsible_party--> sample painting
    CE-001 --related_rfi--> RFI-003  (record to record)       CE-001 --issued_as--> CO-001

Two punch items are therefore connected through a shared trade or room even though their text never
mentions each other, which is what multi-hop questions need ("what else is Sample Painting working on?").

This file is pure: it turns chunks into nodes and edges, and does no I/O. The PostgreSQL store saves the
result in its tables; the in-memory store walks it directly.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag.records.chunker import Chunk

RECORD_KINDS = ("rfi", "submittal", "change_event", "daily_report", "punch_item")
ENTITY_KINDS = ("trade", "room", "spec_section", "drawing_sheet", "change_order", "party")

# "Everything the Architect is holding" is not a useful neighbour of one RFI, and it would flood the answer.
EXPANSION_EXCLUDED_RELATIONS = frozenset({"ball_in_court"})

Relation = tuple[str, str, str]  # (relation, entity kind, name as written)

# An entity shared by more records than this is not a useful neighbour ("everything on site", "the whole
# project"): listing it would flood the answer, and walking it gets slower with its size.
HUB_DEGREE = 500

# Drawing sheets: A-201, S-103. Not CE-001, PL-007, CO-001 or DR-2026: those fit the shape but name records.
_SHEET = re.compile(r"\b(?!(?:CE|CO|PL|DR)-)[A-Za-z]{1,2}-\d{2,4}\b", re.IGNORECASE)
_SPEC_IN_TEXT = re.compile(r"\b(\d{2})[ -](\d{2})[ -](\d{2})\b")  # in free text a separator is required (09 21 16, 09-21-16)
_SPEC_IN_FIELD = re.compile(r"(\d{2})[ -]?(\d{2})[ -]?(\d{2})")  # in a spec-section column 092116 is fine too


def clean_name(value: str) -> str:
    """Collapse runs of spaces and tabs: 'Sample   Painting ' -> 'Sample Painting'."""
    return " ".join(value.split())


def entity_key(name: str) -> str:
    """What identifies an entity: its name ignoring case and spacing, so 'sample  PAINTING' is the same trade."""
    return clean_name(name).lower()


def normalise_spec(value: str) -> str | None:
    """A spec section written 092116, 09-21-16 or '09 21 16 - Gypsum Board' -> '09 21 16'. None if there is none."""
    match = _SPEC_IN_FIELD.search(clean_name(value))
    return " ".join(match.groups()) if match else None


def order_sharing(records: list[tuple[str, str]], seed_kind: str) -> list[tuple[str, str]]:
    """Which of the records sharing an entity to show first when only a few fit: records of a DIFFERENT type
    from the seed (an RFI is more informative next to a submittal than next to more RFIs), then the newest id
    first. `records` is a list of (type, id). The PostgreSQL store applies the same rule in SQL."""
    newest_first = sorted(records, key=lambda r: r[1], reverse=True)
    return sorted(newest_first, key=lambda r: r[0] == seed_kind)  # stable: False (other types) sorts before True


def _one(relation: str, kind: str, value: str) -> list[Relation]:
    name = clean_name(value)
    return [(relation, kind, name)] if name else []


def extract_relations(record_type: str, row: dict[str, str]) -> tuple[Relation, ...]:
    """The entities one CSV row mentions. (RFI-to-change-event links are handled in build_graph.)"""
    found: list[Relation] = []
    if record_type == "rfi":
        references = row.get("references", "")
        found += [("references_sheet", "drawing_sheet", sheet.upper()) for sheet in _SHEET.findall(references)]
        found += [("references_spec", "spec_section", " ".join(spec)) for spec in _SPEC_IN_TEXT.findall(references)]
        found += _one("ball_in_court", "party", row.get("ball_in_court", ""))
    elif record_type == "submittal":
        spec = row.get("spec_section", "")
        found += _one("in_spec_section", "spec_section", normalise_spec(spec) or spec)
        found += _one("responsible_party", "trade", row.get("responsible_party", ""))
        found += _one("ball_in_court", "party", row.get("ball_in_court", ""))
    elif record_type == "punch_item":
        found += _one("located_in", "room", row.get("location", ""))
        found += _one("responsible_party", "trade", row.get("responsible_party", ""))
    elif record_type == "change_event":
        found += _one("issued_as", "change_order", row.get("change_order_number", ""))
    return tuple(dict.fromkeys(found))  # no repeats, order kept


@dataclass(frozen=True)
class Node:
    kind: str
    key: str  # a record id (RFI-003), or an entity's normalised name (sample painting)
    label: str  # for people to read: the record id, or the entity's name as first written


@dataclass(frozen=True)
class Edge:
    src_kind: str
    src_key: str
    relation: str
    dst_kind: str
    dst_key: str


@dataclass(frozen=True)
class Graph:
    nodes: list[Node]
    edges: list[Edge]


def build_graph(chunks: list[Chunk]) -> Graph:
    """Every record is a node, but only its CURRENT revision says what it is connected to. An old revision that
    named a different trade must not keep connecting that trade to the record."""
    nodes: dict[tuple[str, str], Node] = {}
    current = [chunk for chunk in chunks if chunk.metadata.get("is_latest", "yes") != "no"]
    for chunk in chunks:
        nodes.setdefault((chunk.record_type, chunk.record_id), Node(chunk.record_type, chunk.record_id, chunk.record_id))
    for chunk in current:
        for _, kind, name in chunk.relations:
            nodes.setdefault((kind, entity_key(name)), Node(kind, entity_key(name), clean_name(name)))

    rfi_ids = {chunk.record_id.upper(): chunk.record_id for chunk in chunks if chunk.record_type == "rfi"}
    edges: set[Edge] = set()
    for chunk in current:
        related = chunk.metadata.get("related_rfi", "").strip().upper()
        if chunk.record_type == "change_event" and related in rfi_ids:  # only when the RFI really exists
            edges.add(Edge(chunk.record_type, chunk.record_id, "related_rfi", "rfi", rfi_ids[related]))
        for relation, kind, name in chunk.relations:
            edges.add(Edge(chunk.record_type, chunk.record_id, relation, kind, entity_key(name)))
    return Graph(sorted(nodes.values(), key=lambda n: (n.kind, n.key)),
                 sorted(edges, key=lambda e: (e.src_kind, e.src_key, e.relation, e.dst_kind, e.dst_key)))
