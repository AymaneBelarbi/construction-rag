"""STORE API: what the rest of the program needs from a place that holds chunks.

Two implementations exist: the in-memory Index (store.py, used for tests and offline demos) and
PostgresStore (pgstore.py, the production store). Code that searches (retrieve.py) or answers (agent.py)
only talks to this interface, so it works with either, and tests/storage/test_store_contract.py runs the same
checks against both.

A store offers two ranked searches (by vector and by keyword), exact lookups, the graph walk, and exact aggregates.
Merging the searches, pinning exact IDs and amounts and the score cut-off live in retrieve.py, once, for every store.
"""
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Protocol

from rag.records.chunker import NUMERIC_FIELDS, Chunk

Filters = Mapping[str, str]  # metadata key -> wanted value (compared ignoring case)


class StoreError(Exception):
    """The store cannot do what was asked (not set up, not ingested, wrong vector size ...)."""


@dataclass
class Hit:
    chunk: Chunk
    score: float  # only the order matters; the scale depends on the search that produced it


@dataclass
class Related:
    chunk: Chunk
    via: str  # why it is connected: 'related_rfi', or 'shares trade: Sample Painting' (several reasons joined with '; ')


class RelatedRecords(list):
    """The result of Store.related: a list of Related, plus what was left out and why, for example
    {'shares trade: Sample Painting': 3, 'beyond the overall limit': 2}. The agent shows this to the model so
    it knows the list is not complete."""

    def __init__(self, items: list[Related] | tuple = (), omitted: dict[str, int] | None = None) -> None:
        super().__init__(items)
        self.omitted: dict[str, int] = dict(omitted or {})


AGGREGATE_OPERATIONS = ("max", "min", "sum", "avg", "count")
MAX_AGGREGATE_IDS = 10  # how many records that reach a maximum or minimum are named; the rest are only counted
_CENTS = Decimal("0.01")


@dataclass(frozen=True)
class Aggregate:
    """An exact figure computed by the store over the current revision of every matching record that has the field."""

    record_type: str
    field: str
    operation: str
    value: Decimal | None  # None when no record has a value (a count is then 0, not None)
    count: int  # how many records had a value
    record_ids: tuple[str, ...] = ()  # max and min only: the records that reach the value, in id order
    more_ids: int = 0  # ... and how many further ones there are beyond MAX_AGGREGATE_IDS


def check_aggregate_request(record_type: str, field: str, operation: str) -> None:
    """Refuse anything but the fixed list of numeric fields and operations, before a query is built. The model
    chooses these three words, so they are only ever compared with this list, never put into SQL."""
    if record_type not in NUMERIC_FIELDS:
        raise ValueError(f"record_type must be one of: {', '.join(NUMERIC_FIELDS)}")
    if field not in NUMERIC_FIELDS[record_type]:
        raise ValueError(f"{record_type} records have these numeric fields: {', '.join(NUMERIC_FIELDS[record_type])}")
    if operation not in AGGREGATE_OPERATIONS:
        raise ValueError(f"operation must be one of: {', '.join(AGGREGATE_OPERATIONS)}")


def summarize(record_type: str, field: str, operation: str, values: Mapping[str, Decimal]) -> Aggregate:
    """The memory store's calculation: record id -> number in, the Aggregate out. (PostgreSQL does the same in SQL.)"""
    if not values:
        return Aggregate(record_type, field, operation, Decimal(0) if operation == "count" else None, 0)
    numbers = list(values.values())
    ids: tuple[str, ...] = ()
    more = 0
    if operation in ("max", "min"):
        result = max(numbers) if operation == "max" else min(numbers)
        reaching = sorted(record_id for record_id, number in values.items() if number == result)
        ids, more = tuple(reaching[:MAX_AGGREGATE_IDS]), max(0, len(reaching) - MAX_AGGREGATE_IDS)
    elif operation == "sum":
        result = sum(numbers, Decimal(0))
    elif operation == "avg":
        result = (sum(numbers, Decimal(0)) / len(numbers)).quantize(_CENTS, rounding=ROUND_HALF_UP)
    else:
        result = Decimal(len(numbers))
    return Aggregate(record_type, field, operation, result, len(numbers), ids, more)


def chunk_matches(chunk: Chunk, filters: Filters | None) -> bool:
    """True when every filter equals the chunk's metadata value, ignoring case."""
    if not filters:
        return True
    return all(chunk.metadata.get(key, "").lower() == str(wanted).lower() for key, wanted in filters.items())


class Store(Protocol):
    def count(self) -> int:
        """How many chunks are stored."""

    def get(self, record_id: str) -> list[Chunk]:
        """Every chunk with this record id (a submittal has one per revision), oldest revision first."""

    def list_records(self, filters: Filters, limit: int) -> list[Chunk]:
        """Current records matching exact filters, sorted by ID, up to limit plus one to detect omissions."""

    def linked(self, record_id: str) -> list[Chunk]:
        """Chunks of the records directly connected to this one in the graph, in either direction."""

    def related(self, record_id: str, limit: int = 15, per_entity: int = 5) -> RelatedRecords:
        """Records connected to this one in the knowledge graph (latest revisions only): direct links first, then
        records sharing an entity (trade, room, spec section, drawing sheet) with it. At most per_entity per entity
        (other record types first, then the newest ids) and limit overall; an entity shared by more than
        graph.HUB_DEGREE records is not listed. Each record appears once. What was left out is in `.omitted`."""

    def aggregate(self, record_type: str, field: str, operation: str, filters: Filters | None = None) -> Aggregate:
        """max, min, sum, avg or count of a numeric field (chunker.NUMERIC_FIELDS) over the CURRENT revision of every
        record of that type that matches the filters and has a value. Exact (decimal) arithmetic; the average is
        rounded to cents. For max and min the records that reach the value are named. Raises ValueError for an
        unknown record type, field or operation."""

    def vector_search(self, vector: list[float], limit: int, filters: Filters | None = None) -> list[Hit]:
        """Nearest chunks by cosine similarity (score = similarity, best first)."""

    def keyword_search(self, query: str, limit: int, filters: Filters | None = None) -> list[Hit]:
        """Chunks sharing words with the query (best first). Chunks with no shared word are not returned."""
