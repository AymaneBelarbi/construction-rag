"""INDEX, step 2: hold chunks, their vectors, and the links between records.

A plain in-memory index saved to one JSON file: the offline stand-in (`--store memory`, tests, demos). The
production store is PostgreSQL (pgstore.py); both implement store_api.Store and pass the same contract tests.

Idempotent re-ingest: the cache stores a fingerprint (a SHA-256 hash) of all
chunk texts plus the embedder's name. If both match, saved vectors are reused
and no embedding call is made. Changing a CSV, the chunker, or the embedding
model changes the fingerprint and triggers a rebuild.
"""
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from pathlib import Path

from rag.providers.embed import Embedder, tokenize, tokenize_query
from rag.records.chunker import RECORD_TYPES, Chunk, mark_latest_revisions, plain_number, row_to_chunk
from rag.records.graph import (
    ENTITY_KINDS,
    EXPANSION_EXCLUDED_RELATIONS,
    HUB_DEGREE,
    RECORD_KINDS,
    Graph,
    build_graph,
    order_sharing,
)
from rag.records.loader import load_csv
from rag.records.snapshot import capture, materialize
from rag.storage.bm25 import Bm25
from rag.storage.store_api import (
    Aggregate,
    Filters,
    Hit,
    Related,
    RelatedRecords,
    StoreError,
    check_aggregate_request,
    chunk_matches,
    summarize,
)

RECORD_FILES = {
    "rfi": "rfis.csv",
    "submittal": "submittals.csv",
    "change_event": "change_events.csv",
    "daily_report": "daily_reports.csv",
    "punch_item": "punch_items.csv",
}


@dataclass
class Index:
    chunks: list[Chunk]
    vectors: list[list[float]]  # vectors[i] belongs to chunks[i]
    links: dict[str, list[str]]  # record ID -> IDs of directly related records
    embedder_id: str
    fingerprint: str
    source_files: dict[str, str] | None = field(default=None, repr=False, compare=False)
    # Built once from the chunks. The chunk's own ID is added to its keyword text a second time, which
    # boosts a record when someone searches for its ID over records that merely mention it.
    _bm25: Bm25 = field(init=False, repr=False, compare=False)

    _graph: Graph = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._bm25 = Bm25([tokenize(f"{c.record_id} {c.text}") for c in self.chunks])
        self._graph = build_graph(self.chunks)

    def count(self) -> int:
        return len(self.chunks)

    def list_records(self, filters: Filters, limit: int) -> list[Chunk]:
        if limit < 1:
            raise ValueError("limit must be positive")
        matches = [chunk for chunk in self.chunks if chunk.metadata.get("is_latest", "yes") == "yes"
                   and chunk_matches(chunk, filters)]
        return sorted(matches, key=lambda chunk: chunk.record_id)[:limit + 1]

    def aggregate(self, record_type: str, field: str, operation: str, filters: Filters | None = None) -> Aggregate:
        check_aggregate_request(record_type, field, operation)
        values: dict[str, Decimal] = {}
        for chunk in self.chunks:
            if chunk.record_type != record_type or chunk.metadata.get("is_latest", "yes") != "yes" or not chunk_matches(chunk, filters):
                continue
            number = plain_number(chunk.metadata.get(field))
            if number is not None:
                values[chunk.record_id] = Decimal(number)
        return summarize(record_type, field, operation, values)

    def vector_search(self, vector: list[float], limit: int, filters: Filters | None = None) -> list[Hit]:
        # Vectors are unit length, so the dot product is the cosine similarity.
        scored = [
            Hit(chunk, sum(a * b for a, b in zip(vector, stored, strict=True)))
            for chunk, stored in zip(self.chunks, self.vectors, strict=True)
            if chunk_matches(chunk, filters)
        ]
        return sorted(scored, key=lambda hit: -hit.score)[:limit]

    def keyword_search(self, query: str, limit: int, filters: Filters | None = None) -> list[Hit]:
        scores = self._bm25.scores(tokenize_query(query))
        scored = [Hit(chunk, score) for chunk, score in zip(self.chunks, scores, strict=True)
                  if score > 0 and chunk_matches(chunk, filters)]
        return sorted(scored, key=lambda hit: -hit.score)[:limit]

    def related(self, record_id: str, limit: int = 15, per_entity: int = 5) -> RelatedRecords:
        """Records connected to this one: direct links first, then records sharing a trade, room, spec section
        or drawing sheet with it. Each record appears once, with the reasons joined. See graph.py."""
        wanted = record_id.strip().upper()
        edges = self._graph.edges
        labels = {(n.kind, n.key): n.label for n in self._graph.nodes}
        seeds = sorted((n.kind, n.key) for n in self._graph.nodes if n.kind in RECORD_KINDS and n.key.upper() == wanted)
        if not seeds:
            return RelatedRecords()
        seed_kind, seed_key = seeds[0]

        reasons: dict[str, list[str]] = {}
        omitted: dict[str, int] = {}
        direct: list[tuple[str, str]] = []  # (other record id, relation)
        for edge in edges:
            if edge.src_kind in RECORD_KINDS and edge.dst_kind in RECORD_KINDS:
                if edge.src_key == seed_key and edge.src_kind == seed_kind:
                    direct.append((edge.dst_key, edge.relation))
                elif edge.dst_key == seed_key and edge.dst_kind == seed_kind:
                    direct.append((edge.src_key, edge.relation))
        for other, relation in sorted(direct):
            reasons.setdefault(other, []).append(relation)

        mentioned = sorted({(e.relation, e.dst_kind, e.dst_key) for e in edges
                            if e.src_kind == seed_kind and e.src_key == seed_key and e.dst_kind in ENTITY_KINDS
                            and e.relation not in EXPANSION_EXCLUDED_RELATIONS})
        seen_entities: set[tuple[str, str]] = set()
        for _, kind, key in mentioned:
            if (kind, key) in seen_entities:
                continue
            seen_entities.add((kind, key))
            reason = f"shares {kind}: {labels[(kind, key)]}"
            sharing = sorted({(e.src_kind, e.src_key) for e in edges
                              if e.dst_kind == kind and e.dst_key == key and e.src_kind in RECORD_KINDS
                              and (e.src_kind, e.src_key) != (seed_kind, seed_key)})
            if len(sharing) > HUB_DEGREE:
                omitted[reason] = len(sharing)  # too many to be informative: say so instead of listing
                continue
            ordered = order_sharing(sharing, seed_kind)
            for _, other in ordered[:per_entity]:
                reasons.setdefault(other, []).append(reason)
            if len(ordered) > per_entity:
                omitted[reason] = len(ordered) - per_entity

        latest = {c.record_id: c for c in self.chunks if c.metadata.get("is_latest", "yes") == "yes"}
        items = [Related(latest[other], "; ".join(dict.fromkeys(why)))
                 for other, why in reasons.items() if other in latest]
        if len(items) > limit:
            omitted["beyond the overall limit"] = len(items) - limit
        return RelatedRecords(items[:limit], omitted)

    def get(self, record_id: str) -> list[Chunk]:
        """All chunks with this ID. A submittal has one per revision."""
        wanted = record_id.strip().upper()
        return [c for c in self.chunks if c.record_id.upper() == wanted]

    def linked(self, record_id: str) -> list[Chunk]:
        found: list[Chunk] = []
        for other_id in self.links.get(record_id.strip().upper(), []):
            found.extend(self.get(other_id))
        return found


def load_chunks(folder: Path) -> list[Chunk]:
    """Read every known CSV in the project folder. One missing CSV is skipped, but a missing
    folder or a folder with no records is an error: a typo must not become an empty index."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Project folder not found: {folder}")
    chunks: list[Chunk] = []
    for record_type in RECORD_TYPES:
        path = folder / RECORD_FILES[record_type]
        if path.exists():
            chunks.extend(row_to_chunk(record_type, row) for row in load_csv(path))
    if not chunks:
        raise ValueError(f"No records found in {folder}")
    return mark_latest_revisions(chunks)


def chunks_fingerprint(chunks: list[Chunk]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        metadata = json.dumps(chunk.metadata, sort_keys=True)
        relations = json.dumps(chunk.relations)
        digest.update(f"{chunk.record_id}\n{chunk.text}\n{metadata}\n{relations}\n".encode())
    return digest.hexdigest()


def build_links(chunks: list[Chunk]) -> dict[str, list[str]]:
    """Links come from explicit fields, so a dict is enough (no graph database).
    Today: change_events.related_rfi connects a change event and its RFI, both ways."""
    links: dict[str, list[str]] = {}

    def connect(a: str, b: str) -> None:
        for source, target in ((a, b), (b, a)):
            if target not in links.setdefault(source, []):
                links[source].append(target)

    for chunk in chunks:
        related = chunk.metadata.get("related_rfi")
        if chunk.record_type == "change_event" and related:
            connect(chunk.record_id.upper(), related.upper())
    return links


def make_index(chunks: list[Chunk], embed: Embedder, embedder_id: str) -> Index:
    seen: set[tuple[str, str]] = set()
    for chunk in chunks:  # the same rule as the database, which keeps one row per record and revision
        if chunk.key in seen:
            raise StoreError(f"Two chunks share record {chunk.record_id} revision {chunk.key[1]}")
        seen.add(chunk.key)
    vectors = embed([c.text for c in chunks]) if chunks else []
    if len(vectors) != len(chunks):
        raise ValueError(f"Embedder returned {len(vectors)} vectors for {len(chunks)} chunks")
    return Index(chunks, vectors, build_links(chunks), embedder_id, chunks_fingerprint(chunks))


def build_index(folder: Path, embed: Embedder, embedder_id: str) -> Index:
    return make_index(load_chunks(folder), embed, embedder_id)


def save_index(index: Index, path: Path) -> None:
    """Write the cache so a crash cannot leave a half-written file: write to a temporary file in
    the same folder, then swap it in with os.replace, which is atomic (a reader sees either the
    old file or the new one, never a mix). This does not promise the new file survives a power
    cut, which is fine for a cache that can be rebuilt. The file is readable by its owner only,
    because mkstemp creates it that way."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({
        "embedder_id": index.embedder_id,
        "fingerprint": index.fingerprint,
        "chunks": [asdict(c) for c in index.chunks],
        "vectors": index.vectors,
    })
    # The prefix names the cache file, so a leftover from a killed process is easy to recognise.
    handle, temp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as temp_file:
            temp_file.write(payload)
            temp_file.flush()
            os.fsync(temp_file.fileno())  # make sure the bytes reached the disk before the swap
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def load_index(path: Path) -> Index:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    # JSON turns the relation tuples into lists; turn them back.
    chunks = [Chunk(**{**c, "relations": tuple(tuple(r) for r in c.get("relations", ()))}) for c in data["chunks"]]
    if len(data["vectors"]) != len(chunks):
        raise ValueError("Cache is damaged: vector count does not match chunk count")
    # Links are cheap to compute, so they are rebuilt instead of saved: they can never go stale.
    return Index(chunks, data["vectors"], build_links(chunks), data["embedder_id"], data["fingerprint"])


def load_or_build(folder: Path, embed: Embedder, embedder_id: str, cache_path: Path) -> Index:
    files = capture(Path(folder))
    with materialize(files) as snapshot:
        chunks = load_chunks(snapshot)
    fingerprint = chunks_fingerprint(chunks)
    if Path(cache_path).exists():
        try:
            cached = load_index(cache_path)
            if cached.fingerprint == fingerprint and cached.embedder_id == embedder_id:
                cached.source_files = files
                return cached
        except (ValueError, KeyError, TypeError):  # ValueError also covers bad JSON and bad UTF-8
            pass  # damaged cache: fall through and rebuild it
    index = make_index(chunks, embed, embedder_id)
    index.source_files = files
    save_index(index, cache_path)
    return index
