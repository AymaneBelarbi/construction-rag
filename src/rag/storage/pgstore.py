"""POSTGRES STORE: the production store. Chunks, vectors and the graph live in PostgreSQL with pgvector.

Reading (the Store interface from store_api.py):
- vector_search: cosine similarity with pgvector's HNSW index.
- keyword_search: PostgreSQL full-text search, ranked by the rarity of the matched words (PostgreSQL's own
  ranking cannot tell a rare word from a common one). IDs and dates are split at hyphens when indexed and
  searched as exact phrases, so "RFI-003" matches RFI-003 and not a record that merely mentions
  RFI-011 and CE-003.
- get / linked: exact lookups, and neighbours in the graph.

Writing (sync): make the database match a list of chunks, doing as little work as possible. A chunk is
re-embedded only if its text changed, its embedding model changed, or its vector is missing. The write is
one transaction, so a failure leaves the database as it was. All SQL uses parameters; nothing the user or
the model types is ever pasted into SQL text.
"""
import hashlib
import json
import math
import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

import psycopg

from rag.db.connection import fetch_one
from rag.infra.telemetry import log_event
from rag.providers.embed import STOPWORDS, Embedder
from rag.records.chunker import Chunk
from rag.records.graph import ENTITY_KINDS, EXPANSION_EXCLUDED_RELATIONS, HUB_DEGREE, RECORD_KINDS, Graph, build_graph
from rag.records.snapshot import materialize, version
from rag.storage.store import chunks_fingerprint, load_chunks
from rag.storage.store_api import (
    MAX_AGGREGATE_IDS,
    Aggregate,
    Filters,
    Hit,
    Related,
    RelatedRecords,
    StoreError,
    check_aggregate_request,
)

# A first load of thousands of vectors is dominated by updating the HNSW index one row at a time (measured: 5 s
# for 3,000 rows with the index, 0.8 s without). From this many new or re-embedded rows on, sync drops the index,
# inserts, and rebuilds it once, inside the same transaction. Readers wait for that transaction, which is the
# right trade for an ingest of this size and is never the case for the small updates of normal use.
BULK_LOAD_THRESHOLD = 5000
BULK_INDEX_MEMORY_MB = 512  # memory for the index rebuild; the database server needs at least this much shared memory (shm_size)
VECTOR_INDEX = "chunks_embedding_hnsw"
# Must stay identical to the index in migration 0001; a test compares the two definitions.
_CREATE_VECTOR_INDEX = f"CREATE INDEX {VECTOR_INDEX} ON chunks USING hnsw (embedding vector_cosine_ops)"

_COLUMNS = "record_id, record_type, text, metadata"
_TOKEN = re.compile(r"[^\W_]+(?:-[^\W_]+)*")  # letters and digits in any language; hyphens join the parts of an ID
MAX_QUERY_TERMS = 32
def vector_literal(vector: list[float]) -> str:
    """pgvector reads a vector written as text: [0.1,0.2,...]. repr() keeps every digit of each float."""
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def revision_of(chunk: Chunk) -> int:
    try:
        return int(chunk.metadata.get("revision", "0"))
    except ValueError:
        return 0


def build_terms(query: str) -> list[str]:
    """The searchable pieces of a question, as PostgreSQL text-search terms (empty if nothing is left).

    A plain word is one term. A hyphenated token such as RFI-003 or 2026-08-03 becomes an exact phrase
    (rfi <-> 003): the parts must sit next to each other. Only letters, digits and the operators we add
    ever reach the database, so the query text cannot inject anything. A lone letter is not a word
    ("contractor's" would leave an 's' that matches every S-xxx record) except inside an ID."""
    terms: list[str] = []
    for token in _TOKEN.findall(query.lower()):
        parts = token.split("-")
        if len(parts) > 1:
            term = "(" + " <-> ".join(parts) + ")"
        elif token in STOPWORDS or (len(token) == 1 and token.isalpha()):
            continue
        else:
            term = token
        if term not in terms:
            terms.append(term)
    return terms[:MAX_QUERY_TERMS]


def build_tsquery(query: str) -> str | None:
    """All the terms OR-ed into one text-search query, or None if there are none."""
    return " | ".join(build_terms(query)) or None


def idf(documents_with_term: int, documents: int) -> float:
    """How much a term is worth: high when few records contain it, near zero when almost all do (the same
    inverse-document-frequency formula BM25 uses). PostgreSQL's own ranking has no such notion, so without
    this a query of 'abandoned conduit reading room ceiling' ranks a record with three common words above
    the one with the two rare ones."""
    return math.log(1 + (documents - documents_with_term + 0.5) / (documents_with_term + 0.5))


def _filter_sql(filters: Filters | None) -> tuple[str, list[str]]:
    """' AND ...' conditions and their parameters, for metadata equality ignoring case."""
    clauses: list[str] = []
    params: list[str] = []
    for key, value in (filters or {}).items():
        clauses.append(" AND lower(metadata ->> %s) = lower(%s)")
        params += [str(key), str(value)]
    return "".join(clauses), params


@dataclass(frozen=True)
class SyncReport:
    inserted: int
    updated: int
    unchanged: int
    deleted: int
    embedded: int  # how many texts were sent to the embedding model


@dataclass(frozen=True)
class _Stored:
    text_hash: str
    embedder_id: str | None
    has_embedding: bool
    metadata: dict[str, str]


class PostgresStore:
    def __init__(self, conn: psycopg.Connection, project_slug: str, bulk_threshold: int = BULK_LOAD_THRESHOLD) -> None:
        self.conn = conn
        self.project_slug = project_slug
        self.bulk_threshold = bulk_threshold
        self._project_id: int | None = None
        self.embedding_dim = self._read_embedding_dim()
        # Without these, an HNSW search with a filter quietly returns too few rows: the index finds its 40
        # nearest neighbours overall and the filter then throws most of them away. Iterative scan keeps
        # looking until the LIMIT is met; ef_search 100 widens the search for better recall.
        conn.execute("SET hnsw.ef_search = 100")
        conn.execute("SET hnsw.iterative_scan = relaxed_order")

    # ---- setup ---------------------------------------------------------------------------------

    def _read_embedding_dim(self) -> int:
        try:
            row = self.conn.execute("SELECT value FROM rag_settings WHERE key = 'embedding_dim'").fetchone()
        except psycopg.errors.UndefinedTable:
            raise StoreError("The database has no tables yet. Run: python -m rag.db.migrate") from None
        if row is None:
            raise StoreError("The database is missing its embedding_dim setting. Run: python -m rag.db.migrate")
        return int(row[0])

    def _find_project_id(self) -> int | None:
        row = self.conn.execute("SELECT id FROM projects WHERE slug = %s", (self.project_slug,)).fetchone()
        return row[0] if row else None

    def _require_project_id(self) -> int:
        if self._project_id is None:
            self._project_id = self._find_project_id()
        if self._project_id is None:
            raise StoreError(f"Project {self.project_slug!r} has not been ingested. Run: python -m rag.cli.ingest")
        return self._project_id

    def embedder_ids(self) -> set[str]:
        """Which embedding models made the stored vectors (normally exactly one)."""
        rows = self.conn.execute(
            "SELECT DISTINCT embedder_id FROM chunks WHERE project_id = %s AND embedder_id IS NOT NULL",
            (self._require_project_id(),),
        )
        return {row[0] for row in rows}

    def source_snapshot(self) -> tuple[str, dict[str, str]] | None:
        try:
            row = self.conn.execute("SELECT version, files FROM project_source_snapshots WHERE project_id = %s",
                                    (self._require_project_id(),)).fetchone()
        except psycopg.errors.UndefinedTable:
            raise StoreError("Project snapshots are not installed. Run the database migration and re-ingest.") from None
        if row is None:
            return None
        files = row[1]
        if not isinstance(files, dict) or version(files) != row[0]:
            raise StoreError("The saved project snapshot is damaged. Re-ingest the project.")
        return row[0], files

    def source_generation(self) -> int | None:
        """Monotonic identity of the currently committed project snapshot, or None if unavailable."""
        try:
            row = self.conn.execute("SELECT generation FROM project_source_snapshots WHERE project_id = %s",
                                    (self._require_project_id(),)).fetchone()
        except (psycopg.errors.UndefinedTable, psycopg.errors.UndefinedColumn):
            raise StoreError("Project snapshot generations are not installed. Run the database migration.") from None
        return int(row[0]) if row else None

    # ---- reading -------------------------------------------------------------------------------

    @staticmethod
    def _chunk(row: tuple) -> Chunk:
        record_id, record_type, text, metadata = row[:4]
        return Chunk(record_type, record_id, text, metadata)

    def count(self) -> int:
        (count,) = fetch_one(self.conn, "SELECT count(*) FROM chunks WHERE project_id = %s", (self._require_project_id(),))
        return count

    def list_records(self, filters: Filters, limit: int) -> list[Chunk]:
        if limit < 1:
            raise ValueError("limit must be positive")
        filter_sql, params = _filter_sql(filters)
        rows = self.conn.execute(
            f"SELECT {_COLUMNS} FROM chunks WHERE project_id = %s "
            "AND coalesce(metadata ->> 'is_latest', 'yes') = 'yes'" + filter_sql +
            ' ORDER BY record_id COLLATE "C", revision LIMIT %s',
            (self._require_project_id(), *params, limit + 1),
        )
        return [self._chunk(row) for row in rows]

    def get(self, record_id: str) -> list[Chunk]:
        rows = self.conn.execute(
            f"SELECT {_COLUMNS} FROM chunks WHERE project_id = %s AND upper(record_id) = upper(%s) ORDER BY revision",
            (self._require_project_id(), record_id),
        )
        return [self._chunk(row) for row in rows]

    def linked(self, record_id: str) -> list[Chunk]:
        project_id = self._require_project_id()
        rows = self.conn.execute(
            f"""
            SELECT {', '.join('c.' + col.strip() for col in _COLUMNS.split(','))}
            FROM chunks c
            JOIN nodes neighbour ON neighbour.project_id = c.project_id AND neighbour.key = c.record_id
                                AND neighbour.kind = c.record_type
            WHERE c.project_id = %s AND neighbour.id IN (
                SELECT e.dst FROM edges e JOIN nodes n ON n.id = e.src
                 WHERE n.project_id = %s AND upper(n.key) = upper(%s)
                UNION
                SELECT e.src FROM edges e JOIN nodes n ON n.id = e.dst
                 WHERE n.project_id = %s AND upper(n.key) = upper(%s)
            )
            ORDER BY c.record_id COLLATE "C", c.revision
            """,
            (project_id, project_id, record_id, project_id, record_id),
        )
        return [self._chunk(row) for row in rows]

    def related(self, record_id: str, limit: int = 15, per_entity: int = 5) -> RelatedRecords:
        """Direct links first, then records sharing a trade, room, spec section or drawing sheet with this one.
        The graph is walked with a few small indexed queries: the seed's links, then one bounded query per entity
        the seed mentions (never a join across a whole entity's members). The ordering rule is graph.order_sharing."""
        project_id = self._require_project_id()
        record_kinds = list(RECORD_KINDS)
        seed = self.conn.execute(
            "SELECT id, kind FROM nodes WHERE project_id = %s AND kind = ANY(%s) AND upper(key) = upper(%s) "
            "ORDER BY kind LIMIT 1", (project_id, record_kinds, record_id)).fetchone()
        if seed is None:
            return RelatedRecords()
        seed_id, seed_kind = seed

        direct = self.conn.execute(
            """
            SELECT n.key, e.relation FROM edges e JOIN nodes n ON n.id = e.dst
             WHERE e.src = %(seed)s AND n.kind = ANY(%(kinds)s)
            UNION ALL
            SELECT n.key, e.relation FROM edges e JOIN nodes n ON n.id = e.src
             WHERE e.dst = %(seed)s AND n.kind = ANY(%(kinds)s)
            """, {"seed": seed_id, "kinds": record_kinds}).fetchall()
        reasons: dict[str, list[str]] = {}
        for other, relation in sorted(direct):
            reasons.setdefault(other, []).append(relation)

        entities = self.conn.execute(
            """
            SELECT DISTINCT ON (ent.id) ent.id, ent.kind, ent.key, ent.label, e.relation
              FROM edges e JOIN nodes ent ON ent.id = e.dst AND ent.kind = ANY(%s)
             WHERE e.src = %s AND e.relation <> ALL(%s)
             ORDER BY ent.id, e.relation
            """, (list(ENTITY_KINDS), seed_id, sorted(EXPANSION_EXCLUDED_RELATIONS))).fetchall()
        omitted: dict[str, int] = {}
        for entity_id, kind, _, label, _ in sorted(entities, key=lambda r: (r[4], r[1], r[2])):
            reason = f"shares {kind}: {label}"
            (members,) = fetch_one(
                self.conn, "SELECT count(DISTINCT src) FROM edges WHERE dst = %s AND src <> %s", (entity_id, seed_id))
            if members > HUB_DEGREE:
                omitted[reason] = members  # too many to be informative: say so instead of listing
                continue
            shown = self.conn.execute(
                """
                SELECT o.key FROM edges e JOIN nodes o ON o.id = e.src AND o.kind = ANY(%s)
                 WHERE e.dst = %s AND e.src <> %s
                 GROUP BY o.id, o.kind, o.key
                 ORDER BY (o.kind = %s), o.key COLLATE "C" DESC
                 LIMIT %s
                """, (record_kinds, entity_id, seed_id, seed_kind, per_entity)).fetchall()
            for (other,) in shown:
                reasons.setdefault(other, []).append(reason)
            if members > len(shown):
                omitted[reason] = members - len(shown)

        if not reasons:
            return RelatedRecords([], omitted)
        chunks = {
            row[0]: self._chunk(row)
            for row in self.conn.execute(
                f"SELECT {_COLUMNS} FROM chunks WHERE project_id = %s AND record_id = ANY(%s) "
                "AND coalesce(metadata ->> 'is_latest', 'yes') = 'yes'",
                (project_id, list(reasons)),
            )
        }
        items = [Related(chunks[record], "; ".join(dict.fromkeys(why)))
                 for record, why in reasons.items() if record in chunks]
        if len(items) > limit:
            omitted["beyond the overall limit"] = len(items) - limit
        return RelatedRecords(items[:limit], omitted)

    def vector_search(self, vector: list[float], limit: int, filters: Filters | None = None) -> list[Hit]:
        if len(vector) != self.embedding_dim:
            raise StoreError(
                f"The query vector has {len(vector)} dimensions but the database holds {self.embedding_dim}-"
                "dimensional vectors. The embedding model used for the question must be the one used to ingest."
            )
        filter_sql, filter_params = _filter_sql(filters)
        literal = vector_literal(vector)
        rows = self.conn.execute(
            f"""
            SELECT {_COLUMNS}, 1 - (embedding <=> %s::vector) AS similarity
            FROM chunks
            WHERE project_id = %s AND embedding IS NOT NULL{filter_sql}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (literal, self._require_project_id(), *filter_params, literal, limit),
        ).fetchall()
        hits = [Hit(self._chunk(row), float(row[4])) for row in rows]
        hits = [hit for hit in hits if hit.score == hit.score]  # a zero query vector gives NaN; drop those
        # Iterative scanning can return rows slightly out of order, and equal scores have no order at all, so
        # sort here: best first, then by record id and revision so the result is the same every time.
        return sorted(hits, key=lambda hit: (-hit.score, hit.chunk.record_id, revision_of(hit.chunk)))

    def aggregate(self, record_type: str, field: str, operation: str, filters: Filters | None = None) -> Aggregate:
        """Exact figures computed by the database over current revisions. `field` and `operation` were checked against
        a fixed list, and the field name still travels as a query parameter, never as SQL text."""
        check_aggregate_request(record_type, field, operation)
        project_id = self._require_project_id()
        filter_sql, filter_params = _filter_sql(filters)
        # Only rows whose metadata value is a plain number take part (the chunker writes nothing else), so the cast cannot fail.
        where = (
            "FROM chunks WHERE project_id = %s AND record_type = %s AND coalesce(metadata ->> 'is_latest', 'yes') = 'yes' "
            "AND metadata ->> %s ~ '^-?[0-9]+(\\.[0-9]+)?$'" + filter_sql
        )
        params = [project_id, record_type, field, *filter_params]
        count, high, low, total = fetch_one(
            self.conn, f"SELECT count(*), max((metadata ->> %s)::numeric), min((metadata ->> %s)::numeric), "
                       f"sum((metadata ->> %s)::numeric) {where}", [field, field, field, *params])
        if count == 0:
            return Aggregate(record_type, field, operation, Decimal(0) if operation == "count" else None, 0)
        ids: tuple[str, ...] = ()
        more = 0
        if operation in ("max", "min"):
            wanted = high if operation == "max" else low
            reaching = [r[0] for r in self.conn.execute(
                f"SELECT record_id {where} AND (metadata ->> %s)::numeric = %s ORDER BY record_id COLLATE \"C\" LIMIT %s",
                [*params, field, wanted, MAX_AGGREGATE_IDS + 1]).fetchall()]
            ids, more = tuple(reaching[:MAX_AGGREGATE_IDS]), 0
            if len(reaching) > MAX_AGGREGATE_IDS:  # ask for the true number of the others only when there are some
                (total_reaching,) = fetch_one(self.conn, f"SELECT count(*) {where} AND (metadata ->> %s)::numeric = %s",
                                              [*params, field, wanted])
                more = total_reaching - MAX_AGGREGATE_IDS
            value = wanted
        elif operation == "sum":
            value = total
        elif operation == "avg":
            value = (total / count).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)  # sum and count are exact, so is this
        else:
            value = Decimal(count)
        return Aggregate(record_type, field, operation, value, count, ids, more)

    def keyword_search(self, query: str, limit: int, filters: Filters | None = None) -> list[Hit]:
        """Records containing any word of the question, scored by the rarity of the words they contain. Records that
        match the same words tie on that score; PostgreSQL's ts_rank (how often the words occur) then puts the ones that
        mention them most first, as BM25 does in the memory store. Versions that also divide by the text length
        scored worse on the retrieval evaluation."""
        terms = build_terms(query)
        if not terms:
            return []
        project_id = self._require_project_id()
        total = self.count()
        # How many records contain each term (one indexed probe per term, in a single round trip).
        frequencies: dict[str, int] = dict(self.conn.execute(
            """
            SELECT t.term, count(c.id)
              FROM unnest(%s::text[]) AS t(term)
              LEFT JOIN chunks c ON c.project_id = %s AND c.search_text @@ to_tsquery('simple', t.term)
             GROUP BY t.term
            """, (terms, project_id)).fetchall())
        weights = [idf(frequencies.get(term, 0), total) for term in terms]
        filter_sql, filter_params = _filter_sql(filters)
        # The score is written out as one CASE per term (the number of terms is capped, and every value is a
        # parameter). A sub-query per matching row was four times slower once a question matched thousands of rows.
        score_sql = " + ".join("(CASE WHEN search_text @@ to_tsquery('simple', %s) THEN %s::float8 ELSE 0 END)" for _ in terms)
        score_params = [value for term, weight in zip(terms, weights, strict=True) for value in (term, weight)]
        rows = self.conn.execute(
            f"""
            SELECT {_COLUMNS}, {score_sql} AS score
              FROM chunks
             WHERE project_id = %s AND search_text @@ to_tsquery('simple', %s){filter_sql}
             ORDER BY score DESC, ts_rank(search_text, to_tsquery('simple', %s), 0) DESC, record_id COLLATE "C", revision
             LIMIT %s
            """,
            (*score_params, project_id, " | ".join(terms), *filter_params, " | ".join(terms), limit),
        ).fetchall()
        return [Hit(self._chunk(row), float(row[4])) for row in rows]

    # ---- writing -------------------------------------------------------------------------------

    def _stored_rows(self, project_id: int) -> dict[tuple[str, int], _Stored]:
        rows = self.conn.execute(
            "SELECT record_id, revision, text_hash, embedder_id, embedding IS NOT NULL, metadata "
            "FROM chunks WHERE project_id = %s", (project_id,))
        return {(r[0], r[1]): _Stored(r[2], r[3], r[4], r[5]) for r in rows}

    @staticmethod
    def _refuse_nul(chunk: Chunk) -> None:
        """PostgreSQL cannot store the NUL character in text, and the CSV reader lets it through. Say which
        record has it, up front, instead of failing in the middle of a write after the embedding was paid for."""
        values = [chunk.record_id, chunk.text, *chunk.metadata.keys(), *chunk.metadata.values()]
        if any("\x00" in value for value in values):
            raise StoreError(f"Record {chunk.record_id} contains a NUL (\\x00) character, which PostgreSQL cannot "
                             "store. Remove it from the CSV.")

    def _embed(self, texts: list[str], embed: Embedder) -> list[list[float]]:
        if not texts:
            return []
        vectors = embed(texts)
        if len(vectors) != len(texts):
            raise StoreError(f"The embedder returned {len(vectors)} vectors for {len(texts)} texts")
        for vector in vectors:
            if len(vector) != self.embedding_dim:
                raise StoreError(
                    f"The embedding model returns {len(vector)}-dimensional vectors but the database expects "
                    f"{self.embedding_dim}. Use the model the database was created for (see EMBED_* in .env). Without "
                    "EMBED_API_KEY the offline test embedder is used, and it makes 256-dimensional vectors."
                )
        return vectors

    def sync(self, chunks: list[Chunk], embed: Embedder, embedder_id: str, allow_missing_types: bool = False,
             source_files: dict[str, str] | None = None) -> SyncReport:
        """Make this project's rows equal `chunks`. Idempotent: running it twice changes nothing the second time.

        A whole record type that is stored but absent from `chunks` (a CSV that went missing from the folder) is
        refused, because syncing would delete every record of that type; allow_missing_types=True says it is meant."""
        if source_files is not None:
            source_files = dict(source_files)
            with materialize(source_files) as folder:
                if chunks_fingerprint(load_chunks(folder)) != chunks_fingerprint(chunks):
                    raise StoreError("Project snapshot does not match the indexed records.")
        incoming: dict[tuple[str, int], Chunk] = {}
        for chunk in chunks:
            self._refuse_nul(chunk)  # before anything is embedded: the embedding calls cost money
            key = (chunk.record_id, revision_of(chunk))
            if key in incoming:
                raise StoreError(f"Two chunks share record {key[0]} revision {key[1]}")
            incoming[key] = chunk

        project_id = self._find_project_id()
        stored = self._stored_rows(project_id) if project_id is not None else {}
        if not allow_missing_types:
            vanished = sorted({row.metadata.get("record_type", "") for row in stored.values()} - {c.record_type for c in chunks} - {""})
            if vanished:
                raise StoreError(
                    f"The new data has no {', '.join(vanished)} records, but the database holds some. Syncing would delete "
                    "all of them. If a CSV is missing by mistake, restore it; if they should go, say so "
                    "(allow_missing_types=True, or --allow-missing-types on the command line).")

        # Decide what to do. Only 'embed_now' rows cost an embedding call.
        embed_now: list[tuple[tuple[str, int], Chunk]] = []
        metadata_only: list[tuple[tuple[str, int], Chunk]] = []
        unchanged = 0
        for key, chunk in incoming.items():
            old = stored.get(key)
            if old is None or old.text_hash != text_hash(chunk.text) or old.embedder_id != embedder_id or not old.has_embedding:
                embed_now.append((key, chunk))
            elif old.metadata != chunk.metadata:
                metadata_only.append((key, chunk))
            else:
                unchanged += 1
        deleted = [key for key in stored if key not in incoming]

        # The slow, failure-prone part (the network) happens BEFORE the transaction, so nothing is held open.
        vectors = self._embed([chunk.text for _, chunk in embed_now], embed)

        with self.conn.transaction():  # everything below is saved together, or not at all
            project_id = self._ensure_project()
            self.conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", ("rag_sync:" + self.project_slug,))
            if self._stored_rows(project_id) != stored:
                raise StoreError(f"Project {self.project_slug!r} was changed by someone else while this ingest "
                                 "was running. Nothing was written; run it again.")
            bulk = len(embed_now) >= self.bulk_threshold
            if bulk:
                log_event("bulk_load", rows=len(embed_now), project=self.project_slug)
                self.conn.execute(f"DROP INDEX IF EXISTS {VECTOR_INDEX}")
            with self.conn.cursor() as cursor:
                cursor.executemany(
                    """
                    INSERT INTO chunks (project_id, record_id, revision, record_type, text, text_hash,
                                        metadata, embedding, embedder_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector, %s)
                    ON CONFLICT (project_id, record_id, revision) DO UPDATE SET
                        record_type = EXCLUDED.record_type, text = EXCLUDED.text, text_hash = EXCLUDED.text_hash,
                        metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding,
                        embedder_id = EXCLUDED.embedder_id, updated_at = now()
                    """,
                    [(project_id, key[0], key[1], chunk.record_type, chunk.text, text_hash(chunk.text),
                      json.dumps(chunk.metadata), vector_literal(vector), embedder_id)
                     for (key, chunk), vector in zip(embed_now, vectors, strict=True)],
                )
                cursor.executemany(
                    "UPDATE chunks SET metadata = %s::jsonb, updated_at = now() "
                    "WHERE project_id = %s AND record_id = %s AND revision = %s",
                    [(json.dumps(chunk.metadata), project_id, key[0], key[1]) for key, chunk in metadata_only],
                )
                cursor.executemany(
                    "DELETE FROM chunks WHERE project_id = %s AND record_id = %s AND revision = %s",
                    [(project_id, record_id, revision) for record_id, revision in deleted],
                )
            self._sync_graph(project_id, build_graph(chunks))
            if source_files is None:
                self.conn.execute("DELETE FROM project_source_snapshots WHERE project_id = %s", (project_id,))
            else:
                self.conn.execute("""INSERT INTO project_source_snapshots (project_id, version, files)
                                  VALUES (%s, %s, %s::jsonb) ON CONFLICT (project_id) DO UPDATE
                                  SET version = EXCLUDED.version, files = EXCLUDED.files, generation = EXCLUDED.generation""",
                                  (project_id, version(source_files), json.dumps(source_files)))
            if bulk:
                self.conn.execute(f"SET LOCAL maintenance_work_mem = '{BULK_INDEX_MEMORY_MB}MB'")  # a faster index build, for this transaction only
                self.conn.execute(_CREATE_VECTOR_INDEX)
        self._project_id = project_id

        inserted = sum(1 for key, _ in embed_now if key not in stored)
        return SyncReport(inserted=inserted, updated=len(embed_now) - inserted + len(metadata_only),
                          unchanged=unchanged, deleted=len(deleted), embedded=len(embed_now))

    def _ensure_project(self) -> int:
        (project_id,) = fetch_one(
            self.conn,
            "INSERT INTO projects (slug) VALUES (%s) ON CONFLICT (slug) DO UPDATE SET slug = EXCLUDED.slug RETURNING id",
            (self.project_slug,),
        )
        return project_id

    def _sync_graph(self, project_id: int, graph: Graph) -> None:
        """Make the graph tables match `graph`. Skipped when nothing in it changed and the tables are intact, so
        syncing unchanged data leaves the nodes (and their ids) exactly as they were."""
        fingerprint = hashlib.sha256(json.dumps([
            [(n.kind, n.key, n.label) for n in graph.nodes],
            [(e.src_kind, e.src_key, e.relation, e.dst_kind, e.dst_key) for e in graph.edges],
        ]).encode("utf-8")).hexdigest()
        setting = f"graph:{project_id}"
        row = self.conn.execute("SELECT value FROM rag_settings WHERE key = %s", (setting,)).fetchone()
        if row is not None and row[0] == fingerprint:
            # The fingerprint is stored apart from the graph it describes, so also check the tables still hold it:
            # after a manual delete or a partial restore the counts differ and the graph is rebuilt.
            nodes, edges = fetch_one(
                self.conn,
                "SELECT (SELECT count(*) FROM nodes WHERE project_id = %s), (SELECT count(*) FROM edges WHERE project_id = %s)",
                (project_id, project_id))
            if (nodes, edges) == (len(graph.nodes), len(graph.edges)):
                return
        self._rebuild_graph(project_id, graph)
        self.conn.execute(
            "INSERT INTO rag_settings (key, value) VALUES (%s, %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
            (setting, fingerprint))

    def _rebuild_graph(self, project_id: int, graph: Graph) -> None:
        """Recreate this project's nodes and edges from the graph (graph.build_graph, shared with the in-memory store)."""
        self.conn.execute("DELETE FROM edges WHERE project_id = %s", (project_id,))  # first: cheaper than cascading
        self.conn.execute("DELETE FROM nodes WHERE project_id = %s", (project_id,))
        with self.conn.cursor() as cursor:
            cursor.executemany("INSERT INTO nodes (project_id, kind, key, label) VALUES (%s, %s, %s, %s)",
                               [(project_id, n.kind, n.key, n.label) for n in graph.nodes])
        node_ids = {(kind, key): node_id for node_id, kind, key in
                    self.conn.execute("SELECT id, kind, key FROM nodes WHERE project_id = %s", (project_id,))}
        with self.conn.cursor() as cursor:
            cursor.executemany(
                "INSERT INTO edges (project_id, src, dst, relation) VALUES (%s, %s, %s, %s)",
                [(project_id, node_ids[(e.src_kind, e.src_key)], node_ids[(e.dst_kind, e.dst_key)], e.relation)
                 for e in graph.edges])
