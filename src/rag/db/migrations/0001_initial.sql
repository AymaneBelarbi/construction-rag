-- Initial schema: projects, record chunks (with vectors and full-text), and a knowledge graph.
--
-- {{EMBEDDING_DIM}} is filled in by the migration runner (1536 for OpenAI text-embedding-3-small).
-- Changing it later needs a NEW migration and a re-embed; the runner refuses to silently change it.

CREATE EXTENSION IF NOT EXISTS vector;

-- One row per project (a job). Every other table hangs off it, so deleting a project deletes its data.
CREATE TABLE projects (
    id          bigserial PRIMARY KEY,
    slug        text NOT NULL UNIQUE,
    created_at  timestamptz NOT NULL DEFAULT now()
);

-- One row per record revision (an RFI, a submittal revision, a daily report ...): the text the model
-- reads, its vector, and a full-text index.
CREATE TABLE chunks (
    id           bigserial PRIMARY KEY,
    project_id   bigint NOT NULL REFERENCES projects (id) ON DELETE CASCADE,
    record_id    text NOT NULL,                    -- RFI-003, S-001, DR-2026-08-03
    revision     integer NOT NULL DEFAULT 0,       -- submittals have several
    record_type  text NOT NULL,
    text         text NOT NULL,
    text_hash    text NOT NULL,                    -- lets ingest skip unchanged records
    metadata     jsonb NOT NULL DEFAULT '{}',      -- status, is_latest, injection_suspected ...
    embedding    vector({{EMBEDDING_DIM}}),        -- NULL until embedded
    embedder_id  text,                             -- which model made the vector
    search_text  tsvector GENERATED ALWAYS AS (to_tsvector('simple', text)) STORED,
    updated_at   timestamptz NOT NULL DEFAULT now(),
    UNIQUE (project_id, record_id, revision),
    CHECK ((embedding IS NULL) = (embedder_id IS NULL))   -- a vector must say which model made it
);

-- Approximate nearest-neighbour search by cosine distance (HNSW), keyword search, and the filters.
CREATE INDEX chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX chunks_search_gin ON chunks USING gin (search_text);
CREATE INDEX chunks_metadata_gin ON chunks USING gin (metadata jsonb_path_ops);
CREATE INDEX chunks_project_type ON chunks (project_id, record_type);

-- Knowledge graph. A node is a record (kind = its type, key = its id) or an entity such as a trade,
-- a room, a spec section or a drawing sheet (kind = trade/room/..., key = its name).
CREATE TABLE nodes (
    id          bigserial PRIMARY KEY,
    project_id  bigint NOT NULL REFERENCES projects (id) ON DELETE CASCADE,
    kind        text NOT NULL CHECK (kind <> ''),
    key         text NOT NULL CHECK (key <> ''),
    UNIQUE (project_id, kind, key)
);

-- A directed, typed edge: RFI-011 --related_rfi--> CE-004, PL-001 --responsible_party--> Sample Painting.
-- Search walks edges in both directions.
CREATE TABLE edges (
    project_id  bigint NOT NULL REFERENCES projects (id) ON DELETE CASCADE,
    src         bigint NOT NULL REFERENCES nodes (id) ON DELETE CASCADE,
    dst         bigint NOT NULL REFERENCES nodes (id) ON DELETE CASCADE,
    relation    text NOT NULL CHECK (relation ~ '^[a-z][a-z_]*$'),
    PRIMARY KEY (src, dst, relation),
    CHECK (src <> dst)
);
CREATE INDEX edges_dst ON edges (dst);

-- Facts about this database that the code checks on start (the embedding size must never drift).
CREATE TABLE rag_settings (
    key    text PRIMARY KEY,
    value  text NOT NULL
);
INSERT INTO rag_settings (key, value) VALUES ('embedding_dim', '{{EMBEDDING_DIM}}');
