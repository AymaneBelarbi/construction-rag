-- A monotonic generation detects any committed ingest, including changes later reverted to identical CSVs.
CREATE SEQUENCE rag_snapshot_generation_seq;
ALTER TABLE project_source_snapshots
    ADD COLUMN generation bigint NOT NULL DEFAULT nextval('rag_snapshot_generation_seq');
