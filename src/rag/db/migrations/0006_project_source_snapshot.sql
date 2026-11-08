-- The exact source files and their content hash are committed with the index.
CREATE TABLE project_source_snapshots (
    project_id bigint PRIMARY KEY REFERENCES projects (id) ON DELETE CASCADE,
    version text NOT NULL,
    files jsonb NOT NULL
);
