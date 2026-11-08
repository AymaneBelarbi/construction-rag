-- Fixes found while building the store and in review of 0001 (an applied migration is never edited).

-- 1. Full-text search on IDs and dates.
--    The default text parser reads "RFI-003" as the two tokens 'rfi' and '-003', so searching for RFI-003
--    could match a record that only mentions "RFI-011 ... CE-003". Hyphens are now replaced by spaces
--    before indexing (tokens: rfi, 003), the store searches IDs as exact phrases (rfi <-> 003), and the
--    record id itself is indexed too (a daily report's id DR-2026-08-03 is not in its text).
ALTER TABLE chunks DROP COLUMN search_text;
ALTER TABLE chunks ADD COLUMN search_text tsvector
    GENERATED ALWAYS AS (to_tsvector('simple', replace(record_id || ' ' || text, '-', ' '))) STORED;
CREATE INDEX chunks_search_gin ON chunks USING gin (search_text);

-- 2. The metadata index must also answer "does this key exist?" (metadata ? 'injection_suspected'),
--    which the smaller jsonb_path_ops index cannot.
DROP INDEX chunks_metadata_gin;
CREATE INDEX chunks_metadata_gin ON chunks USING gin (metadata);

-- 3. An edge may only join two nodes of its own project. Without this, an edge in project A could point at
--    a node in project B, and graph search would leak B's records into A's answers.
ALTER TABLE nodes ADD CONSTRAINT nodes_project_id_id_key UNIQUE (project_id, id);
ALTER TABLE edges ADD CONSTRAINT edges_src_in_same_project
    FOREIGN KEY (project_id, src) REFERENCES nodes (project_id, id) ON DELETE CASCADE;
ALTER TABLE edges ADD CONSTRAINT edges_dst_in_same_project
    FOREIGN KEY (project_id, dst) REFERENCES nodes (project_id, id) ON DELETE CASCADE;
