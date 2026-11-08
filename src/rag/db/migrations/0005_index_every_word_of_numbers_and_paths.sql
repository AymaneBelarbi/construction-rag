-- Keyword search could not find an exact amount.
--
-- PostgreSQL's text parser keeps "4850.00", "john.smith@acme.com", "A-201.pdf" and "08/05/2026" as ONE token, but a
-- question is split at every punctuation mark, so the search for 4850 looked for a token that is not in the index:
-- "which change event costs 4850.00" could not find the change event that says "Estimated cost: 4850.00".
--
-- Every run of characters that is not a letter or a digit now becomes one space before indexing (migration 0002 did
-- this for hyphens only), which is the same rule the question side uses (a word is letters and digits). IDs and
-- dates are still searched as exact phrases (rfi <-> 003). Postgres cannot change a generated column in place, so it
-- is dropped and added again, with its index.
ALTER TABLE chunks DROP COLUMN search_text;
ALTER TABLE chunks ADD COLUMN search_text tsvector
    GENERATED ALWAYS AS (to_tsvector('simple', regexp_replace(record_id || ' ' || text, '[^[:alnum:]]+', ' ', 'g'))) STORED;
CREATE INDEX chunks_search_gin ON chunks USING gin (search_text);
