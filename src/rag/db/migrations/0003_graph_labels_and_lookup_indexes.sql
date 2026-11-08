-- Knowledge graph support.

-- A node has a KEY used for matching and a LABEL used for display. Entity keys are normalised (lower case,
-- single spaces) so 'Sample Painting' and 'sample  painting' are one trade; the label keeps the name as first
-- written. For records the label is simply the record id.
ALTER TABLE nodes ADD COLUMN label text;
UPDATE nodes SET label = key WHERE label IS NULL;
ALTER TABLE nodes ALTER COLUMN label SET NOT NULL;

-- Lookups by record id ignore case (upper(key) = upper(...)), which the plain unique indexes cannot serve.
CREATE INDEX nodes_project_upper_key ON nodes (project_id, upper(key));
CREATE INDEX chunks_project_upper_record ON chunks (project_id, upper(record_id));
