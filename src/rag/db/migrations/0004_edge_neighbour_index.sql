-- Finding the records that share an entity reads the edges INTO that entity, ordered by their source record.
-- (dst, src) answers that from the index alone, and replaces the old single-column index on dst.
CREATE INDEX edges_dst_src ON edges (dst, src);
DROP INDEX edges_dst;
