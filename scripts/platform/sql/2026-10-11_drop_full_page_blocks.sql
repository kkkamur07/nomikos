-- One-off data fix (ADR 0011): a segmenter's full-page block is not a region.
--
-- The in-app kraken segmenter (blla) wrapped every line of a page in one block whose box
-- is the whole image, and the segment merge stored it. On such a page drawing a region
-- assigns no lines (they all have a region already) and a drawn line is numbered last.
-- The merge no longer stores that block; this removes the ones already stored.
--
-- Rule (segment_merge_service applies the same one to a segmenter's blocks; the manual and
-- label checks keep a region a researcher drew or named). A block is a full-page block when:
--   * it is the only block on its part,
--   * the segmenter drew it (manual_geometry is false) and nobody named it (label is null),
--   * the part has a recorded width and height, and
--   * the bounds of its box points span at least 90 percent of that width and at least
--     90 percent of that height.
-- Its lines keep their geometry and order and get block_id NULL; the block is deleted.
-- Line order does not change: with one block the page was already numbered as one group.
--
-- Idempotent and in one transaction: the first SELECT prints how many blocks and lines
-- it will touch, a second run finds nothing. Run with psql against the production
-- database (a direct session connection, not the transaction pooler port):
--     psql "$DATABASE_URL" -f scripts/platform/sql/2026-10-11_drop_full_page_blocks.sql
-- To look without writing, change the final COMMIT to ROLLBACK.

BEGIN;

CREATE TEMP TABLE full_page_blocks ON COMMIT DROP AS
WITH single AS (
    SELECT part_id
    FROM blocks
    GROUP BY part_id
    HAVING count(*) = 1
),
bounds AS (
    SELECT
        b.id,
        b.part_id,
        max((point ->> 0)::float) - min((point ->> 0)::float) AS box_width,
        max((point ->> 1)::float) - min((point ->> 1)::float) AS box_height
    FROM blocks b
    JOIN single s ON s.part_id = b.part_id
    CROSS JOIN LATERAL jsonb_array_elements(
        CASE WHEN jsonb_typeof(b.box -> 'points') = 'array' THEN b.box -> 'points'
             ELSE '[]'::jsonb END
    ) AS point
    WHERE NOT b.manual_geometry AND b.label IS NULL
    GROUP BY b.id, b.part_id
)
SELECT bounds.id, bounds.part_id
FROM bounds
JOIN document_parts p ON p.id = bounds.part_id
WHERE p.width > 0
  AND p.height > 0
  AND bounds.box_width >= 0.9 * p.width
  AND bounds.box_height >= 0.9 * p.height;

SELECT
    (SELECT count(*) FROM full_page_blocks) AS full_page_blocks,
    (SELECT count(*) FROM lines WHERE block_id IN (SELECT id FROM full_page_blocks))
        AS lines_to_unblock;

UPDATE lines SET block_id = NULL WHERE block_id IN (SELECT id FROM full_page_blocks);

DELETE FROM blocks WHERE id IN (SELECT id FROM full_page_blocks);

COMMIT;
