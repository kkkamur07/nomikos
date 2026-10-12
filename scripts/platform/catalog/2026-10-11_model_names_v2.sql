-- Catalog rename to the language-vN display scheme (2026-10-11).
--
-- The pickers print `inference_models.name`; dispatch parses `artifact_ref`,
-- so only `name` changes here. Run once by hand with psql against production:
--
--   psql -U postgres -d <prod-db> -v ON_ERROR_STOP=1 \
--     -f scripts/platform/catalog/2026-10-11_model_names_v2.sql
--
-- Mapping: kraken->seg-v1, ppocr->seg-v2, greek-calamari-v1->greek-v1,
-- armenian-calamari-v1->armenian-v1, coptic-calamari-v1->coptic-v1,
-- syriac-ppocr-v1->syriac-v2. Bindings on the retired syriac-calamari-v2 row
-- move to syriac-v2 first (`model_bindings` has no unique constraint, so a
-- plain UPDATE cannot collide), then the Calamari row is deleted.
--
-- Jobs on syriac-calamari-v2: the script refuses (RAISE, nothing applied) while
-- any of them is pending, waiting or running, since a queued transcribe job that
-- lost its model would fall back to the registry's syriac-calamari-v2 and run the
-- retired model. Wait for them to finish or cancel them, then re-run. Finished
-- jobs (done, failed, cancelled) are repointed to syriac-v2 before the DELETE, so
-- none ends up with `model_id = NULL` (ON DELETE SET NULL); the SELECT below
-- prints how many moved. Jobs on the six renamed rows keep their UUID references.
--
-- Admin dashboard models card: the six renamed rows show under their new names,
-- with their history. syriac-calamari-v2 disappears from the card and its past
-- jobs count under syriac-v2, so syriac-v2's job count, success rate and median
-- time mix the Calamari and PP-OCR runs for windows that reach back before today.
-- Nothing shows as "(no model)" because of this script.
BEGIN;

-- Guard: abort unless all 7 pre-rename rows are present. Re-running after a
-- successful apply raises here instead of renaming twice.
DO $$
DECLARE
  missing text[];
BEGIN
  SELECT array_agg(e.expected ORDER BY e.expected) INTO missing
  FROM (VALUES
    ('kraken'),
    ('ppocr'),
    ('syriac-ppocr-v1'),
    ('greek-calamari-v1'),
    ('armenian-calamari-v1'),
    ('syriac-calamari-v2'),
    ('coptic-calamari-v1')
  ) AS e(expected)
  LEFT JOIN inference_models m ON m.name = e.expected
  WHERE m.id IS NULL;
  IF missing IS NOT NULL THEN
    RAISE EXCEPTION 'catalog rename: expected rows missing: %', array_to_string(missing, ', ');
  END IF;
END
$$;

-- Guard: a queued or running job must not lose its model (see the header).
DO $$
DECLARE
  active bigint;
BEGIN
  SELECT count(*) INTO active FROM jobs
  WHERE status IN ('pending', 'waiting', 'running')
    AND model_id = (SELECT id FROM inference_models WHERE name = 'syriac-calamari-v2');
  IF active > 0 THEN
    RAISE EXCEPTION 'catalog rename: % queued or running job(s) use syriac-calamari-v2; '
      'let them finish or cancel them, then re-run', active;
  END IF;
END
$$;

UPDATE inference_models SET name = 'seg-v1' WHERE name = 'kraken';
UPDATE inference_models SET name = 'seg-v2' WHERE name = 'ppocr';
UPDATE inference_models SET name = 'greek-v1' WHERE name = 'greek-calamari-v1';
UPDATE inference_models SET name = 'armenian-v1' WHERE name = 'armenian-calamari-v1';
UPDATE inference_models SET name = 'coptic-v1' WHERE name = 'coptic-calamari-v1';
UPDATE inference_models SET name = 'syriac-v2' WHERE name = 'syriac-ppocr-v1';

UPDATE model_bindings SET model_id = (SELECT id FROM inference_models WHERE name = 'syriac-v2')
WHERE model_id = (SELECT id FROM inference_models WHERE name = 'syriac-calamari-v2');

WITH moved AS (
  UPDATE jobs SET model_id = (SELECT id FROM inference_models WHERE name = 'syriac-v2')
  WHERE model_id = (SELECT id FROM inference_models WHERE name = 'syriac-calamari-v2')
  RETURNING 1
)
SELECT count(*) AS finished_jobs_moved_to_syriac_v2 FROM moved;

DELETE FROM inference_models WHERE name = 'syriac-calamari-v2';

SELECT name, task, artifact_ref FROM inference_models
WHERE name IN ('seg-v1', 'seg-v2', 'greek-v1', 'armenian-v1', 'coptic-v1', 'syriac-v2')
ORDER BY name;

COMMIT;
