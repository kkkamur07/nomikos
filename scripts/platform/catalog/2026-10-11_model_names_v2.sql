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
-- plain UPDATE cannot collide), then the Calamari row is deleted. Jobs keep
-- their UUID references and are unaffected.
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

UPDATE inference_models SET name = 'seg-v1' WHERE name = 'kraken';
UPDATE inference_models SET name = 'seg-v2' WHERE name = 'ppocr';
UPDATE inference_models SET name = 'greek-v1' WHERE name = 'greek-calamari-v1';
UPDATE inference_models SET name = 'armenian-v1' WHERE name = 'armenian-calamari-v1';
UPDATE inference_models SET name = 'coptic-v1' WHERE name = 'coptic-calamari-v1';
UPDATE inference_models SET name = 'syriac-v2' WHERE name = 'syriac-ppocr-v1';

UPDATE model_bindings SET model_id = (SELECT id FROM inference_models WHERE name = 'syriac-v2')
WHERE model_id = (SELECT id FROM inference_models WHERE name = 'syriac-calamari-v2');

DELETE FROM inference_models WHERE name = 'syriac-calamari-v2';

SELECT name, task, artifact_ref FROM inference_models
WHERE name IN ('seg-v1', 'seg-v2', 'greek-v1', 'armenian-v1', 'coptic-v1', 'syriac-v2')
ORDER BY name;

COMMIT;
