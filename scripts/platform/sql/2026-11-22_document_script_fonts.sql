-- Equivalent of alembic revision 012_document_script_fonts, for production, which never
-- runs alembic automatically. Apply once, before deploying the API build that reads the
-- column (the ORM selects documents.script_fonts whenever it loads a document).
--
-- Metadata-only on Postgres: a nullable column without a default does not rewrite the
-- table. Existing documents come back with NULL, which means every script uses its
-- default font.

ALTER TABLE documents ADD COLUMN IF NOT EXISTS script_fonts jsonb;

-- Keep alembic's bookkeeping in step, so a later `alembic upgrade head` does not
-- add the column a second time. Skipped without an alembic_version table,
-- and a no-op unless the database was at 011_model_output_provenance before the
-- statements above.
DO $$
BEGIN
  IF to_regclass('alembic_version') IS NOT NULL THEN
    UPDATE alembic_version SET version_num = '012_document_script_fonts'
    WHERE version_num = '011_model_output_provenance';
  END IF;
END
$$;
