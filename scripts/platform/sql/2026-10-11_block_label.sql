-- Equivalent of alembic revision 009_block_label, for production, which never runs
-- alembic automatically. Apply once, before deploying the API build that reads the
-- column (the ORM selects blocks.label whenever it loads a page's layout).
--
-- Metadata-only on Postgres: a nullable column without a default does not rewrite the
-- table. Existing regions come back unlabelled.

ALTER TABLE blocks ADD COLUMN IF NOT EXISTS label varchar(64);

-- Keep alembic's bookkeeping in step, so a later `alembic upgrade head` does not
-- add the column a second time. Skipped without an alembic_version table,
-- and a no-op unless the database was at 008_user_is_admin before the statements above.
DO $$
BEGIN
  IF to_regclass('alembic_version') IS NOT NULL THEN
    UPDATE alembic_version SET version_num = '009_block_label'
    WHERE version_num = '008_user_is_admin';
  END IF;
END
$$;
