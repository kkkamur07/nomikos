-- Equivalent of alembic revision 008_block_label, for production, which never runs
-- alembic automatically. Apply once, before deploying the API build that reads the
-- column (the ORM selects blocks.label whenever it loads a page's layout).
--
-- Metadata-only on Postgres: a nullable column without a default does not rewrite the
-- table. Existing regions come back unlabelled.

ALTER TABLE blocks ADD COLUMN IF NOT EXISTS label varchar(64);

-- Only if this database has an alembic_version table: keep alembic's bookkeeping in step
-- so a later `alembic upgrade head` does not add the column a second time. The WHERE
-- makes it a no-op unless the database was at 007 before the ALTER above.
-- UPDATE alembic_version SET version_num = '008_block_label'
-- WHERE version_num = '007_user_is_admin';
