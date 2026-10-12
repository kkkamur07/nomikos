-- Equivalent of alembic revision 007_user_is_admin, for production, which never runs
-- alembic automatically. Apply once, before deploying the API build that reads the
-- column (the ORM selects users.is_admin on every authenticated request).
--
-- Metadata-only on Postgres 11+: the constant default does not rewrite the table.

ALTER TABLE users ADD COLUMN IF NOT EXISTS is_admin boolean NOT NULL DEFAULT false;

-- Keep alembic's bookkeeping in step, so a later `alembic upgrade head` does not
-- add the column a second time. Skipped without an alembic_version table,
-- and a no-op unless the database was at 006_public_sharing before the statements above.
DO $$
BEGIN
  IF to_regclass('alembic_version') IS NOT NULL THEN
    UPDATE alembic_version SET version_num = '007_user_is_admin'
    WHERE version_num = '006_public_sharing';
  END IF;
END
$$;

-- Promote one account to admin (run separately, after the block above):
-- UPDATE users SET is_admin = true WHERE lower(email) = lower('someone@example.com');
--
-- Demote:
-- UPDATE users SET is_admin = false WHERE lower(email) = lower('someone@example.com');
