-- Equivalent of alembic revision 010_feedback, for production, which never runs
-- alembic automatically. Apply once, before deploying the API build that serves
-- POST /feedback and GET /admin/feedback.
--
-- A plain CREATE TABLE with no data migration: the table starts empty and
-- fills as users send notes.

-- Constraint names follow alembic's naming convention, so a later migration that
-- alters them by name finds the same names here as on a migrated database.
CREATE TABLE IF NOT EXISTS feedback (
    id uuid NOT NULL,
    user_id uuid NOT NULL,
    message varchar(2000) NOT NULL,
    page_path varchar(512),
    created_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT pk_feedback PRIMARY KEY (id),
    CONSTRAINT fk_feedback_user_id_users FOREIGN KEY (user_id)
        REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS ix_feedback_user_created ON feedback (user_id, created_at);

-- Default privileges cover only tables created by the role that set them (002), so
-- grant the API role explicitly, as migration 003 does. Check `\dp feedback` after.
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'nomikos_api') THEN
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE feedback TO nomikos_api;
  END IF;
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'nomikos_migrator') THEN
    GRANT ALL PRIVILEGES ON TABLE feedback TO nomikos_migrator;
  END IF;
END
$$;

-- Keep alembic's bookkeeping in step, so a later `alembic upgrade head` does not
-- create the table a second time. Skipped without an alembic_version table,
-- and a no-op unless the database was at 009_block_label before the statements above.
DO $$
BEGIN
  IF to_regclass('alembic_version') IS NOT NULL THEN
    UPDATE alembic_version SET version_num = '010_feedback'
    WHERE version_num = '009_block_label';
  END IF;
END
$$;
