-- Equivalent of alembic revision 009_feedback, for production, which never runs
-- alembic automatically. Apply once, before deploying the API build that serves
-- POST /feedback and GET /admin/feedback.
--
-- A plain CREATE TABLE with no data migration: the table starts empty and
-- fills as users send notes.

CREATE TABLE IF NOT EXISTS feedback (
    id uuid PRIMARY KEY,
    user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    message varchar(2000) NOT NULL,
    page_path varchar(512),
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_feedback_user_created ON feedback (user_id, created_at);

-- Only if this database has an alembic_version table: keep alembic's bookkeeping in step
-- so a later `alembic upgrade head` does not create the table a second time. The WHERE
-- makes it a no-op unless the database was at 008 before the CREATE above.
-- UPDATE alembic_version SET version_num = '009_feedback'
-- WHERE version_num = '008_block_label';
