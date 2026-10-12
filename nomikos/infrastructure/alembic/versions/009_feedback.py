"""User feedback: one small table behind the Feedback button.

Any signed-in user can send a short note from anywhere in the app; the admin
reads the newest rows on the dashboard. The table carries who wrote it and
which app route they were on, nothing else. Deleting a user deletes their
notes with them, so no orphaned rows outlive the account they came from.

Downgrading drops the table and every note with it.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision: str = "009_feedback"
down_revision: str | None = "008_block_label"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "feedback",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("message", sa.String(length=2000), nullable=False),
        sa.Column("page_path", sa.String(length=512), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_feedback_user_created", "feedback", ["user_id", "created_at"])
    _grant_runtime_privileges()


def _grant_runtime_privileges() -> None:
    """Grant the API role explicitly, as 003 does: 002's default privileges only
    cover tables created by the role that ran 002."""
    op.execute(
        """
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
        """
    )


def downgrade() -> None:
    op.drop_index("ix_feedback_user_created", table_name="feedback")
    op.drop_table("feedback")
