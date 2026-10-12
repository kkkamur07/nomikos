"""Add ``users.is_admin``, the one switch behind the admin dashboard.

Until this revision there was no notion of an operator inside the application: every
account was a researcher, and the only way to see how the platform was used, or how much
work the database was doing, was to open a SQL console on the production host. The
admin dashboard (``GET /admin/overview``, ADR "Admin dashboard and request metrics")
answers those questions from inside the app, and it needs a way to tell who may see
them.

A boolean on the account rather than a roles table, deliberately. There is exactly one
privilege to grant and it is all-or-nothing: an admin reads aggregate usage and database
statistics, nothing more. A roles table would be a join on every admin request and a
second migration's worth of design for permissions nobody has asked for. If a second
privilege ever appears, that is the moment to replace this column, not before.

``NOT NULL DEFAULT false`` so every existing and future account is a non-admin until
somebody promotes it on purpose. There is no API that writes this column: promotion is
an ``UPDATE`` run by whoever operates the database (see
``scripts/platform/sql/2026-10-11_user_is_admin.sql`` and
``docs/guides/admin-dashboard.md``), so a compromised account cannot grant itself the
dashboard. The default also makes the ``ADD COLUMN`` a metadata-only change on Postgres
11+, so it does not rewrite ``users``.

Downgrading drops the column and with it every promotion; re-upgrading brings every
account back as a non-admin, which fails closed.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "008_user_is_admin"
down_revision: str | None = "007_line_character_confidences"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column("is_admin", sa.Boolean(), server_default="false", nullable=False),
    )


def downgrade() -> None:
    op.drop_column("users", "is_admin")
