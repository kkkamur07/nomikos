"""Store per-character model confidences on line transcriptions.

Before this revision the platform kept one confidence for a whole line, so the
page editor painted every character with the line score even though the model
had reported a score per character. ``line_transcriptions.character_confidences``
holds those scores as a JSON array of floats aligned one to one with the code
points of the row's text. The characters themselves are not repeated here
because they are already in ``text``.

Nullable with no default and no backfill: human-written rows describe no model
guess, and rows written before this revision keep null, which the API serves as
"no per-character scores". Adding a nullable column without a default does not
rewrite the table.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "007_line_character_confidences"
down_revision: str | None = "006_public_sharing"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "line_transcriptions",
        sa.Column(
            "character_confidences",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("line_transcriptions", "character_confidences")
