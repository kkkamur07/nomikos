"""Add ``documents.script_fonts``, the font a document renders each script in.

The editor, the PDF and the Word export all drew Syriac in one fixed style. A
document of West Syriac manuscripts needs the Serto letterforms and one of East
Syriac ones the Madnhaya forms, so the choice belongs to the document. The column
holds ``{script: choice id}`` from ``backend.core.fonts.SCRIPT_FONTS``.

Nullable and without a default: NULL means every script uses its default, and
``ADD COLUMN`` of a nullable column is metadata-only on Postgres.

Downgrading drops the column and every document's choice with it.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "012_document_script_fonts"
down_revision: str | None = "011_model_output_provenance"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("script_fonts", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("documents", "script_fonts")
