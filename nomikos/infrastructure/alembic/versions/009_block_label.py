"""Add ``blocks.label``, the name a researcher gives a region.

Many photographs carry two folios side by side, and a page's lines were one flat list:
nothing could say "these lines are folio 4v, those are 5r". Blocks already existed as
the layout unit above lines (ADR "Regions are the unit of layout above lines"), but a
block had only a box and an order. The label is what exports print as the region's
heading and what the editor shows on the region and in the transcript.

Nullable and without a default: a segmenter's regions are unnamed until somebody names
them, and ``ADD COLUMN`` of a nullable column is metadata-only on Postgres.

Downgrading drops the column and every label with it; the regions themselves stay.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "009_block_label"
down_revision: str | None = "008_user_is_admin"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("blocks", sa.Column("label", sa.String(length=64), nullable=True))


def downgrade() -> None:
    op.drop_column("blocks", "label")
