"""Record what the model proposed, separately from what the human ended with.

Before this revision the platform could answer "what is on this page" and nothing
else. Every human edit overwrote the segmenter's geometry in place and flipped
``lines.source`` to manual; a deletion removed the row; a re-segment deleted the
untouched lines outright. The raw segmenter output existed only in ``jobs.result``,
and finished jobs are user-deletable. On the transcription side the layering was
already right, but the model's *identity* was reachable only through
``transcriptions.created_by_job_id -> jobs.model_id -> inference_models``, two
``ON DELETE SET NULL`` hops, so clearing job history left a layer named after a job
that no longer exists and no way to say whose output it was.

What this adds:

* ``segmentation_runs``: one immutable row per segment job per part, holding the
  canonical output verbatim plus the model identity and effective params copied in
  at write time;
* ``lines.segmentation_run_id`` and ``lines.run_external_id``: the origin link, kept
  through every human edit, which is what makes the four training classes
  (unchanged, corrected, deleted, added) derivable without a tombstone table;
* ``lines.updated_at`` and ``lines.updated_by``: when and by whom a line was last
  written;
* the same denormalised model identity on ``transcriptions``, and
  ``line_transcriptions.seeded_from_transcription_id``, which records that a
  ground-truth line was accepted from a model layer rather than typed;
* ``document_parts.reviewed_at`` and ``reviewed_by`` beside the bare ``reviewed``
  boolean.

Every column is nullable or carries a constant server default, so this is metadata
only: PostgreSQL does not rewrite a table to add one, and there is no data
migration that can time out on a large ``lines``. Existing rows simply have no run,
no reviewer and no model identity. Those facts are not recoverable from the current
schema, so nothing is invented for them here; ``scripts/platform/backfill_segmentation_runs.py``
can reconstruct runs for the pages whose ``jobs.result`` rows still exist, and is
deliberately a separate, opt-in, dry-run-by-default script rather than part of this
revision.

Grants: ``002_service_roles`` set ``ALTER DEFAULT PRIVILEGES`` for ``nomikos_migrator``
and ``nomikos_api``, so the new table is reachable by both without anything further
here. ``nomikos_platform_worker`` is not granted it: the worker claims and updates
jobs, and the merge that writes runs happens in the API process.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "011_model_output_provenance"
down_revision: str | None = "010_feedback"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "segmentation_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("part_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_by_job_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("model_name", sa.String(length=255), nullable=True),
        sa.Column("model_provider", sa.String(length=64), nullable=True),
        sa.Column("model_artifact_ref", sa.String(length=1024), nullable=True),
        sa.Column("registry_model_id", sa.String(length=255), nullable=True),
        sa.Column("registry_tag", sa.String(length=64), nullable=True),
        sa.Column(
            "params", postgresql.JSONB(astext_type=sa.Text()), server_default="{}", nullable=False
        ),
        sa.Column("image_width", sa.Integer(), nullable=True),
        sa.Column("image_height", sa.Integer(), nullable=True),
        sa.Column("line_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("block_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column(
            "lines", postgresql.JSONB(astext_type=sa.Text()), server_default="[]", nullable=False
        ),
        sa.Column(
            "blocks", postgresql.JSONB(astext_type=sa.Text()), server_default="[]", nullable=False
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        # The run describes one page and dies with it. The job is only where it came
        # from, so deleting job history nulls the reference and keeps the record,
        # which is the entire point of denormalising the identity columns above.
        sa.ForeignKeyConstraint(["part_id"], ["document_parts.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["created_by_job_id"], ["jobs.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_segmentation_runs_part_id"), "segmentation_runs", ["part_id"], unique=False
    )
    op.create_index(
        "ix_segmentation_runs_part_created",
        "segmentation_runs",
        ["part_id", "created_at"],
        unique=False,
    )

    op.add_column(
        "lines", sa.Column("segmentation_run_id", postgresql.UUID(as_uuid=True), nullable=True)
    )
    op.add_column("lines", sa.Column("run_external_id", sa.String(length=255), nullable=True))
    op.add_column("lines", sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("lines", sa.Column("updated_by", postgresql.UUID(as_uuid=True), nullable=True))
    # SET NULL, not CASCADE. A page must never lose its geometry because the record
    # of what proposed it was removed.
    op.create_foreign_key(
        "fk_lines_segmentation_run_id",
        "lines",
        "segmentation_runs",
        ["segmentation_run_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "fk_lines_updated_by", "lines", "users", ["updated_by"], ["id"], ondelete="SET NULL"
    )
    op.create_index(
        "ix_lines_segmentation_run_external",
        "lines",
        ["segmentation_run_id", "run_external_id"],
        unique=False,
    )

    op.add_column("transcriptions", sa.Column("model_name", sa.String(length=255), nullable=True))
    op.add_column(
        "transcriptions", sa.Column("model_provider", sa.String(length=64), nullable=True)
    )
    op.add_column(
        "transcriptions", sa.Column("model_artifact_ref", sa.String(length=1024), nullable=True)
    )
    op.add_column(
        "transcriptions", sa.Column("registry_model_id", sa.String(length=255), nullable=True)
    )
    op.add_column("transcriptions", sa.Column("registry_tag", sa.String(length=64), nullable=True))
    op.add_column(
        "transcriptions",
        sa.Column(
            "model_params",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="{}",
            nullable=False,
        ),
    )

    op.add_column(
        "line_transcriptions",
        sa.Column("seeded_from_transcription_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    # SET NULL rather than CASCADE: deleting the model layer a text was accepted from
    # must not delete the corrected ground truth that grew out of it.
    op.create_foreign_key(
        "fk_line_transcriptions_seeded_from",
        "line_transcriptions",
        "transcriptions",
        ["seeded_from_transcription_id"],
        ["id"],
        ondelete="SET NULL",
    )

    op.add_column(
        "document_parts", sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True)
    )
    op.add_column(
        "document_parts", sa.Column("reviewed_by", postgresql.UUID(as_uuid=True), nullable=True)
    )
    op.create_foreign_key(
        "fk_document_parts_reviewed_by",
        "document_parts",
        "users",
        ["reviewed_by"],
        ["id"],
        ondelete="SET NULL",
    )


# Downgrading destroys every model-output record on the page: the runs themselves,
# the origin links, the accepted-from notes and the reviewer stamps. None of it is
# reconstructible from what remains, and a later re-upgrade brings the columns back
# empty. Take a dump first if the database has runs in it.
def downgrade() -> None:
    op.drop_constraint("fk_document_parts_reviewed_by", "document_parts", type_="foreignkey")
    op.drop_column("document_parts", "reviewed_by")
    op.drop_column("document_parts", "reviewed_at")

    op.drop_constraint(
        "fk_line_transcriptions_seeded_from", "line_transcriptions", type_="foreignkey"
    )
    op.drop_column("line_transcriptions", "seeded_from_transcription_id")

    op.drop_column("transcriptions", "model_params")
    op.drop_column("transcriptions", "registry_tag")
    op.drop_column("transcriptions", "registry_model_id")
    op.drop_column("transcriptions", "model_artifact_ref")
    op.drop_column("transcriptions", "model_provider")
    op.drop_column("transcriptions", "model_name")

    op.drop_index("ix_lines_segmentation_run_external", table_name="lines")
    op.drop_constraint("fk_lines_updated_by", "lines", type_="foreignkey")
    op.drop_constraint("fk_lines_segmentation_run_id", "lines", type_="foreignkey")
    op.drop_column("lines", "updated_by")
    op.drop_column("lines", "updated_at")
    op.drop_column("lines", "run_external_id")
    op.drop_column("lines", "segmentation_run_id")

    op.drop_index("ix_segmentation_runs_part_created", table_name="segmentation_runs")
    op.drop_index(op.f("ix_segmentation_runs_part_id"), table_name="segmentation_runs")
    op.drop_table("segmentation_runs")
