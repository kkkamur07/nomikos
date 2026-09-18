#!/usr/bin/env python3
"""Reconstruct ``segmentation_runs`` rows from the job history that still exists.

Optional, and deliberately not part of migration 007. The migration is additive
metadata only and must stay that way: a data migration over every finished segment
job on a populated database is exactly the sort of thing that times out halfway and
leaves a half-written table behind a held lock.

What it can and cannot recover:

* the model identity and the effective params come from the job row, the same read
  the live merge does, so they are exact for every job that survives;
* the *geometry* does not. ``jobs.result`` for a segment job holds counts
  (``lines_count``, ``added_lines``, ...), never the polygons; the canonical output
  was consumed by the merge and never stored. So a reconstructed run has
  ``lines = []`` and ``line_count`` taken from the recorded count, and the diff
  endpoint will report every live line as *added* against it. That is honest: what
  the model proposed for those pages is genuinely gone. The row is still worth
  writing, because it is the only record of which model version ran on which page
  and with what threshold;
* the origin link on existing ``lines`` is reconstructed where it can be. A line
  whose ``source_metadata`` still carries the ``external_id`` and ``job_id`` the
  merge wrote is matched to the run created from that job. Lines written before
  ``source_metadata`` carried a ``job_id`` stay unlinked.

Nothing is overwritten. A part that already has a run from a given job is skipped,
and a line that already has ``segmentation_run_id`` is left alone, so the script is
safe to run twice.

Usage::

    DATABASE_URL=postgresql://... python scripts/platform/backfill_segmentation_runs.py
    DATABASE_URL=postgresql://... python scripts/platform/backfill_segmentation_runs.py --apply

Defaults to a dry run: it prints what it would write and rolls back. The database
comes from the environment (``BACKFILL_DATABASE_URL``, else ``SYNC_DATABASE_URL``,
else ``DATABASE_URL``) and is never taken from an argument, so a URL cannot end up
in a shell history or a process listing.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from typing import Any
from uuid import UUID

from _bootstrap import ensure_nomikos_on_path

_URL_ENV_VARS = ("BACKFILL_DATABASE_URL", "SYNC_DATABASE_URL", "DATABASE_URL")


def _database_url() -> str:
    for name in _URL_ENV_VARS:
        value = os.environ.get(name)
        if value:
            # SQLAlchemy's async driver cannot be used from this synchronous script,
            # and the two URLs differ only by the driver, so normalise rather than
            # refuse and make the operator retype a URL they already have exported.
            return value.replace("postgresql+asyncpg://", "postgresql://")
    raise SystemExit("Set one of " + ", ".join(_URL_ENV_VARS) + " to the database to back-fill.")


def run(*, apply: bool) -> int:
    ensure_nomikos_on_path()

    from sqlalchemy import create_engine, select
    from sqlalchemy.orm import Session

    import infrastructure.models  # noqa: F401 - register all mappers
    from backend.document.infrastructure.orm_models import Line, SegmentationRun
    from backend.jobs.infrastructure.orm_models import Job, JobStatus, JobType
    from backend.ml.application.model_identity import identity_for_job

    engine = create_engine(_database_url())
    written = 0
    linked = 0
    skipped = 0

    with Session(engine) as session:
        jobs = list(
            session.execute(
                select(Job)
                .where(
                    Job.type == JobType.segment,
                    Job.status == JobStatus.done,
                    Job.document_part_id.is_not(None),
                )
                .order_by(Job.created_at)
            )
            .scalars()
            .all()
        )
        existing = {
            run_id
            for run_id in session.execute(
                select(SegmentationRun.created_by_job_id).where(
                    SegmentationRun.created_by_job_id.is_not(None)
                )
            ).scalars()
        }

        runs_by_job: dict[UUID, SegmentationRun] = {}
        for job in jobs:
            if job.id in existing:
                skipped += 1
                continue
            result: dict[str, Any] = job.result or {}
            identity = identity_for_job(session, job)
            run_row = SegmentationRun(
                part_id=job.document_part_id,
                created_by_job_id=job.id,
                model_name=identity.name,
                model_provider=identity.provider,
                model_artifact_ref=identity.artifact_ref,
                registry_model_id=identity.registry_model_id,
                registry_tag=identity.registry_tag,
                params=dict(identity.params or {}),
                line_count=int(result.get("lines_count") or 0),
                block_count=int(result.get("blocks_count") or 0),
                lines=[],
                blocks=[],
                created_at=job.completed_at or job.created_at,
            )
            session.add(run_row)
            runs_by_job[job.id] = run_row
            written += 1
        session.flush()

        # Re-link the live lines that still carry the merge's own breadcrumbs.
        by_part: dict[UUID, list[SegmentationRun]] = defaultdict(list)
        for run_row in runs_by_job.values():
            by_part[run_row.part_id].append(run_row)
        if by_part:
            lines = list(
                session.execute(
                    select(Line).where(
                        Line.part_id.in_(list(by_part)),
                        Line.segmentation_run_id.is_(None),
                        Line.source_metadata.is_not(None),
                    )
                )
                .scalars()
                .all()
            )
            for line in lines:
                metadata = line.source_metadata or {}
                job_id = metadata.get("job_id")
                external_id = metadata.get("external_id")
                if not job_id or not external_id:
                    continue
                try:
                    run_row = runs_by_job.get(UUID(str(job_id)))
                except ValueError:
                    continue
                if run_row is None or run_row.part_id != line.part_id:
                    continue
                line.segmentation_run_id = run_row.id
                line.run_external_id = str(external_id)
                linked += 1

        prefix = "" if apply else "[dry run] "
        print(
            f"{prefix}{written} segmentation_runs row(s) reconstructed "
            f"from {len(jobs)} finished segment job(s)"
        )
        print(f"{prefix}{linked} existing line(s) re-linked to a run")
        print(f"{prefix}{skipped} job(s) skipped, they already have a run")
        if apply:
            session.commit()
        else:
            session.rollback()
            print("Dry run: nothing was committed. Re-run with --apply to write.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Commit the reconstructed rows. Without it the script rolls back.",
    )
    args = parser.parse_args()
    return run(apply=args.apply)


if __name__ == "__main__":
    sys.exit(main())
