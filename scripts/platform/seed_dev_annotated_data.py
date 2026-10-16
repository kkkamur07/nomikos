#!/usr/bin/env python3
"""Import legacy on-disk annotated corpus into the dev@example.com platform user.

Reads the Option D layout under ``data/annotated/data/`` (or ``ANNOTATED_DATA_ROOT``):

  manuscripts/pages/        - page images
  annotations/pages/        - current segment geometry + pairings
  annotations/history/      - restorable annotation snapshots (per page stem)
  transcriptions/pages/     - optional page transcription text (line-broken)

Creates one owned project with documents grouped by manuscript prefix
(e.g. ``Grec_1360_..._6/7/8`` become one multi-part document). Images are copied
into ``MEDIA_ROOT``; structured data is written to Postgres.

Idempotent: skips documents that already exist in the target project (by name).
When a document is skipped, missing history snapshots are backfilled automatically.

Development only: this refuses to run unless ``ENVIRONMENT=development`` (see
``backend/dev/bootstrap.require_development_environment``). It creates the
well-known dev account, so anywhere else it would be planting a backdoor.

Set ``DEV_USER_PASSWORD`` if you need a known login; otherwise a throwaway
password is generated and printed once when the account is created.

Usage (from repository root):

  python scripts/platform/seed_dev_annotated_data.py
  python scripts/platform/seed_dev_annotated_data.py --force
  python scripts/platform/seed_dev_annotated_data.py --skip-history
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from _bootstrap import ensure_nomikos_on_path

ensure_nomikos_on_path()

from sqlalchemy import select

from backend.core.settings._env import REPO_ROOT
from backend.dev.bootstrap import DEV_EMAIL, require_development_environment
from backend.document.application.document_service import DocumentService
from backend.document.infrastructure.media_store import get_media_store
from backend.document.infrastructure.orm_models import Document, DocumentWorkflow
from infrastructure import models as _orm_models  # noqa: F401 - register all mappers
from infrastructure.db import engine, system_session

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.platform.seed._db import (
    _ensure_dev_user,
    _ensure_project,
    _import_document,
)
from scripts.platform.seed._fixtures import (
    ImportStats,
    _discover_page_stems,
    _document_display_name,
    _group_pages_by_document,
)
from scripts.platform.seed._fixtures import (
    _resolve_dev_password as _resolve_dev_password,
)

log = logging.getLogger("seed_dev_annotated_data")

DEFAULT_DATA_ROOT = REPO_ROOT.parent / "data" / "annotated" / "data"
ANNOTATED_DATA_ROOT = Path(os.environ.get("ANNOTATED_DATA_ROOT", str(DEFAULT_DATA_ROOT))).resolve()


def _configure_logging(*, verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(message)s")
    logging.getLogger("sqlalchemy.engine").setLevel(logging.DEBUG if verbose else logging.ERROR)
    # infrastructure.db turns echo on for development; echo logs through its own
    # InstanceLogger level, bypassing the logger level set above. Set it
    # directly on the engine instead.
    engine.echo = verbose


async def run_seed(*, force: bool, import_history: bool) -> ImportStats:
    # Creates the well-known dev account and writes fixture data, so this must
    # never touch a non-development deployment. The guard lives with the other
    # dev bootstrap helpers so scripts share one implementation, not their own.
    require_development_environment()

    if not ANNOTATED_DATA_ROOT.is_dir():
        raise SystemExit(f"Annotated data root not found: {ANNOTATED_DATA_ROOT}")

    page_stems = _discover_page_stems(ANNOTATED_DATA_ROOT)
    if not page_stems:
        raise SystemExit(f"No pages found under {ANNOTATED_DATA_ROOT}")

    document_groups = _group_pages_by_document(page_stems)
    document_service = DocumentService()
    media = get_media_store()
    totals = ImportStats()

    log.info("Data root: %s", ANNOTATED_DATA_ROOT)
    log.info("Pages: %d across %d document(s)", len(page_stems), len(document_groups))
    if import_history:
        log.info("History import: enabled (annotations/history/)")
    else:
        log.info("History import: skipped (--skip-history)")

    async with system_session() as session:
        user = await _ensure_dev_user(session)
        project = await _ensure_project(session, user)
        log.info("Dev user: %s (id=%s)", DEV_EMAIL, user.id)
        log.info("Project: %s (id=%s)", project.slug, project.id)

        for doc_key, stems in sorted(document_groups.items(), key=lambda item: item[0]):
            doc_name = _document_display_name(doc_key, stems)
            log.info("")
            log.info("Document: %s", doc_name)
            doc_stats = await _import_document(
                session=session,
                user=user,
                project=project,
                doc_name=doc_name,
                page_stems=stems,
                data_root=ANNOTATED_DATA_ROOT,
                document_service=document_service,
                media=media,
                force=force,
                import_history=import_history,
            )
            if doc_stats.documents_skipped:
                log.info("  = already exists (use --force to re-import)")
            totals.merge(doc_stats)

        # Landing / public live view needs published workflow.
        published = await session.execute(select(Document).where(Document.project_id == project.id))
        for document in published.scalars().all():
            if document.workflow != DocumentWorkflow.published:
                document.workflow = DocumentWorkflow.published
        await session.commit()
        log.info("Published all documents in project %s for public live view", project.slug)

    return totals


def _print_summary(stats: ImportStats) -> None:
    log.info("")
    log.info(
        "Done: %d document(s) imported, %d skipped, %d part(s), %d segment(s), %d history snapshot(s).",
        stats.documents_created,
        stats.documents_skipped,
        stats.parts,
        stats.segments,
        stats.history_snapshots,
    )
    if stats.warnings:
        log.info("Warnings (%d):", len(stats.warnings))
        for warning in stats.warnings:
            log.info("  - %s", warning)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-import documents even when they already exist in the project.",
    )
    parser.add_argument(
        "--skip-history",
        action="store_true",
        help="Do not import legacy annotation history snapshots.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show SQLAlchemy query logging.",
    )
    args = parser.parse_args()

    _configure_logging(verbose=args.verbose)
    stats = asyncio.run(run_seed(force=args.force, import_history=not args.skip_history))
    _print_summary(stats)


if __name__ == "__main__":
    main()
