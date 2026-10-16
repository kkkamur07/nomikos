#!/usr/bin/env python3
"""Database steps of the annotated-corpus dev seed (library, needs a session).

Account, project, document and history-row handling used by ``run_seed``:
everything here takes an explicit session, so the entry script owns
bootstrapping and orchestration. Page-level cutting lives in ``seed._pages``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from scripts.platform.seed._fixtures import (
    ImportStats,
    _resolve_dev_password,
    _stem_from_image_key,
)
from scripts.platform.seed._pages import _import_page_part, _import_part_history

log = logging.getLogger("seed_dev_annotated_data")

PROJECT_SLUG = os.environ.get("DEV_ANNOTATED_PROJECT_SLUG", "byzantine-greek-manuscripts")
PROJECT_NAME = os.environ.get("DEV_ANNOTATED_PROJECT_NAME", "Byzantine Greek Manuscripts")
PROJECT_GUIDELINES = os.environ.get(
    "DEV_ANNOTATED_PROJECT_GUIDELINES",
    "This is annotated data given by professor chitwood for byzantine greek.",
)


async def _ensure_dev_user(session: AsyncSession):
    from backend.dev.bootstrap import DEV_EMAIL, DEV_USERNAME
    from backend.users.application.auth_service import AuthService
    from backend.users.infrastructure.orm_models import User

    password, generated = _resolve_dev_password()
    auth = AuthService()
    user, _token = await auth.register_if_absent(
        session,
        email=DEV_EMAIL,
        username=DEV_USERNAME,
        password=password,
    )
    if user is None:
        result = await session.execute(select(User).where(User.email == DEV_EMAIL))
        return result.scalar_one()
    if generated:
        # Only reachable when this run created the account; the plaintext is gone
        # after this line, so print it rather than leave an unloggable-into user.
        log.warning("Created %s with a generated password: %s", DEV_EMAIL, password)
    return user


async def _ensure_project(session: AsyncSession, user):
    from backend.project.application.project_service import ProjectService
    from backend.project.infrastructure.orm_models import Project

    result = await session.execute(select(Project).where(Project.slug == PROJECT_SLUG))
    project = result.scalar_one_or_none()
    if project is not None:
        return project
    return await ProjectService().create_project(
        session,
        user,
        name=PROJECT_NAME,
        slug=PROJECT_SLUG,
        guidelines=PROJECT_GUIDELINES,
    )


async def _find_document_by_name(session: AsyncSession, project_id: UUID, name: str):
    from backend.document.infrastructure.orm_models import Document

    result = await session.execute(
        select(Document).where(Document.project_id == project_id, Document.name == name)
    )
    return result.scalar_one_or_none()


async def _part_has_history(session: AsyncSession, part_id: UUID) -> bool:
    from backend.annotation.infrastructure.orm_models import AnnotationHistorySnapshot

    result = await session.execute(
        select(func.count())
        .select_from(AnnotationHistorySnapshot)
        .where(AnnotationHistorySnapshot.part_id == part_id)
    )
    return int(result.scalar_one()) > 0


async def _backfill_history(
    session: AsyncSession,
    *,
    document,
    page_stems: list[str],
    data_root: Path,
    import_history: bool,
    stats: ImportStats,
) -> None:
    if not import_history:
        return

    history_dir = data_root / "annotations" / "history"
    parts_by_stem = {
        stem.lower(): part
        for part in document.parts
        if (stem := _stem_from_image_key(part.image_key)) is not None
    }

    for page_stem in page_stems:
        part = parts_by_stem.get(page_stem.lower())
        if part is None:
            continue
        if await _part_has_history(session, part.id):
            continue
        history_count = await _import_part_history(
            session, part_id=part.id, history_dir=history_dir, page_stem=page_stem
        )
        if history_count:
            stats.history_snapshots += history_count
            log.info("  + backfilled %d history snapshot(s) for %s", history_count, page_stem)


async def _import_document(
    *,
    session: AsyncSession,
    user,
    project,
    doc_name: str,
    page_stems: list[str],
    data_root: Path,
    document_service,
    media,
    force: bool,
    import_history: bool,
) -> ImportStats:
    stats = ImportStats()
    existing = await _find_document_by_name(session, project.id, doc_name)

    if existing is not None and not force:
        stats.documents_skipped = 1
        await session.refresh(existing, attribute_names=["parts"])
        await _backfill_history(
            session,
            document=existing,
            page_stems=page_stems,
            data_root=data_root,
            import_history=import_history,
            stats=stats,
        )
        return stats

    if existing is not None and force:
        for part in existing.parts:
            media.delete(part.image_key)
        await session.delete(existing)
        await session.commit()

    document = await document_service.create_document(session, user, project.id, name=doc_name)
    stats.documents_created = 1

    for page_stem in page_stems:
        await _import_page_part(
            session=session,
            user=user,
            project=project,
            document=document,
            page_stem=page_stem,
            data_root=data_root,
            document_service=document_service,
            import_history=import_history,
            stats=stats,
        )

    return stats
