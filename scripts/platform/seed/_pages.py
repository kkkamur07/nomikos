#!/usr/bin/env python3
"""Line and part import for the annotated-corpus dev seed (library, needs a session).

Page-level import steps used by ``seed._db``: the segment-to-line shaping,
the transcription pairing, the legacy history conversion and the per-page
part import. Everything here takes an explicit session and service, so the
entry script owns bootstrapping and orchestration.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from backend.document.infrastructure.orm_models import DocumentPart
from scripts.platform.seed._fixtures import (
    ImportStats,
    SeedAnnotation,
    SeedSegment,
    SnapshotState,
    _approved_text,
    _image_dimensions,
    _legacy_snapshot_state,
    _list_legacy_history_files,
    _load_annotation,
    _load_json,
    _load_page_transcription,
    _parse_legacy_timestamp,
    _resolve_page_image,
)

log = logging.getLogger("seed_dev_annotated_data")


def _segment_to_line_payload(segment: SeedSegment) -> dict[str, object]:
    from backend.document.infrastructure.orm_models import LineGeometryKind, LineSource

    payload: dict[str, object] = {
        "order": int(segment.get("number", 1)) - 1,
        "kind": LineGeometryKind(segment.get("kind", "polygon")),
        "points": segment["points"],
        "source": LineSource(segment.get("source", "manual")),
        "source_metadata": segment.get("source_metadata"),
        "kraken_ceiling": segment.get("kraken_ceiling"),
        "approved_text": _approved_text(segment),
    }
    if segment.get("baseline") is not None:
        payload["baseline"] = segment["baseline"]
    if segment.get("mask") is not None:
        payload["mask"] = segment["mask"]
    return payload


def _snapshot_counts(state: SnapshotState) -> tuple[int, int]:
    """Line count plus the paired (approved-text) line count for one snapshot."""
    lines = state["lines"]
    paired = 0
    for line in lines:
        approved = line.get("approved_text")
        if isinstance(approved, str) and approved.strip():
            paired += 1
    return len(lines), paired


async def _import_part_history(
    session: AsyncSession,
    *,
    part_id: UUID,
    history_dir: Path,
    page_stem: str,
) -> int:
    """Convert legacy on-disk history snapshots into platform AnnotationHistorySnapshot rows."""
    from backend.annotation.infrastructure.orm_models import AnnotationHistorySnapshot

    snapshot_files = _list_legacy_history_files(history_dir, page_stem)
    if not snapshot_files:
        return 0

    imported = 0
    for path in snapshot_files:
        payload = _load_json(path)
        annotation = payload.get("annotation")
        if not isinstance(annotation, dict):
            continue

        state = _legacy_snapshot_state(cast("SeedAnnotation", annotation))

        line_count, paired_line_count = _snapshot_counts(state)
        snapshot = AnnotationHistorySnapshot(
            part_id=part_id,
            state=state,
            line_count=line_count,
            paired_line_count=paired_line_count,
        )
        timestamp = payload.get("timestamp")
        created_at = _parse_legacy_timestamp(timestamp if isinstance(timestamp, str) else None)
        if created_at is not None:
            snapshot.created_at = created_at
        session.add(snapshot)
        imported += 1

    if imported:
        await session.commit()
    return imported


async def _pair_indexed_segments(
    document_service,
    session: AsyncSession,
    user,
    project,
    document,
    part,
    segments: list[SeedSegment],
) -> None:
    indexed = [segment for segment in segments if segment.get("paired_text_line_index") is not None]
    if not indexed:
        return

    lines = await document_service.list_part_lines(session, user, project.id, document.id, part.id)
    lines_by_order = {line.order: line for line in lines}
    for segment in indexed:
        order = int(segment.get("number", 1)) - 1
        line = lines_by_order.get(order)
        if line is None:
            continue
        await document_service.pair_page_text_line(
            session,
            user,
            project.id,
            document.id,
            part.id,
            line_id=line.id,
            text_line_order=int(segment["paired_text_line_index"]),
        )


async def _import_page_part(
    *,
    session: AsyncSession,
    user,
    project,
    document,
    page_stem: str,
    data_root: Path,
    document_service,
    import_history: bool,
    stats: ImportStats,
) -> DocumentPart | None:

    pages_dir = data_root / "manuscripts" / "pages"
    annotations_dir = data_root / "annotations" / "pages"
    transcriptions_dir = data_root / "transcriptions" / "pages"
    history_dir = data_root / "annotations" / "history"

    image_path = _resolve_page_image(pages_dir, page_stem)
    if image_path is None:
        stats.warnings.append(f"{page_stem}: no image in {pages_dir}")
        log.warning("  ! skipping %s: no image file", page_stem)
        return None

    annotation = _load_annotation(annotations_dir, page_stem)
    if annotation is None:
        stats.warnings.append(f"{page_stem}: no annotation JSON")
        log.warning("  ! skipping %s: no annotation JSON", page_stem)
        return None

    image_bytes = image_path.read_bytes()
    part = await document_service.upload_part(
        session,
        user,
        project.id,
        document.id,
        data=image_bytes,
        filename=image_path.name,
    )
    width, height = _image_dimensions(image_bytes)
    if width is not None and height is not None:
        part.width = width
        part.height = height
        await session.commit()
        await session.refresh(part)

    segments = annotation.get("segments") or []
    if segments:
        await document_service.replace_part_lines(
            session,
            user,
            project.id,
            document.id,
            part.id,
            lines=[_segment_to_line_payload(segment) for segment in segments],
        )
        stats.segments += len(segments)

    page_transcription = _load_page_transcription(transcriptions_dir, page_stem)
    if page_transcription is not None:
        await document_service.import_page_transcription(
            session,
            user,
            project.id,
            document.id,
            part.id,
            text=page_transcription,
        )
        await _pair_indexed_segments(
            document_service, session, user, project, document, part, segments
        )

    history_count = 0
    if import_history:
        history_count = await _import_part_history(
            session, part_id=part.id, history_dir=history_dir, page_stem=page_stem
        )
        stats.history_snapshots += history_count

    paired = sum(
        1
        for segment in segments
        if segment.get("text_override") or segment.get("paired_text_line_index") is not None
    )
    log.info(
        "  + part %s: %d segments%s%s",
        page_stem,
        len(segments),
        f", {paired} paired" if paired else "",
        f", {history_count} history snapshot(s)" if history_count else "",
    )
    stats.parts += 1
    return part
