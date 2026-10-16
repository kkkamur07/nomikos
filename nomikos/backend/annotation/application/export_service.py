"""Stateless Export workflow for approved Line artifacts."""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from uuid import UUID

from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession

from backend.annotation.application.processing import apply_step
from backend.document.application.document_service import DocumentService
from backend.document.infrastructure.document_repository import DocumentRepository
from backend.document.infrastructure.media_store import MediaStore, get_media_store
from backend.document.infrastructure.media_store.encoding import bounded_image
from backend.document.infrastructure.orm_models import (
    Line,
    TranscriptionKind,
)
from backend.users.infrastructure.orm_models import User


@dataclass(frozen=True)
class ExportWarnings:
    unpaired_segments: list[int]
    unused_text_lines: list[int]


@dataclass(frozen=True)
class ExportArtifact:
    line_id: UUID
    segment_number: int
    image_filename: str
    transcription_filename: str
    transcription_text: str
    image_base64: str


@dataclass(frozen=True)
class ExportResult:
    exported_count: int
    artifacts: list[ExportArtifact]
    warnings: ExportWarnings
    steps: list[str]


class AnnotationExportService:
    def __init__(
        self,
        *,
        documents: DocumentRepository | None = None,
        document_service: DocumentService | None = None,
        media: MediaStore | None = None,
    ) -> None:
        self._documents = documents or DocumentRepository()
        self._document_service = document_service or DocumentService()
        self._media = media or get_media_store()

    async def export_part(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        *,
        steps: list[str] | None = None,
    ) -> ExportResult:
        document = await self._document_service.get_document(session, user, project_id, document_id)
        part = await self._documents.get_part(session, part_id)
        if part is None or part.document_id != document.id:
            from backend.core.exceptions import NotFoundError

            raise NotFoundError("Part not found")

        page_stem = self._page_stem(part.image_key)
        export_steps = steps if steps is not None else ["rectify"]
        text_lines = await self._documents.list_page_transcription_lines(session, part.id)

        # Every attribute the renderer needs is read here, on the event loop: the ORM rows
        # belong to an async session that cannot be touched from a worker thread.
        pending: list[tuple[UUID, int, str, dict]] = []
        unpaired_segments: list[int] = []
        paired_text_orders: set[int] = set()
        order_by_line = {text_line.paired_line_id: text_line.order for text_line in text_lines}
        for line in await self._documents.list_part_lines(session, part.id):
            segment_number = line.order + 1
            text = self._ground_truth_text(line)
            if text is None:
                unpaired_segments.append(segment_number)
                continue
            text_order = order_by_line.get(line.id)
            if text_order is not None:
                paired_text_orders.add(text_order)
            segment = {"points": line.points, "kind": line.kind.value}
            pending.append((line.id, segment_number, text, segment))

        # Decode, rectify, JPEG-encode and base64 the whole page in one hop off the loop:
        # a manuscript-sized page costs hundreds of milliseconds per segment.
        images = await asyncio.to_thread(
            self._render_segment_images,
            part.image_key,
            [segment for *_, segment in pending],
            export_steps,
        )

        artifacts = [
            ExportArtifact(
                line_id=line_id,
                segment_number=segment_number,
                image_filename=f"{page_stem}_{segment_number}.jpg",
                transcription_filename=f"{page_stem}_{segment_number}.txt",
                transcription_text=text,
                image_base64=image_base64,
            )
            for (line_id, segment_number, text, _), image_base64 in zip(
                pending, images, strict=True
            )
        ]

        unused_text_lines = [
            text_line.order for text_line in text_lines if text_line.order not in paired_text_orders
        ]
        return ExportResult(
            exported_count=len(artifacts),
            artifacts=artifacts,
            warnings=ExportWarnings(
                unpaired_segments=unpaired_segments,
                unused_text_lines=unused_text_lines,
            ),
            steps=export_steps,
        )

    def _render_segment_images(
        self,
        image_key: str,
        segments: list[dict],
        steps: list[str],
    ) -> list[str]:
        """Blocking: reads the blob, decodes it once, and renders every segment from it."""
        source_image = self._load_page_image(image_key)
        try:
            return [
                self._processed_image_base64(source_image, segment, steps) for segment in segments
            ]
        finally:
            source_image.close()

    def _load_page_image(self, image_key: str) -> Image.Image:
        raw = self._media.read(image_key)
        with bounded_image(raw) as pil:
            return pil.convert("RGB")

    def _page_stem(self, image_key: str) -> str:
        return Path(image_key).stem

    def _ground_truth_text(self, line: Line) -> str | None:
        for transcription in line.transcriptions:
            if (
                transcription.transcription.kind == TranscriptionKind.ground_truth
                and transcription.text.strip()
            ):
                # The label a trainer fits verbatim: no edge whitespace (ADR 0007).
                return transcription.text.strip()
        return None

    def _processed_image_base64(
        self,
        source_image: Image.Image,
        segment: dict,
        steps: list[str],
    ) -> str:
        image = source_image
        for step in steps:
            image = apply_step(image, segment, step)
        buf = BytesIO()
        dpi = source_image.info.get("dpi")
        if isinstance(dpi, tuple) and len(dpi) >= 2:
            image.save(
                buf,
                format="JPEG",
                quality=100,
                subsampling=0,
                optimize=False,
                dpi=(int(round(dpi[0])), int(round(dpi[1]))),
            )
        else:
            image.save(buf, format="JPEG", quality=100, subsampling=0, optimize=False)
        return base64.b64encode(buf.getvalue()).decode("ascii")
