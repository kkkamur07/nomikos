"""Stateless Transcription PDF artifact generation."""

from __future__ import annotations

import asyncio
import hashlib
import io
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from uuid import UUID

from PIL import ImageFont
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.exceptions import NotFoundError, ValidationError
from backend.core.fonts import resolve_unicode_font
from backend.document.application.document_service import DocumentService
from backend.document.infrastructure.document_repository import DocumentRepository
from backend.document.infrastructure.orm_models import (
    DocumentPart,
    Line,
    LineTranscription,
    TranscriptionKind,
)
from backend.users.infrastructure.orm_models import User

_FONT_NAME = "AnnotePlatformTranscriptionPdf"
# Rendering runs on worker threads, so two requests can reach the font
# registration at once; without the lock the second could draw with a font
# reportlab has not finished registering.
_font_lock = threading.Lock()
_PAGE_FILL = (1.0, 1.0, 1.0)
_TEXT_FILL = (0.1, 0.1, 0.35)
# Machine text nobody has approved yet. Legible, but unmistakably not the ink colour of
# approved text: a reader must be able to tell a model's guess from a researcher's word.
_MODEL_TEXT_FILL = (0.55, 0.55, 0.6)


@dataclass(frozen=True)
class _PdfLine:
    """A line reduced to the values the renderer needs.

    ORM rows never cross the worker-thread boundary: touching an expired attribute there
    would drive the async session from a thread that has no greenlet context.
    """

    text: str
    points: list[list[float]]
    approved: bool = True

    @property
    def fill(self) -> tuple[float, float, float]:
        return _TEXT_FILL if self.approved else _MODEL_TEXT_FILL


class TranscriptionPdfService:
    def __init__(
        self,
        *,
        documents: DocumentRepository | None = None,
        document_service: DocumentService | None = None,
    ) -> None:
        self._documents = documents or DocumentRepository()
        self._document_service = document_service or DocumentService()

    async def generate_part_pdf(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
    ) -> bytes:
        document = await self._document_service.get_document(session, user, project_id, document_id)
        part = await self._documents.get_part(session, part_id)
        if part is None or part.document_id != document.id:
            raise NotFoundError("Part not found")

        width, height = await self._page_size(session, part)
        lines = self._pdf_lines(await self._documents.list_part_lines(session, part.id))
        # reportlab lays out and compresses the page synchronously; on a single-worker
        # event loop that would stall every other in-flight request.
        return await asyncio.to_thread(self._render_pdf, width=width, height=height, lines=lines)

    async def generate_part_pdf_public(
        self,
        session: AsyncSession,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        *,
        token: str | None = None,
    ) -> bytes:
        part = await self._document_service.get_published_part(
            session, project_id, document_id, part_id, token=token
        )
        width, height = await self._page_size(session, part)
        lines = self._pdf_lines(await self._documents.list_part_lines(session, part.id))
        return await asyncio.to_thread(self._render_pdf, width=width, height=height, lines=lines)

    async def _page_size(self, session: AsyncSession, part: DocumentPart) -> tuple[int, int]:
        """Page dimensions come from the row, not from re-decoding the stored image.

        Parts uploaded before the dimensions were persisted are backfilled lazily by the
        document service, so at most one request per legacy part pays for a decode.
        """
        if part.width is None or part.height is None:
            await self._document_service.backfill_part_dimensions(session, [part])
        if part.width is None or part.height is None:
            raise NotFoundError("Part image not found")
        return part.width, part.height

    def _pdf_lines(self, lines: list[Line]) -> list[_PdfLine]:
        """Every line that has any text at all, approved text first.

        A line with ground truth renders that and nothing else. A line with only a
        model's output renders the model's output, marked unapproved, because a page
        that has been transcribed but not yet reviewed used to come back as a blank PDF
        with no error, and "blank" is not a way to say "nothing here is approved yet".
        A line with no text of either kind is skipped.
        """
        prepared: list[_PdfLine] = []
        for line in lines:
            text = self._ground_truth_text(line)
            if text is not None:
                prepared.append(_PdfLine(text=text, points=line.points, approved=True))
                continue
            text = self._model_text(line)
            if text is not None:
                prepared.append(_PdfLine(text=text, points=line.points, approved=False))
        return prepared

    def _render_pdf(self, *, width: int, height: int, lines: list[_PdfLine]) -> bytes:
        try:
            font_path = resolve_unicode_font()
        except RuntimeError as exc:
            raise ValidationError(str(exc)) from exc
        font_name = _ensure_font(font_path)
        buffer = io.BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=(width, height))
        pdf.setFillColorRGB(*_PAGE_FILL)
        pdf.rect(0, 0, width, height, fill=1, stroke=0)

        for line in lines:
            text = line.text
            x0, y0, x1, y1 = _line_bbox(line.points)
            box_w = max(x1 - x0, 1)
            box_h = max(y1 - y0, 1)
            font_size = _fit_font_size(text, font_path, box_w * 0.95, box_h * 0.85)
            pdf.setFont(font_name, font_size)
            pdf.setFillColorRGB(*line.fill)
            pdf.drawString(x0 + 2, height - y0 - font_size, text)

        pdf.showPage()
        pdf.save()
        return buffer.getvalue()

    def _ground_truth_text(self, line: Line) -> str | None:
        for transcription in line.transcriptions:
            if (
                transcription.transcription.kind == TranscriptionKind.ground_truth
                and transcription.text.strip()
            ):
                return transcription.text
        return None

    def _model_text(self, line: Line) -> str | None:
        """The newest non-empty model transcription on ``line``, or None."""
        candidates = [
            transcription
            for transcription in line.transcriptions
            if transcription.transcription.kind == TranscriptionKind.model
            and transcription.text.strip()
        ]
        if not candidates:
            return None
        newest = max(candidates, key=_layer_age)
        return newest.text


def _layer_age(transcription: LineTranscription) -> float:
    created_at = transcription.transcription.created_at
    return created_at.timestamp() if created_at is not None else 0.0


def _ensure_font(font_path: Path) -> str:
    with _font_lock:
        return _cached_font_name(str(font_path))


@lru_cache(maxsize=4)
def _cached_font_name(font_path: str) -> str:
    """Register once per path; the cache itself is the registered flag."""
    name = f"{_FONT_NAME}-{hashlib.sha1(font_path.encode(), usedforsecurity=False).hexdigest()[:8]}"
    pdfmetrics.registerFont(TTFont(name, font_path))
    return name


def _line_bbox(points: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


@lru_cache(maxsize=256)
def _cached_typeface(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(font_path, size=size)


def _fit_font_size(
    text: str,
    font_path: Path,
    max_width: float,
    max_height: float,
    *,
    min_size: int = 8,
    max_size: int = 48,
) -> int:
    # A smaller size fits whenever a larger one does, so this finds the same
    # answer as a linear scan with about six font loads instead of 41.
    path = str(font_path)
    best = min_size
    low, high = min_size, max_size
    while low <= high:
        mid = (low + high) // 2
        bbox = _cached_typeface(path, mid).getbbox(text)
        if bbox[2] - bbox[0] <= max_width and bbox[3] - bbox[1] <= max_height:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    return best
