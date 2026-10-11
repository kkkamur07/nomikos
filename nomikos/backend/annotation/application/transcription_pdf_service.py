"""Stateless Transcription PDF artifact generation.

Drawn with fpdf2 and shaped with HarfBuzz (ADR 0009). Every line is placed at its own
geometry, in the bundled font for the script its characters are in: Syriac joins and
runs right to left, Armenian and Coptic get a font that has their glyphs, and Greek and
Latin stay in Noto Sans. A line that mixes scripts falls back to Noto Sans for the
characters its own font lacks.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from uuid import UUID

from fpdf import FPDF
from fpdf.fonts import TTFFont
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.exceptions import NotFoundError, ValidationError
from backend.core.fonts import font_for_script, resolve_unicode_font
from backend.document.application.document_service import DocumentService
from backend.document.application.script_detection import Script, detect_script, is_rtl
from backend.document.infrastructure.document_repository import DocumentRepository
from backend.document.infrastructure.orm_models import (
    DocumentPart,
    Line,
    LineTranscription,
    TranscriptionKind,
)
from backend.users.infrastructure.orm_models import User

# The family every page falls back to for characters a script font does not carry.
_BASE_FAMILY = "nomikos-base"
_PAGE_FILL = (1.0, 1.0, 1.0)
_TEXT_FILL = (0.1, 0.1, 0.35)
# Machine text nobody has approved yet. Legible, but unmistakably not the ink colour of
# approved text: a reader must be able to tell a model's guess from a researcher's word.
_MODEL_TEXT_FILL = (0.55, 0.55, 0.6)
# The range a line's text is sized within to fit its box, in points.
_MIN_FONT_SIZE = 8
_MAX_FONT_SIZE = 48


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
        # fpdf2 shapes, lays out and compresses the page synchronously; on a single-worker
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
        pdf = _TranscriptionPdf()
        draw_page(pdf, width=width, height=height, lines=lines)
        return pdf.finish()

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


class _TranscriptionPdf(FPDF):
    """One PDF in points, with the script fonts loaded the first time a line needs one.

    Points, so a page's pixel grid maps one to one onto the PDF page as it did before.
    Fonts are loaded per document rather than once per process: fpdf2 keeps them on the
    instance, so there is no shared registry for two worker threads to race on, and a
    page with no Syriac on it never parses the Syriac font.
    """

    def __init__(self) -> None:
        super().__init__(unit="pt")
        self.set_auto_page_break(False)
        self.set_margin(0)
        # The two points of left padding the line has always been drawn with.
        self.c_margin = 2
        try:
            self.add_font(_BASE_FAMILY, fname=resolve_unicode_font())
        except RuntimeError as exc:
            raise ValidationError(str(exc)) from exc
        self._families: dict[Script, str] = {}

    def family_for(self, script: Script) -> str:
        if script not in self._families:
            path = font_for_script(script.value)
            family = f"nomikos-{script.value}"
            self.add_font(family, fname=path)
            self._families[script] = family
        return self._families[script]

    def finish(self) -> bytes:
        return bytes(self.output())


def draw_page(pdf: _TranscriptionPdf, *, width: int, height: int, lines: list[_PdfLine]) -> None:
    """One page of ``lines`` at their geometry, appended to ``pdf``.

    The single drawing routine both the per-page and the whole-document PDF use, so a
    page reads the same whichever endpoint produced it.
    """
    pdf.add_page(format=(width, height))
    pdf.set_fill_color(*_rgb(_PAGE_FILL))
    pdf.rect(0, 0, width, height, style="F")
    for line in lines:
        script = detect_script(line.text)
        rtl = is_rtl(script)
        family = pdf.family_for(script)
        x0, y0, x1, y1 = _line_bbox(line.points)
        box_w = max(x1 - x0, 1)
        box_h = max(y1 - y0, 1)
        # Font, fallback and shaping are set before sizing so the line is measured as it
        # will be drawn: shaped by HarfBuzz, with the base font filling in missing glyphs.
        pdf.set_font(family, size=_MAX_FONT_SIZE)
        pdf.set_fallback_fonts([] if family == _BASE_FAMILY else [_BASE_FAMILY])
        pdf.set_text_shaping(use_shaping_engine=True, direction="rtl" if rtl else "ltr")
        font_size = _fit_font_size(pdf, line.text, box_w * 0.95, box_h * 0.85)
        pdf.set_font_size(font_size)
        pdf.set_text_color(*_rgb(line.fill))
        # A cell of one font size high puts the baseline 0.8 em below its top, so the
        # top sits 0.2 em down to keep the baseline one em under the line's top edge.
        pdf.set_xy(x0, y0 + 0.2 * font_size)
        # A right-to-left line starts at the right edge of its box, as it does on the page.
        pdf.cell(w=box_w, h=font_size, text=line.text, align="R" if rtl else "L")


def _rgb(fill: tuple[float, float, float]) -> tuple[int, int, int]:
    red, green, blue = (round(channel * 255) for channel in fill)
    return red, green, blue


def _line_bbox(points: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def _fit_font_size(
    pdf: _TranscriptionPdf,
    text: str,
    max_width: float,
    max_height: float,
    *,
    min_size: int = _MIN_FONT_SIZE,
    max_size: int = _MAX_FONT_SIZE,
) -> int:
    """The largest whole size at which ``text`` fits the box, in the font ``pdf`` has set.

    Measured once, in the unhinted font units the PDF is drawn in, and scaled: width and
    ink height both grow linearly with the size. Rasterising the line at every candidate
    size instead cost about 0.2 ms a glyph (FreeType runs Noto Sans's hinting program on
    each one), which came to 13 seconds for a 50-line page.
    """
    unit_width = pdf.get_string_width(text) / pdf.font_size_pt
    unit_height = _ink_height(pdf, text)
    limits = [float(max_size)]
    if unit_width > 0:
        limits.append(max_width / unit_width)
    if unit_height > 0:
        limits.append(max_height / unit_height)
    return max(min_size, min(max_size, math.floor(min(limits))))


def _ink_height(pdf: _TranscriptionPdf, text: str) -> float:
    """The vertical ink extent of ``text`` at a size of one, from the glyph bounding boxes.

    A character the current font lacks is measured in the base font, which is where the
    fallback draws it from; one neither font has adds nothing.
    """
    fonts = [pdf.current_font, pdf.fonts.get(_BASE_FAMILY)]
    top = bottom = None
    for char in set(text):
        for font in fonts:
            bounds = _glyph_bounds(font, char) if font is not None else None
            if bounds is not None:
                low, high = bounds
                bottom = low if bottom is None else min(bottom, low)
                top = high if top is None else max(top, high)
                break
    if top is None or bottom is None:
        return 0.0
    return top - bottom


def _glyph_bounds(font: TTFFont, char: str) -> tuple[float, float] | None:
    """``char``'s ink from bottom to top in ems, or None when ``font`` has no glyph for it.

    An empty glyph, such as the space, has no ink and so no bounds either.
    """
    ttfont = font.ttfont
    name = ttfont.getBestCmap().get(ord(char))
    if name is None:
        return None
    glyph = ttfont["glyf"][name]
    if not hasattr(glyph, "yMax"):
        return None
    units = ttfont["head"].unitsPerEm
    return glyph.yMin / units, glyph.yMax / units
