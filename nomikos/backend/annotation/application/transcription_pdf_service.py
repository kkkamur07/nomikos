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
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from fpdf import FPDF
from sqlalchemy.ext.asyncio import AsyncSession

# Imported for its side effect: fpdf2 must keep combining marks with their letters.
import backend.core.fpdf_marks  # noqa: F401
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
# Every line's ink, approved or not: researchers leave most lines unapproved, and an
# export that greyed them out read as a page of drafts.
_TEXT_FILL = (0.1, 0.1, 0.35)
# The range a page's text size is chosen within, in points.
_MIN_FONT_SIZE = 8
_MAX_FONT_SIZE = 48
# A page's text size as a share of the distance between consecutive baselines: room for
# ascenders and descenders without touching the next line.
_LINE_GAP_FONT_SHARE = 0.7
# How much of its baseline's length a line may fill before it is drawn smaller.
_BASELINE_WIDTH_SHARE = 0.98
# A baseline that climbs more than this many line gaps is a malformed polyline, not one
# line's baseline (some segmenters wrap it round the polygon).
_MAX_BASELINE_RISE_IN_GAPS = 1.5
# A baseline spanning less of its box than this is a stub under part of the line, not the
# line's baseline.
_MIN_BASELINE_SHARE_OF_BOX = 0.8
# The least room kept between any text and the page's left and right edges, in points.
_PAGE_EDGE_MARGIN = 4


@dataclass(frozen=True)
class _PdfLine:
    """A line reduced to the values the renderer needs.

    ORM rows never cross the worker-thread boundary: touching an expired attribute there
    would drive the async session from a thread that has no greenlet context.
    """

    text: str
    points: list[list[float]]
    approved: bool = True
    baseline: list[list[float]] | None = None


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
        model's output renders the model's output, because a page that has been
        transcribed but not yet reviewed used to come back as a blank PDF with no error.
        Both are drawn alike. A line with no text of either kind is skipped.
        """
        prepared: list[_PdfLine] = []
        for line in lines:
            text = self._ground_truth_text(line)
            # Read leniently: callers that stub a line with only text and points still
            # render, falling back to the line's box.
            baseline = _baseline_points(getattr(line, "baseline", None))
            if text is not None:
                prepared.append(
                    _PdfLine(text=text, points=line.points, approved=True, baseline=baseline)
                )
                continue
            text = self._model_text(line)
            if text is not None:
                prepared.append(
                    _PdfLine(text=text, points=line.points, approved=False, baseline=baseline)
                )
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


def _baseline_points(baseline: object) -> list[list[float]] | None:
    """The points of a stored ``{"points": [[x, y], ...]}`` baseline, or None."""
    if not isinstance(baseline, dict):
        return None
    points = baseline.get("points")
    if not isinstance(points, list):
        return None
    return [list(point[:2]) for point in points if isinstance(point, list | tuple)]


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

    Each line's text sits on the line's own baseline, and the page uses one size, set by
    how far apart its lines are. Fitting every line to its polygon instead made sizes
    jump from 8 to 48 points down one page and stacked lines on top of each other,
    because segmenter polygons overlap their neighbours and some are malformed; the
    baseline is the part of the geometry that reliably says where the writing is. A page
    whose baselines cannot be trusted still gets one size, from the pitch of its boxes.
    """
    pdf.add_page(format=(width, height))
    pdf.set_fill_color(*_rgb(_PAGE_FILL))
    pdf.rect(0, 0, width, height, style="F")
    gap = _line_gap(lines)
    if gap is None:
        return
    placed = [_place(pdf, line, gap=gap, page_width=width) for line in lines]
    page_size = _page_font_size(gap, [fit for _, fit in placed])
    for line, (place, fit) in zip(lines, placed, strict=True):
        _set_line_font(pdf, line)
        size = min(page_size, fit)
        left, span, baseline_y, rtl = place
        if baseline_y is None:
            # No trusted baseline: the box bottom, raised by the descent (see _place).
            baseline_y = _box_bottom(line) - abs(pdf.current_font.desc.descent) / 1000 * size
        pdf.set_font_size(size)
        # A cell of one font size high puts the baseline 0.8 em below its top.
        pdf.set_xy(left, baseline_y - 0.8 * size)
        # A right-to-left line starts at the right end of its baseline, as on the page.
        pdf.cell(w=span, h=size, text=line.text, align="R" if rtl else "L")


def _set_line_font(pdf: _TranscriptionPdf, line: _PdfLine) -> bool:
    """Sets the font, fallback, shaping and colour ``line`` is drawn in; True for RTL.

    Set before sizing so the line is measured as it will be drawn: shaped by HarfBuzz,
    with the base font filling in missing glyphs.
    """
    script = detect_script(line.text)
    rtl = is_rtl(script)
    family = pdf.family_for(script)
    pdf.set_font(family, size=_MAX_FONT_SIZE)
    pdf.set_fallback_fonts([] if family == _BASE_FAMILY else [_BASE_FAMILY])
    pdf.set_text_shaping(use_shaping_engine=True, direction="rtl" if rtl else "ltr")
    pdf.set_text_color(*_rgb(_TEXT_FILL))
    return rtl


def _place(
    pdf: _TranscriptionPdf, line: _PdfLine, *, gap: float, page_width: float
) -> tuple[tuple[float, float, float | None, bool], float]:
    """Where ``line`` goes and the largest size at which it fits there.

    Returns ``(left, span, baseline_y, rtl)`` and that size. ``baseline_y`` is None when
    the baseline cannot be trusted (see :func:`_baseline_anchor`): the text then stands
    on the bottom of the line's box. The span is cut to the page, so no text runs off
    either edge, and the size has no floor: clipped words are worse than small ones.
    """
    rtl = _set_line_font(pdf, line)
    anchor = _baseline_anchor(line, gap)
    baseline_y: float | None
    if anchor is None:
        left, _, right, _ = _line_bbox(line.points)
        baseline_y = None
    else:
        left, right, baseline_y = anchor
    left = max(left, _PAGE_EDGE_MARGIN)
    right = min(right, page_width - _PAGE_EDGE_MARGIN)
    span = max(right - left, 1)
    # The cell pads its text by c_margin on the side it is aligned to (and fpdf2 keeps
    # the same room on the other), so that padding is not room for glyphs.
    room = max(span * _BASELINE_WIDTH_SHARE - 2 * pdf.c_margin, 1)
    unit_width = pdf.get_string_width(line.text) / pdf.font_size_pt
    # Tenths of a point, rounded down, so a line drawn at its fit never ends past ``room``.
    fit = math.floor(room / unit_width * 10) / 10 if unit_width > 0 else float(_MAX_FONT_SIZE)
    return (left, span, baseline_y, rtl), max(0.1, fit)


def _box_bottom(line: _PdfLine) -> float:
    return _line_bbox(line.points)[3]


def _usable_baseline(line: _PdfLine) -> list[list[float]] | None:
    """``line``'s baseline when it describes the whole line, else None.

    A hand-drawn line stores its polygon as its baseline (see ``page_xml_geometry``),
    which says nothing about where the text sits inside the box. Some segmenters store a
    stub: two equal points, or a piece under some of its words, well short of the box.
    Its height may be right but its ends are not the line's, so it is not used either.
    """
    baseline = line.baseline
    if not baseline or len(baseline) < 2 or baseline == line.points:
        return None
    xs = [point[0] for point in baseline]
    box_left, _, box_right, _ = _line_bbox(line.points) if line.points else (0, 0, 0, 0)
    if max(xs) - min(xs) < (box_right - box_left) * _MIN_BASELINE_SHARE_OF_BOX:
        return None
    return baseline


def _baseline_y(baseline: list[list[float]]) -> float:
    """The height a baseline's text sits at: the mean of its points."""
    return statistics.fmean(point[1] for point in baseline)


def _line_gap(lines: list[_PdfLine]) -> float | None:
    """The typical distance between consecutive lines in reading order, or None.

    Measured between baselines when at least half the lines (and two or more) have a
    usable one. Otherwise measured between the vertical centres of the lines'
    boxes, which overlap but are spaced like the writing; a page that gives no gap
    either way (one line, or one row) falls back to its median box height. None only
    for a page with no lines at all.
    """
    boxes = [_line_bbox(line.points) for line in lines if line.points]
    if not boxes:
        return None
    heights = [
        _baseline_y(baseline) if (baseline := _usable_baseline(line)) else None for line in lines
    ]
    usable = sum(height is not None for height in heights)
    if usable >= 2 and usable * 2 >= len(lines):
        gap = _typical_step(heights)
        if gap is not None:
            return gap
    gap = _typical_step([(top + bottom) / 2 for _, top, _, bottom in boxes])
    if gap is not None:
        return gap
    return max(statistics.median(bottom - top for _, top, _, bottom in boxes), 1)


def _typical_step(heights: Sequence[float | None]) -> float | None:
    """The median step down between consecutive ``heights``, or None without one.

    A step needs both of its heights: a gap measured across a line with no height would
    span two lines. Only steps down the page count: the jump back up to the top of the next column is
    not a line gap. Steps under half or over twice the first median (a line missed by
    the segmenter, two baselines on one line of writing) are dropped before the final
    median is taken.
    """
    steps = [
        lower - upper
        for upper, lower in zip(heights, heights[1:], strict=False)
        if upper is not None and lower is not None and lower > upper
    ]
    if not steps:
        return None
    rough = statistics.median(steps)
    typical = [step for step in steps if rough / 2 <= step <= rough * 2]
    return statistics.median(typical)


def _page_font_size(gap: float, fits: list[float]) -> float:
    """The one size a page's lines are drawn in, unless a line must shrink to fit.

    ``_LINE_GAP_FONT_SHARE`` of the line gap, clamped to the size range, and no larger
    than the size at which three lines in four fit their width: where a page's text runs
    wider than its boxes (a line's box drawn round only part of the writing, a font
    wider than the hand), sizing by the gap alone would leave most lines each shrunk to
    a size of its own, and the page would not read as one size.
    """
    size = max(_MIN_FONT_SIZE, min(_MAX_FONT_SIZE, math.floor(gap * _LINE_GAP_FONT_SHARE)))
    if len(fits) >= 4:
        lower_quartile = statistics.quantiles(fits, n=4)[0]
        size = min(size, max(_MIN_FONT_SIZE, math.floor(lower_quartile)))
    return size


def _baseline_anchor(line: _PdfLine, gap: float) -> tuple[float, float, float] | None:
    """Left end, right end and height of ``line``'s baseline, or None when it is malformed.

    Malformed is a baseline :func:`_usable_baseline` rejects, or a polyline that climbs
    more than ``_MAX_BASELINE_RISE_IN_GAPS`` line gaps: that is a trace round the line's
    polygon, not the line its letters stand on.
    """
    baseline = _usable_baseline(line)
    if baseline is None:
        return None
    xs = [point[0] for point in baseline]
    ys = [point[1] for point in baseline]
    if max(ys) - min(ys) > gap * _MAX_BASELINE_RISE_IN_GAPS or max(xs) - min(xs) < 1:
        return None
    return min(xs), max(xs), _baseline_y(baseline)


def _rgb(fill: tuple[float, float, float]) -> tuple[int, int, int]:
    red, green, blue = (round(channel * 255) for channel in fill)
    return red, green, blue


def _line_bbox(points: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)
