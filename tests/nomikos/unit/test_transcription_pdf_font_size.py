"""How large a transcription PDF draws a line's text: the largest whole size that fits."""

from __future__ import annotations

import time

from backend.annotation.application.transcription_pdf_service import (
    _BASE_FAMILY,
    TranscriptionPdfService,
    _fit_font_size,
    _PdfLine,
    _TranscriptionPdf,
)
from backend.document.application.script_detection import Script

_GREEK = "καὶ τὸν νόμον τῶν ἀνθρώπων ἐπὶ τῆς γῆς ταῦτα λέγει ὁ κύριος"


def _pdf() -> _TranscriptionPdf:
    pdf = _TranscriptionPdf()
    pdf.add_page(format=(1000, 1000))
    pdf.set_font(_BASE_FAMILY, size=48)
    pdf.set_text_shaping(use_shaping_engine=True, direction="ltr")
    return pdf


def _width_at(pdf: _TranscriptionPdf, text: str, size: int) -> float:
    pdf.set_font_size(size)
    return pdf.get_string_width(text)


def test_a_wide_line_takes_the_largest_size_whose_width_fits():
    pdf = _pdf()
    size = _fit_font_size(pdf, _GREEK, 400, 1000)
    assert 8 < size < 48
    assert _width_at(pdf, _GREEK, size) <= 400
    assert _width_at(pdf, _GREEK, size + 1) > 400


def test_a_short_box_limits_the_size_by_the_ink_height():
    pdf = _pdf()
    roomy = _fit_font_size(pdf, "Ἐν ἀρχῇ", 1000, 1000)
    short = _fit_font_size(pdf, "Ἐν ἀρχῇ", 1000, 20)
    assert roomy == 48
    assert 8 < short < 48


def test_the_size_stays_within_the_range():
    pdf = _pdf()
    assert _fit_font_size(pdf, _GREEK, 5, 5) == 8
    assert _fit_font_size(pdf, "a", 10_000, 10_000) == 48


def test_a_character_the_script_font_lacks_is_measured_in_the_base_font():
    pdf = _pdf()
    pdf.set_font(pdf.family_for(Script.syriac), size=48)
    pdf.set_fallback_fonts([_BASE_FAMILY])
    # Greek capitals only exist in the base font, and stand taller than a row of alaphs.
    assert _fit_font_size(pdf, "ܐܐܐ ΑΒΓ", 1000, 20) < _fit_font_size(pdf, "ܐܐܐ", 1000, 20)


def test_a_seventy_line_page_renders_in_seconds_not_tens_of_seconds():
    # Sizing once rasterised every line at each candidate size: 13 s for 51 lines.
    lines = [
        _PdfLine(text=_GREEK, points=[[20, top], [580, top], [580, top + 18], [20, top + 18]])
        for top in range(10, 1410, 20)
    ]
    started = time.perf_counter()
    TranscriptionPdfService()._render_pdf(width=600, height=1420, lines=lines)
    assert time.perf_counter() - started < 3
