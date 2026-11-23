"""A transcription PDF sizes a whole page in about the time it takes to shape it."""

from __future__ import annotations

import time

from backend.annotation.application.transcription_pdf_service import (
    TranscriptionPdfService,
    _PdfLine,
)

_GREEK = "καὶ τὸν νόμον τῶν ἀνθρώπων ἐπὶ τῆς γῆς ταῦτα λέγει ὁ κύριος"


def test_a_seventy_line_page_renders_in_seconds_not_tens_of_seconds():
    # Sizing once rasterised every line at each candidate size: 13 s for 51 lines.
    lines = [
        _PdfLine(text=_GREEK, points=[[20, top], [580, top], [580, top + 18], [20, top + 18]])
        for top in range(10, 1410, 20)
    ]
    started = time.perf_counter()
    TranscriptionPdfService()._render_pdf(width=600, height=1420, lines=lines)
    assert time.perf_counter() - started < 3
