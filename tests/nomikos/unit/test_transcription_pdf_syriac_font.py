"""A Syriac line embeds the Eastern style, not the Estrangela base family."""

from __future__ import annotations

from backend.annotation.application.transcription_pdf_service import (
    TranscriptionPdfService,
    _PdfLine,
)


def _render(*texts: str) -> bytes:
    lines = [
        _PdfLine(
            text=text,
            points=[
                [20, 20 + 70 * index],
                [580, 20 + 70 * index],
                [580, 70 + 70 * index],
                [20, 70 + 70 * index],
            ],
        )
        for index, text in enumerate(texts)
    ]
    return TranscriptionPdfService()._render_pdf(width=600, height=400, lines=lines)


def test_syriac_line_embeds_the_eastern_font():
    content = _render("ܫܠܡܐ ܥܠܝܟܘܢ ܘܥܠ ܟܠܗܘܢ ܐܚܝ̈ܢ")
    assert b"NotoSansSyriacEastern" in content
