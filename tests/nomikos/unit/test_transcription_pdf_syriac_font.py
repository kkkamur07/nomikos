"""A Syriac line embeds the default Estrangela style, not a regional one."""

from __future__ import annotations

import io

from pypdf import PdfReader

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


def test_syriac_line_embeds_the_estrangela_font():
    content = _render("ܫܠܡܐ ܥܠܝܟܘܢ ܘܥܠ ܟܠܗܘܢ ܐܚܝ̈ܢ")
    page = PdfReader(io.BytesIO(content)).pages[0]
    # Subset fonts are named ``ABCDEF+Family``. Compared exactly, because a substring
    # check for "NotoSansSyriac" would also match the Eastern and Western styles.
    names = {str(font["/BaseFont"]).split("+")[-1] for font in page["/Resources"]["/Font"].values()}
    assert "NotoSansSyriac" in names
    assert not names & {"NotoSansSyriacEastern", "NotoSansSyriacWestern"}
