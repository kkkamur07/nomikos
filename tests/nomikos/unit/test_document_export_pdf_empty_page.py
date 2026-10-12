"""A PDF page with no transcribed line says so, instead of printing blank."""

from __future__ import annotations

import io

from pypdf import PdfReader

from backend.document.application.document_export_pdf import (
    DocumentTranscriptionPdfService,
    _PdfPage,
)


def test_a_page_with_no_lines_carries_the_note() -> None:
    service = DocumentTranscriptionPdfService.__new__(DocumentTranscriptionPdfService)

    content = service._render_document_pdf([_PdfPage(width=800, height=1000, lines=[])])

    assert PdfReader(io.BytesIO(content)).pages[0].extract_text() == "(no transcribed lines)"
