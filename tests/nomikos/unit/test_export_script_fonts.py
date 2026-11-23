"""Downloads use the document's saved font choice per script, not the defaults only."""

from __future__ import annotations

import io
import uuid
from types import SimpleNamespace

import pytest
from docx import Document as open_word_document
from docx.oxml.ns import qn
from pypdf import PdfReader

from backend.annotation.application.transcription_pdf_service import TranscriptionPdfService
from backend.document.application.document_export_docx import (
    DocumentDocxExportService,
    DocxPage,
    render_docx,
)
from backend.document.application.document_export_pdf import DocumentTranscriptionPdfService
from backend.document.application.document_export_selection import DocumentExportSelection
from backend.document.infrastructure.orm_models import (
    Document,
    DocumentPart,
    Line,
    LineTranscription,
    Transcription,
    TranscriptionKind,
)

_SYRIAC = "ܫܠܡܐ ܥܠܝܟܘܢ ܘܥܠ ܟܠܗܘܢ ܐܚܝ̈ܢ"
_SYRIAC_STYLES = {"NotoSansSyriac", "NotoSansSyriacEastern", "NotoSansSyriacWestern"}


def _document(script_fonts: dict[str, str] | None) -> Document:
    return Document(
        id=uuid.uuid4(), project_id=uuid.uuid4(), name="Codex", script_fonts=script_fonts
    )


def _part(document: Document) -> DocumentPart:
    return DocumentPart(id=uuid.uuid4(), document_id=document.id, order=0, width=600, height=200)


def _line() -> Line:
    line = Line(
        baseline={"points": [[20, 60], [580, 60]]},
        points=[[20, 20], [580, 20], [580, 70], [20, 70]],
    )
    layer = Transcription(name="gt", kind=TranscriptionKind.ground_truth)
    line.transcriptions.append(LineTranscription(transcription=layer, text=_SYRIAC))
    return line


class _Repository:
    def __init__(self, part: DocumentPart) -> None:
        self._part = part

    async def get_part(self, _session, part_id):
        return self._part if part_id == self._part.id else None

    async def list_part_lines(self, _session, _part_id):
        return [_line()]

    async def list_part_blocks(self, _session, _part_id):
        return []

    async def part_page_number(self, _session, _part):
        return 1


class _DocumentService:
    def __init__(self, document: Document, part: DocumentPart) -> None:
        self._document = document
        self._part = part

    async def get_document(self, *_args, **_kwargs):
        return self._document

    async def get_published_part_context(self, *_args, **_kwargs):
        return SimpleNamespace(project=None, document=self._document, part=self._part)

    async def backfill_part_dimensions(self, *_args, **_kwargs):
        return None


class _Selector:
    def __init__(self, document: Document, part: DocumentPart) -> None:
        self._selection = DocumentExportSelection(
            document=document, parts=[part], page_numbers={part.id: 1}
        )

    async def select(self, *_args, **_kwargs):
        return self._selection


def _syriac_fonts(content: bytes) -> set[str]:
    names: set[str] = set()
    for page in PdfReader(io.BytesIO(content)).pages:
        fonts = page["/Resources"]["/Font"].values()
        names |= {str(font["/BaseFont"]).split("+")[-1] for font in fonts}
    return names & _SYRIAC_STYLES


def _fixtures(script_fonts):
    document = _document(script_fonts)
    part = _part(document)
    return document, part, _Repository(part), _DocumentService(document, part)


async def _single_page(script_fonts) -> bytes:
    document, part, repository, service = _fixtures(script_fonts)
    pdf = TranscriptionPdfService(documents=repository, document_service=service)
    return await pdf.generate_part_pdf(
        object(), object(), document.project_id, document.id, part.id
    )


async def _public_page(script_fonts) -> bytes:
    document, part, repository, service = _fixtures(script_fonts)
    pdf = TranscriptionPdfService(documents=repository, document_service=service)
    return await pdf.generate_part_pdf_public(object(), document.project_id, document.id, part.id)


async def _chapter(script_fonts) -> bytes:
    document, part, repository, service = _fixtures(script_fonts)
    pdf = DocumentTranscriptionPdfService(
        documents=repository, document_service=service, selector=_Selector(document, part)
    )
    export = await pdf.generate_document_pdf(object(), object(), document.project_id, document.id)
    return export.content


_PDF_ROUTES = [_single_page, _public_page, _chapter]


@pytest.mark.asyncio
@pytest.mark.parametrize("render", _PDF_ROUTES, ids=["page", "public", "chapter"])
async def test_pdf_embeds_the_documents_chosen_syriac_style(render) -> None:
    assert _syriac_fonts(await render({"syriac": "eastern"})) == {"NotoSansSyriacEastern"}


@pytest.mark.asyncio
@pytest.mark.parametrize("render", _PDF_ROUTES, ids=["page", "public", "chapter"])
@pytest.mark.parametrize("script_fonts", [None, {}, {"greek": "noto-sans"}])
async def test_pdf_without_a_syriac_choice_stays_estrangela(render, script_fonts) -> None:
    assert _syriac_fonts(await render(script_fonts)) == {"NotoSansSyriac"}


def _cs_fonts(content: bytes) -> list[str]:
    word = open_word_document(io.BytesIO(content))
    return [
        paragraph.runs[0]._r.rPr.find(qn("w:rFonts")).get(qn("w:cs"))
        for paragraph in word.paragraphs
        if paragraph.text == _SYRIAC
    ]


def test_word_names_the_chosen_syriac_style() -> None:
    pages = [DocxPage(number=1, lines=[_SYRIAC])]
    assert _cs_fonts(render_docx(pages, {"syriac": "western"})) == ["Noto Sans Syriac Western"]
    assert _cs_fonts(render_docx(pages)) == ["Noto Sans Syriac"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("script_fonts", "family"),
    [({"syriac": "eastern"}, "Noto Sans Syriac Eastern"), (None, "Noto Sans Syriac")],
)
async def test_word_exports_load_the_documents_choice(script_fonts, family) -> None:
    document, part, repository, service = _fixtures(script_fonts)
    word = DocumentDocxExportService(
        documents=repository, document_service=service, selector=_Selector(document, part)
    )
    ids = (document.project_id, document.id)
    whole = await word.export_document_docx(object(), object(), *ids)
    page = await word.export_part_docx(object(), object(), *ids, part.id)
    public = await word.export_part_docx_public(object(), *ids, part.id)
    for export in (whole, page, public):
        assert _cs_fonts(export.content) == [family]
