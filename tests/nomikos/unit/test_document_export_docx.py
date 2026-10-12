"""The Word export: a heading per page, a paragraph per line, the line's script font."""

from __future__ import annotations

import io
import uuid

import pytest
from docx import Document as open_word_document
from docx.oxml.ns import qn

from backend.document.application.document_export_docx import (
    DocumentDocxExportService,
    DocxPage,
    render_docx,
)
from backend.document.application.document_export_selection import DocumentExportSelection
from backend.document.application.document_export_text import DocumentTextExportService
from backend.document.application.regions import RegionHeading
from backend.document.infrastructure.orm_models import (
    Block,
    Document,
    DocumentPart,
    Line,
    LineTranscription,
    Transcription,
    TranscriptionKind,
)

_SYRIAC = "ܫܠܡܐ ܥܠܝܟܘܢ"
_GREEK = "Ἐν ἀρχῇ ἦν ὁ λόγος"
_ARMENIAN = "Բարեւ աշխարհ"
_COPTIC = "ⲡⲛⲟⲩⲧⲉ ⲡⲉ"


def _line(text: str | None, kind: TranscriptionKind = TranscriptionKind.ground_truth) -> Line:
    line = Line(baseline={"points": []}, points=[[0, 0], [10, 0], [10, 5], [0, 5]])
    if text is not None:
        layer = Transcription(name=kind.value, kind=kind)
        line.transcriptions.append(LineTranscription(transcription=layer, text=text))
    return line


class _Selector:
    def __init__(self, selection: DocumentExportSelection) -> None:
        self._selection = selection

    async def select(self, *_args, reviewed_only: bool = False, **_kwargs):
        return self._selection


class _Repository:
    def __init__(
        self,
        lines: dict[uuid.UUID, list[Line]],
        blocks: dict[uuid.UUID, list[Block]] | None = None,
        *,
        parts: dict[uuid.UUID, DocumentPart] | None = None,
        page_numbers: dict[uuid.UUID, int] | None = None,
    ) -> None:
        self._lines = lines
        self._blocks = blocks or {}
        self._parts = parts or {}
        self._page_numbers = page_numbers or {}

    async def list_part_lines(self, _session, part_id):
        return list(self._lines[part_id])

    async def list_part_blocks(self, _session, part_id):
        return list(self._blocks.get(part_id, []))

    async def get_part(self, _session, part_id):
        return self._parts.get(part_id)

    async def part_page_number(self, _session, part):
        return self._page_numbers[part.id]


class _DocumentService:
    def __init__(self, document: Document) -> None:
        self._document = document

    async def get_document(self, _session, _user, _project_id, _document_id):
        return self._document


def _fonts(paragraph) -> dict[str, str]:
    fonts = paragraph.runs[0]._r.rPr.find(qn("w:rFonts"))
    return {slot: fonts.get(qn(f"w:{slot}")) for slot in ("ascii", "hAnsi", "cs")}


def _word(content: bytes):
    return open_word_document(io.BytesIO(content))


@pytest.mark.asyncio
async def test_document_docx_has_a_heading_per_page_and_ground_truth_lines() -> None:
    document = Document(id=uuid.uuid4(), project_id=uuid.uuid4(), name="Codex Syriacus 1")
    first, second, third = (
        DocumentPart(id=uuid.uuid4(), document_id=document.id, order=order) for order in range(3)
    )
    # A reviewed_only run over pages 1 and 3: page numbers keep their document position.
    selection = DocumentExportSelection(
        document=document,
        parts=[first, third],
        page_numbers={first.id: 1, second.id: 2, third.id: 3},
    )
    repository = _Repository(
        {
            first.id: [_line(_SYRIAC), _line("guess", TranscriptionKind.model), _line("  ")],
            third.id: [_line(_GREEK), _line(_ARMENIAN), _line(_COPTIC)],
        }
    )
    service = DocumentDocxExportService(documents=repository, selector=_Selector(selection))

    export = await service.export_document_docx(
        object(), object(), document.project_id, document.id
    )

    assert export.filename == "codex-syriacus-1-transcription.docx"
    word = _word(export.content)
    paragraphs = [(p.style.name, p.text) for p in word.paragraphs]
    # Unapproved model text and blank lines are not ground truth, as in the text export.
    # The empty Normal paragraph between pages is the page break starting page 3.
    assert paragraphs == [
        ("Heading 2", "Page 1"),
        ("Normal", _SYRIAC),
        ("Normal", ""),
        ("Heading 2", "Page 3"),
        ("Normal", _GREEK),
        ("Normal", _ARMENIAN),
        ("Normal", _COPTIC),
    ]


def test_each_line_names_its_script_font_and_only_syriac_is_right_to_left() -> None:
    word = _word(render_docx([DocxPage(number=1, lines=[_SYRIAC, _GREEK, _ARMENIAN, _COPTIC])]))
    syriac, greek, armenian, coptic = word.paragraphs[1:]

    assert _fonts(syriac) == dict.fromkeys(("ascii", "hAnsi", "cs"), "Noto Sans Syriac Eastern")
    assert _fonts(greek) == dict.fromkeys(("ascii", "hAnsi", "cs"), "Noto Sans")
    assert _fonts(armenian) == dict.fromkeys(("ascii", "hAnsi", "cs"), "Noto Sans Armenian")
    assert _fonts(coptic) == dict.fromkeys(("ascii", "hAnsi", "cs"), "Noto Sans Coptic")

    assert syriac._p.pPr.find(qn("w:bidi")) is not None
    assert syriac.runs[0].font.rtl is True
    for paragraph in (greek, armenian, coptic):
        assert paragraph._p.pPr is None or paragraph._p.pPr.find(qn("w:bidi")) is None
        assert not paragraph.runs[0].font.rtl


def test_default_font_is_noto_sans_twelve_point() -> None:
    normal = _word(render_docx([DocxPage(number=1, lines=[])])).styles["Normal"]
    assert normal.font.name == "Noto Sans"
    assert normal.font.size.pt == 12
    assert normal.element.rPr.find(qn("w:szCs")).get(qn("w:val")) == "24"


def test_a_page_with_nothing_transcribed_keeps_its_heading() -> None:
    word = _word(render_docx([DocxPage(number=1, lines=["a"]), DocxPage(number=2, lines=[])]))
    assert [p.text for p in word.paragraphs] == ["Page 1", "a", "", "Page 2"]


def _page_breaks(content: bytes) -> int:
    word = _word(content)
    return sum(
        1 for element in word.element.iter(qn("w:br")) if element.get(qn("w:type")) == "page"
    )


def test_two_pages_hold_exactly_one_page_break() -> None:
    content = render_docx([DocxPage(number=1, lines=["a"]), DocxPage(number=2, lines=["b"])])
    assert _page_breaks(content) == 1


def test_one_page_holds_no_page_break() -> None:
    assert _page_breaks(render_docx([DocxPage(number=1, lines=["a"])])) == 0


@pytest.mark.asyncio
async def test_member_part_docx_names_the_page_stem_and_keeps_ground_truth() -> None:
    document = Document(id=uuid.uuid4(), project_id=uuid.uuid4(), name="Codex Syriacus 1")
    part = DocumentPart(id=uuid.uuid4(), document_id=document.id, order=1)
    repository = _Repository(
        {part.id: [_line(_SYRIAC), _line("guess", TranscriptionKind.model)]},
        parts={part.id: part},
        page_numbers={part.id: 2},
    )
    service = DocumentDocxExportService(
        documents=repository, document_service=_DocumentService(document)
    )

    export = await service.export_part_docx(
        object(), object(), document.project_id, document.id, part.id
    )

    assert export.filename == "Codex_Syriacus_1_page_2.docx"
    word = _word(export.content)
    assert [(p.style.name, p.text) for p in word.paragraphs] == [
        ("Heading 2", "Page 2"),
        ("Normal", _SYRIAC),
    ]


# --- Regions (ADR 0011) ---


def _spread_page():
    """A spread: verso and recto regions, labelled, plus one line in no region.

    The line list is deliberately not region-first, as on a page numbered before the
    regions were drawn: the export groups by region, it does not trust the order.
    """
    document = Document(id=uuid.uuid4(), project_id=uuid.uuid4(), name="Grec 1360")
    part = DocumentPart(id=uuid.uuid4(), document_id=document.id, order=0)
    verso = Block(id=uuid.uuid4(), part_id=part.id, order=0, box={"points": []}, label="4v")
    recto = Block(id=uuid.uuid4(), part_id=part.id, order=1, box={"points": []}, label="5r")
    lines = [_line("recto one"), _line("verso one"), _line("loose"), _line("verso two")]
    for line, block in zip(lines, (recto, verso, None, verso), strict=True):
        line.block_id = block.id if block is not None else None
    selection = DocumentExportSelection(document=document, parts=[part], page_numbers={part.id: 1})
    repository = _Repository({part.id: lines}, {part.id: [verso, recto]})
    return document, selection, repository


@pytest.mark.asyncio
async def test_docx_prints_regions_in_order_under_bold_labels_and_loose_lines_last() -> None:
    document, selection, repository = _spread_page()
    service = DocumentDocxExportService(documents=repository, selector=_Selector(selection))

    export = await service.export_document_docx(
        object(), object(), document.project_id, document.id
    )

    word = _word(export.content)
    assert [p.text for p in word.paragraphs] == [
        "Page 1",
        "4v",
        "verso one",
        "verso two",
        "5r",
        "recto one",
        "loose",
    ]
    labels = [p for p in word.paragraphs if p.text in {"4v", "5r"}]
    assert all(p.runs[0].bold for p in labels)
    assert not any(p.runs[0].bold for p in word.paragraphs if p.text == "verso one")


@pytest.mark.asyncio
async def test_text_export_prints_region_labels_on_their_own_lines() -> None:
    document, selection, repository = _spread_page()
    service = DocumentTextExportService(documents=repository, selector=_Selector(selection))

    export = await service.export_document_text(
        object(), object(), document.project_id, document.id
    )

    assert export.text == "[p.1]\n4v\nverso one\nverso two\n5r\nrecto one\nloose\n"


def test_an_unlabelled_region_has_no_heading_but_keeps_its_lines_together() -> None:
    word = _word(render_docx([DocxPage(number=1, lines=[RegionHeading("4v"), "a", "b", "c"])]))
    assert [p.text for p in word.paragraphs] == ["Page 1", "4v", "a", "b", "c"]


def test_a_vertical_tab_becomes_a_line_break_inside_the_line() -> None:
    word = _word(render_docx([DocxPage(number=1, lines=["abc\x0bdef"])]))
    line = word.paragraphs[1]

    assert line.text == "abc\ndef"
    assert line.runs[0]._r.find(qn("w:br")) is not None


def test_xml_invalid_characters_are_dropped_not_a_failed_export() -> None:
    word = _word(
        render_docx([DocxPage(number=1, lines=[RegionHeading("fol.\x001r"), "a\x00b\x07c￾￿d"])])
    )
    assert [p.text for p in word.paragraphs] == ["Page 1", "fol.1r", "abcd"]
