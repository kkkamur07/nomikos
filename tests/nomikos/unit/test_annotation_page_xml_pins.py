"""Parsed-XML pins for the PAGE export: tags, attributes and coordinates.

Every tag name, attribute name and formatted coordinate below is asserted
exactly as parsed back from the generated document, so renaming a PAGE
element or swapping x and y fails here rather than in a consumer.
"""

from __future__ import annotations

from datetime import datetime
from io import BytesIO
from types import SimpleNamespace
from uuid import uuid4
from xml.etree.ElementTree import fromstring, tostring

import pytest
from PIL import Image

from backend.annotation.application.page_xml_export_service import (
    PAGE_XML_NAMESPACE,
    PageXmlExportService,
    _baseline_export_points,
    _format_coord,
    _ground_truth_text,
    _image_filename,
    _points,
    export_file_stem,
)
from backend.core.exceptions import NotFoundError
from backend.document.infrastructure.orm_models import TranscriptionKind

NS = {"p": PAGE_XML_NAMESPACE}


def _line(*, points, mask=None, baseline=None, source="kraken", kind="polygon", text=None):
    transcriptions = []
    if text is not None:
        transcriptions = [
            SimpleNamespace(
                text=text, transcription=SimpleNamespace(kind=TranscriptionKind.ground_truth)
            )
        ]
    return SimpleNamespace(
        id=uuid4(),
        points=points,
        mask=mask,
        baseline=baseline,
        source=SimpleNamespace(value=source),
        kind=SimpleNamespace(value=kind),
        transcriptions=transcriptions,
    )


def _part(**over):
    values = {
        "id": uuid4(),
        "width": 640,
        "height": 480,
        "image_key": "parts/x.webp",
    }
    values.update(over)
    return SimpleNamespace(**values)


def _service():
    return PageXmlExportService.__new__(PageXmlExportService)


class TestDocumentEnvelope:
    def _xml(self):
        service = _service()
        part = _part()
        line = _line(
            points=[[0.0, 0.0], [10.0, 0.0], [10.0, 5.0]],
            baseline={"points": [[2.0, 3.0], [8.0, 3.0]]},
            text="ἐν ἀρχῇ",
        )
        xml = service._export_part_lines(
            part, [line], image_filename="Codex_page_1.webp", image_size=None
        )
        return xml, part, line

    def test_declaration_root_and_schema_location(self) -> None:
        xml, part, _line = self._xml()
        assert xml.startswith(b"<?xml version='1.0' encoding='utf-8'?>")
        root = fromstring(xml)  # noqa: S314 - XML produced by this test
        assert root.tag == "{http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15}PcGts"
        assert (
            root.get("{http://www.w3.org/2001/XMLSchema-instance}schemaLocation")
            == "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15 "
            "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15/pagecontent.xsd"
        )
        assert root.get("pcGtsId") == f"pcgts_{part.id}"

    def test_metadata_creator_and_created(self) -> None:
        xml, _part, _line = self._xml()
        root = fromstring(xml)  # noqa: S314 - XML produced by this test
        assert root.find("p:Metadata/p:Creator", NS).text == "Nomikos"
        created = root.find("p:Metadata/p:Created", NS).text
        assert datetime.fromisoformat(created) is not None

    def test_page_and_region_attributes(self) -> None:
        xml, part, _line = self._xml()
        root = fromstring(xml)  # noqa: S314 - XML produced by this test
        page = root.find("p:Page", NS)
        assert page.get("imageFilename") == "Codex_page_1.webp"
        assert page.get("imageWidth") == "640"
        assert page.get("imageHeight") == "480"
        region = page.find("p:TextRegion", NS)
        assert region.get("id") == f"region_{part.id}"
        assert region.get("type") == "paragraph"

    def test_text_line_identity_geometry_and_text(self) -> None:
        xml, _part, line = self._xml()
        root = fromstring(xml)  # noqa: S314 - XML produced by this test
        text_line = root.find(".//p:TextLine", NS)
        assert text_line.get("id") == f"line_{line.id}"
        assert text_line.get("custom") == "source:kraken; kind:polygon"
        assert text_line.find("p:Coords", NS).get("points") == "0,0 10,0 10,5"
        assert text_line.find("p:Baseline", NS).get("points") == "2,3 8,3"
        assert text_line.find(".//p:Unicode", NS).text == "ἐν ἀρχῇ"

    def test_reading_order_follows_the_line_list(self) -> None:
        service = _service()
        part = _part()
        lines = [
            _line(points=[[0.0, float(i)], [10.0, float(i)]], text=f"line {i}") for i in range(3)
        ]
        xml = service._export_part_lines(part, lines, image_filename="s.webp", image_size=None)
        root = fromstring(xml)  # noqa: S314 - XML produced by this test
        found = root.findall(".//p:TextLine", NS)
        assert [el.get("id") for el in found] == [f"line_{line.id}" for line in lines]
        assert [el.find(".//p:Unicode", NS).text for el in found] == [
            "line 0",
            "line 1",
            "line 2",
        ]


class TestCoordinateFormatting:
    def test_points_join_xy_pairs_with_spaces(self) -> None:
        assert _points([[1.0, 2.0], [2.5, 3.0]]) == "1,2 2.5,3"

    def test_format_coord_drops_integer_points(self) -> None:
        assert _format_coord(4.0) == "4"
        assert _format_coord(4.5) == "4.5"

    def test_baseline_export_never_emits_xy_swapped(self) -> None:
        line = _line(points=[[0.0, 0.0]], baseline={"points": [[1.0, 9.0], [8.0, 2.0]]})
        assert _baseline_export_points(line) == [[1.0, 9.0], [8.0, 2.0]]

    def test_ground_truth_text_strips_edges(self) -> None:
        line = _line(points=[[0.0, 0.0]], text="  ἔργον  ")
        assert _ground_truth_text(line) == "ἔργον"


class TestFilenames:
    @pytest.mark.parametrize(
        ("image_key", "suffix"),
        [
            ("parts/x.png", "png"),
            ("parts/x.jpg", "jpg"),
            ("parts/x.jpeg", "jpg"),
            ("parts/x.JPG", "jpg"),
            ("parts/x.gif", "gif"),
            ("parts/x.webp", "webp"),
            ("parts/x.bmp", "img"),
            ("parts/noext", "img"),
        ],
    )
    def test_image_filename_maps_suffixes(self, image_key: str, suffix: str) -> None:
        assert _image_filename("Codex_page_1", _part(image_key=image_key)).endswith(f".{suffix}")

    def test_export_file_stem_cleans_and_falls_back(self) -> None:
        assert export_file_stem("Codex Sinaiticus", 1) == "Codex_Sinaiticus_page_1"
        assert export_file_stem("a/b:c", 2) == "abc_page_2"
        assert export_file_stem("   ", 3) == "document_page_3"
        assert export_file_stem("Σιναϊτικός", 4) == "Σιναϊτικός_page_4"


class _Repository:
    def __init__(self, part, lines, page_number: int = 1) -> None:
        self._part = part
        self._lines = lines
        self._page_number = page_number

    async def get_part(self, _session, _part_id):
        return self._part

    async def list_part_lines(self, _session, _part_id):
        return self._lines

    async def part_page_number(self, _session, _part):
        return self._page_number


class _DocumentService:
    def __init__(self, document, part, image_bytes: bytes = b"") -> None:
        self._document = document
        self._part = part
        self._image_bytes = image_bytes
        self.parts = self

    async def get_document(self, _session, _user, _project_id, _document_id):
        return self._document

    async def get_published_part_context(
        self, _session, _project_id, _document_id, _part_id, *, token=None
    ):
        return SimpleNamespace(document=self._document, part=self._part)

    async def read_part_bytes(self, _part, *, width=None):
        return self._image_bytes


def _png_bytes(width: int = 64, height: int = 48) -> bytes:
    image = Image.new("RGB", (width, height), "white")
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


class TestServicePaths:
    def _context(self, part, lines, **over):
        document = SimpleNamespace(id=uuid4(), name="Codex")
        part.document_id = document.id
        repo = _Repository(part, lines)
        svc = _DocumentService(document, part, **over)
        service = PageXmlExportService(documents=repo, document_service=svc)
        return service, document, uuid4(), uuid4()

    @pytest.mark.asyncio
    async def test_export_part_matches_the_pure_renderer(self) -> None:
        part = _part()
        lines = [_line(points=[[0.0, 0.0], [9.0, 0.0]], text="α")]
        service, document, project_id, _document_id = self._context(part, lines)

        xml = await service.export_part(object(), object(), project_id, document.id, part.id)
        expected = service._export_part_lines(
            part, lines, image_filename="Codex_page_1.webp", image_size=None
        )

        assert _without_created(xml) == _without_created(expected)

    @pytest.mark.asyncio
    async def test_export_part_for_another_document_is_not_found(self) -> None:
        part = _part()
        service, _document, project_id, _document_id = self._context(part, [])
        part.document_id = uuid4()

        with pytest.raises(NotFoundError, match="^Part not found$"):
            await service.export_part(object(), object(), project_id, uuid4(), part.id)

    @pytest.mark.asyncio
    async def test_bundle_reads_dimensions_from_the_image_header(self) -> None:
        part = _part(width=None, height=None)
        lines = [_line(points=[[0.0, 0.0], [9.0, 0.0]], text="α")]
        image = _png_bytes(64, 48)
        service, document, project_id, _document_id = self._context(part, lines, image_bytes=image)

        bundle = await service.export_part_bundle(
            object(), object(), project_id, document.id, part.id
        )

        assert bundle.image == image
        root = fromstring(bundle.xml)  # noqa: S314 - XML produced by this test
        page = root.find("p:Page", NS)
        assert page.get("imageWidth") == "64"
        assert page.get("imageHeight") == "48"
        assert bundle.image_filename == "Codex_page_1.webp"
        assert bundle.stem == "Codex_page_1"

    @pytest.mark.asyncio
    async def test_public_export_uses_the_published_context(self) -> None:
        part = _part()
        lines = [_line(points=[[0.0, 0.0], [9.0, 0.0]], text="α")]
        service, _document, project_id, document_id = self._context(part, lines)

        xml = await service.export_part_public(object(), project_id, document_id, part.id)
        expected = service._export_part_lines(
            part, lines, image_filename="Codex_page_1.webp", image_size=None
        )

        assert _without_created(xml) == _without_created(expected)


def _without_created(xml: bytes) -> bytes:
    root = fromstring(xml)  # noqa: S314 - XML produced by this test
    root.find("p:Metadata/p:Created", NS).text = "fixed"
    return tostring(root, encoding="utf-8", xml_declaration=True)


class TestKillerPins:
    def test_text_equiv_parent_tag_is_exact(self) -> None:
        service = _service()
        part = _part()
        line = _line(points=[[0.0, 0.0], [9.0, 0.0]], text="α")
        xml = service._export_part_lines(part, [line], image_filename="s.webp", image_size=None)
        root = fromstring(xml)  # noqa: S314 - XML produced by this test
        text_line = root.find(".//p:TextLine", NS)
        equiv = text_line.find("p:TextEquiv", NS)
        assert equiv is not None
        assert equiv.find("p:Unicode", NS).text == "α"

    def test_stem_strips_a_trailing_underscore(self) -> None:
        assert export_file_stem("Codex_", 1) == "Codex_page_1"
        assert export_file_stem("CODEX_", 1) == "CODEX_page_1"

    def test_three_point_outline_falls_back_to_its_lower_edge(self) -> None:
        line = _line(
            points=[[0.0, 0.0], [10.0, 0.0], [10.0, 5.0]],
            mask={"points": [[1.0, 1.0], [9.0, 1.0], [9.0, 4.0]]},
        )
        assert len(_baseline_export_points(line)) == 2

    @pytest.mark.asyncio
    async def test_public_bundle_uses_the_published_context(self) -> None:
        part = _part()
        lines = [_line(points=[[0.0, 0.0], [9.0, 0.0]], text="α")]
        document = SimpleNamespace(id=uuid4(), name="Codex")
        part.document_id = document.id
        repo = _Repository(part, lines)
        svc = _DocumentService(document, part, image_bytes=_png_bytes(64, 48))
        service = PageXmlExportService(documents=repo, document_service=svc)

        bundle = await service.export_part_bundle_public(
            object(), document.id, document.id, part.id
        )

        assert bundle.image_filename == "Codex_page_1.webp"
        assert bundle.stem == "Codex_page_1"

    @pytest.mark.asyncio
    async def test_bundle_prefers_stored_dimensions(self) -> None:
        part = _part(width=640, height=480)
        lines = [_line(points=[[0.0, 0.0], [9.0, 0.0]], text="α")]
        document = SimpleNamespace(id=uuid4(), name="Codex")
        part.document_id = document.id
        repo = _Repository(part, lines)
        svc = _DocumentService(document, part, image_bytes=_png_bytes(64, 48))
        service = PageXmlExportService(documents=repo, document_service=svc)

        bundle = await service.export_part_bundle(
            object(), object(), document.id, document.id, part.id
        )

        root = fromstring(bundle.xml)  # noqa: S314 - XML produced by this test
        page = root.find("p:Page", NS)
        assert page.get("imageWidth") == "640"
        assert page.get("imageHeight") == "480"

    @pytest.mark.asyncio
    async def test_bundle_with_half_dimensions_reads_the_header(self) -> None:
        part = _part(width=640, height=None)
        lines = [_line(points=[[0.0, 0.0], [9.0, 0.0]], text="α")]
        document = SimpleNamespace(id=uuid4(), name="Codex")
        part.document_id = document.id
        repo = _Repository(part, lines)
        svc = _DocumentService(document, part, image_bytes=_png_bytes(64, 48))
        service = PageXmlExportService(documents=repo, document_service=svc)

        bundle = await service.export_part_bundle(
            object(), object(), document.id, document.id, part.id
        )

        root = fromstring(bundle.xml)  # noqa: S314 - XML produced by this test
        page = root.find("p:Page", NS)
        assert page.get("imageWidth") == "64"
        assert page.get("imageHeight") == "48"
