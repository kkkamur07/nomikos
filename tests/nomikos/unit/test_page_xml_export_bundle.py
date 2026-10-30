"""PAGE XML bundle naming and packaging, without a database.

The parts that touch Postgres and object storage are covered end to end in
``tests/nomikos/integration/test_transcription_pdf_artifact.py``; this file pins
the pure pieces: the shared file stem, the archive layout, and the download header.
"""

from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace
from uuid import uuid4
from xml.etree.ElementTree import fromstring
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import pytest

from backend.annotation.application.page_xml_export_service import (
    PageXmlBundle,
    PageXmlExportService,
    _baseline_export_points,
    _ground_truth_text,
    export_file_stem,
)
from backend.core.api.content_disposition import attachment_disposition
from backend.document.infrastructure.orm_models import (
    LineGeometryKind,
    LineSource,
    TranscriptionKind,
)

PAGE_NS = {"p": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"}


def _line(*, points, mask=None, baseline=None, transcriptions=()):
    return SimpleNamespace(
        id=uuid4(),
        points=points,
        mask=mask,
        baseline=baseline,
        source=LineSource.manual,
        kind=LineGeometryKind.polygon,
        transcriptions=list(transcriptions),
    )


def _transcription(text, kind):
    return SimpleNamespace(text=text, transcription=SimpleNamespace(kind=kind))


@pytest.mark.parametrize(
    ("document_name", "page_number", "expected"),
    [
        ("My Codex", 3, "My_Codex_page_3"),
        (
            'slash/back\\colon:star*q?quote"lt<gt>pipe|',
            2,
            "slashbackcolonstarqquoteltgtpipe_page_2",
        ),
        ("Σιναϊτικός κώδικας", 12, "Σιναϊτικός_κώδικας_page_12"),
        ("", 1, "document_page_1"),
        ("trailing dots...", 4, "trailing_dots_page_4"),
    ],
)
def test_export_file_stem(document_name: str, page_number: int, expected: str) -> None:
    assert export_file_stem(document_name, page_number) == expected


def test_export_file_stem_caps_long_titles() -> None:
    stem = export_file_stem("x" * 500, 7)
    assert stem == "x" * 80 + "_page_7"


def test_bundle_zips_xml_and_image_side_by_side() -> None:
    bundle = PageXmlBundle(
        stem="Codex_page_2",
        xml=b"<?xml version='1.0'?><PcGts/>",
        image_filename="Codex_page_2.webp",
        image=b"RIFF....WEBP",
    )

    assert bundle.xml_filename == "Codex_page_2.xml"
    assert bundle.zip_filename == "Codex_page_2.zip"

    with ZipFile(BytesIO(bundle.to_zip())) as archive:
        assert archive.namelist() == ["Codex_page_2.xml", "Codex_page_2.webp"]
        assert archive.read("Codex_page_2.xml") == bundle.xml
        assert archive.read("Codex_page_2.webp") == bundle.image
        # The XML deflates, the already-compressed raster is stored as-is.
        assert archive.getinfo("Codex_page_2.xml").compress_type == ZIP_DEFLATED
        assert archive.getinfo("Codex_page_2.webp").compress_type == ZIP_STORED


def test_attachment_disposition_plain_ascii_has_no_encoded_form() -> None:
    assert attachment_disposition("Codex_page_1.zip") == 'attachment; filename="Codex_page_1.zip"'


def test_attachment_disposition_non_ascii_keeps_a_fallback_and_the_real_name() -> None:
    header = attachment_disposition("Σιναϊτικός_page_1.zip")
    assert header.startswith("attachment; filename=\"__page_1.zip\"; filename*=UTF-8''")
    assert "%CE%A3" in header  # Σ, percent-encoded UTF-8
    assert header.isascii()


# --- PAGE Baseline geometry priority: mask, then baseline, then coords ---


def test_baseline_export_prefers_mask_over_baseline_over_coords() -> None:
    coords = [[0.0, 0.0], [10.0, 0.0]]
    mask = {"points": [[1.0, 1.0], [2.0, 2.0]]}
    baseline = {"points": [[3.0, 3.0], [4.0, 4.0]]}

    assert (
        _baseline_export_points(_line(points=coords, mask=mask, baseline=baseline))
        == mask["points"]
    )
    assert _baseline_export_points(_line(points=coords, baseline=baseline)) == baseline["points"]
    assert _baseline_export_points(_line(points=coords)) == coords


def test_baseline_export_falls_through_malformed_geometry_to_coords() -> None:
    coords = [[5.0, 6.0], [7.0, 8.0]]
    # points not a list, then a point of wrong arity: both rejected, so coords win.
    line = _line(
        points=coords,
        mask={"points": "not-a-list"},
        baseline={"points": [[1.0, 2.0, 3.0]]},
    )
    assert _baseline_export_points(line) == coords


def test_ground_truth_text_selects_the_ground_truth_layer() -> None:
    line = _line(
        points=[[0.0, 0.0]],
        transcriptions=[
            _transcription("model guess", TranscriptionKind.model),
            _transcription("ἀληθές", TranscriptionKind.ground_truth),
        ],
    )
    assert _ground_truth_text(line) == "ἀληθές"


def test_ground_truth_text_is_empty_without_a_ground_truth_layer() -> None:
    line = _line(
        points=[[0.0, 0.0]],
        transcriptions=[_transcription("model guess", TranscriptionKind.model)],
    )
    assert _ground_truth_text(line) == ""


def test_export_part_lines_renders_coords_baseline_and_ground_truth() -> None:
    # __new__ skips __init__ so no media store or repositories are constructed;
    # _export_part_lines only needs the sibling _append_text_line method.
    service = PageXmlExportService.__new__(PageXmlExportService)
    part = SimpleNamespace(id=uuid4(), width=640, height=480, image_key="parts/x.webp")
    line = _line(
        points=[[0.0, 0.0], [10.0, 0.0], [10.0, 5.0]],
        mask={"points": [[1.0, 1.0], [9.0, 1.0], [9.0, 4.0]]},
        transcriptions=[_transcription("ἐν ἀρχῇ", TranscriptionKind.ground_truth)],
    )

    xml = service._export_part_lines(
        part, [line], image_filename="Codex_page_1.webp", image_size=None
    )
    root = fromstring(xml)  # noqa: S314 - XML produced by this test, not untrusted input

    page = root.find("p:Page", PAGE_NS)
    assert page.get("imageFilename") == "Codex_page_1.webp"
    assert page.get("imageWidth") == "640"
    assert page.get("imageHeight") == "480"

    text_line = root.find(".//p:TextLine", PAGE_NS)
    assert text_line.find("p:Coords", PAGE_NS).get("points") == "0,0 10,0 10,5"
    # Baseline comes from the mask polygon, not the coords.
    assert text_line.find("p:Baseline", PAGE_NS).get("points") == "1,1 9,1 9,4"
    assert text_line.find(".//p:Unicode", PAGE_NS).text == "ἐν ἀρχῇ"


def test_export_part_lines_omits_dimensions_when_the_part_has_none() -> None:
    service = PageXmlExportService.__new__(PageXmlExportService)
    part = SimpleNamespace(id=uuid4(), width=None, height=None, image_key="parts/x.webp")

    xml = service._export_part_lines(part, [], image_filename="Codex_page_2.img", image_size=None)
    page = fromstring(xml).find("p:Page", PAGE_NS)  # noqa: S314 - XML produced by this test

    assert page.get("imageFilename") == "Codex_page_2.img"
    assert "imageWidth" not in page.attrib
    assert "imageHeight" not in page.attrib
