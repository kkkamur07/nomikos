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
    _ground_truth_text,
    export_file_stem,
)
from backend.annotation.application.page_xml_validation import validate_page_xml
from backend.core.api.content_disposition import attachment_disposition
from backend.document.infrastructure.orm_models import (
    LineGeometryKind,
    LineSource,
    TranscriptionKind,
)

PAGE_NS = {"p": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"}


def _line(
    *,
    points,
    mask=None,
    baseline=None,
    transcriptions=(),
    source=LineSource.manual,
    block_id=None,
    manual_geometry=False,
):
    return SimpleNamespace(
        id=uuid4(),
        points=points,
        mask=mask,
        baseline=baseline,
        source=source,
        kind=LineGeometryKind.polygon,
        block_id=block_id,
        manual_geometry=manual_geometry,
        transcriptions=list(transcriptions),
    )


def _block(*, order, block_id=None):
    return SimpleNamespace(id=block_id or uuid4(), order=order, box={})


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


# --- Rendering: Coords is the polygon, Baseline is a baseline ---
#
# The renderer had no unit coverage before 2026-09-18, which is how it shipped for two
# months writing the mask polygon into <Baseline>.


def _render(part, blocks, lines, *, image_filename="Codex_page_1.webp", image_size=None):
    # __new__ skips __init__ so no media store or repositories are constructed;
    # _export_part_lines only needs the sibling _append_text_line method.
    service = PageXmlExportService.__new__(PageXmlExportService)
    xml, counts = service._export_part_lines(
        part, blocks, lines, image_filename=image_filename, image_size=image_size
    )
    root = fromstring(xml)  # noqa: S314 - XML produced by this test, not untrusted input
    return root, counts, xml


def _part(**overrides):
    return SimpleNamespace(id=uuid4(), width=640, height=480, image_key="parts/x.webp", **overrides)


def test_a_line_with_a_mask_and_a_baseline_exports_both_in_their_own_elements() -> None:
    line = _line(
        points=[[0.0, 0.0], [20.0, 0.0], [20.0, 10.0], [0.0, 10.0]],
        mask={"points": [[1.0, 1.0], [19.0, 1.0], [19.0, 9.0], [1.0, 9.0]]},
        baseline={"points": [[2.0, 7.0], [18.0, 7.0]]},
        source=LineSource.kraken,
        transcriptions=[_transcription("ἐν ἀρχῇ", TranscriptionKind.ground_truth)],
    )

    root, counts, xml = _render(_part(), [], [line])

    page = root.find("p:Page", PAGE_NS)
    assert page.get("imageFilename") == "Codex_page_1.webp"
    assert page.get("imageWidth") == "640"
    assert page.get("imageHeight") == "480"

    text_line = root.find(".//p:TextLine", PAGE_NS)
    # Coords is the mask, which is the boundary the editor shows and a human corrects.
    assert text_line.find("p:Coords", PAGE_NS).get("points") == "1,1 19,1 19,9 1,9"
    assert text_line.find("p:Baseline", PAGE_NS).get("points") == "2,7 18,7"
    assert text_line.find(".//p:Unicode", PAGE_NS).text == "ἐν ἀρχῇ"
    assert "baseline:derived" not in text_line.get("custom")
    assert counts == counts._replace(total=1, kraken=1, ground_truth=1)
    assert validate_page_xml(xml).ok


def test_a_line_with_only_points_falls_back_to_them_for_coords() -> None:
    points = [[0.0, 0.0], [20.0, 0.0], [20.0, 10.0], [0.0, 10.0]]
    line = _line(
        points=points,
        baseline={"points": [[2.0, 7.0], [18.0, 7.0]]},
        transcriptions=[_transcription("β", TranscriptionKind.ground_truth)],
    )

    root, counts, _ = _render(_part(), [], [line])

    text_line = root.find(".//p:TextLine", PAGE_NS)
    assert text_line.find("p:Coords", PAGE_NS).get("points") == "0,0 20,0 20,10 0,10"
    assert counts.derived_baseline == 0


def test_a_hand_drawn_box_gets_a_derived_baseline_marked_in_custom() -> None:
    # What ``create_part_line`` writes when the client sends no baseline.
    points = [[0.0, 100.0], [100.0, 100.0], [100.0, 120.0], [0.0, 120.0]]
    line = _line(
        points=points,
        baseline={"points": points},
        manual_geometry=True,
        transcriptions=[_transcription("γ", TranscriptionKind.ground_truth)],
    )

    root, counts, xml = _render(_part(), [], [line])

    text_line = root.find(".//p:TextLine", PAGE_NS)
    assert "baseline:derived" in text_line.get("custom")
    baseline = text_line.find("p:Baseline", PAGE_NS).get("points")
    assert baseline == "1,114 99,114"
    assert counts.derived_baseline == 1
    assert counts.manual == 1
    assert counts.manual_geometry == 1
    # And the derived baseline is still a baseline a trainer accepts.
    assert validate_page_xml(xml).ok


def test_one_block_stays_one_region_with_no_reading_order() -> None:
    # The BLLA adapter emits exactly one block covering the whole page; splitting
    # there would invent a structure nobody drew.
    part = _part()
    block = _block(order=0)
    lines = [
        _line(
            points=[[0.0, 0.0], [10.0, 0.0], [10.0, 5.0]],
            baseline={"points": [[1.0, 4.0], [9.0, 4.0]]},
            block_id=block.id,
        )
    ]

    root, _, _ = _render(part, [block], lines)

    regions = root.findall(".//p:TextRegion", PAGE_NS)
    assert len(regions) == 1
    assert regions[0].get("id") == f"region_{part.id}"
    assert root.find(".//p:ReadingOrder", PAGE_NS) is None


def test_two_blocks_become_two_regions_in_a_reading_order() -> None:
    left, right = _block(order=1), _block(order=0)
    baseline = {"points": [[1.0, 4.0], [9.0, 4.0]]}
    polygon = [[0.0, 0.0], [10.0, 0.0], [10.0, 5.0]]
    lines = [
        _line(points=polygon, baseline=baseline, block_id=left.id),
        _line(points=polygon, baseline=baseline, block_id=right.id),
    ]

    root, counts, _ = _render(_part(), [left, right], lines)

    regions = root.findall(".//p:TextRegion", PAGE_NS)
    assert [region.get("id") for region in regions] == [f"region_{right.id}", f"region_{left.id}"]
    refs = root.findall(".//p:RegionRefIndexed", PAGE_NS)
    assert [(ref.get("index"), ref.get("regionRef")) for ref in refs] == [
        ("0", f"region_{right.id}"),
        ("1", f"region_{left.id}"),
    ]
    # ReadingOrder precedes the regions it names, as PAGE 2019-07-15 requires.
    page_children = [child.tag for child in root.find("p:Page", PAGE_NS)]
    assert page_children[0].endswith("ReadingOrder")
    assert counts.total == 2


def test_a_line_belonging_to_no_block_gets_a_region_of_its_own() -> None:
    part = _part()
    first, second = _block(order=0), _block(order=1)
    baseline = {"points": [[1.0, 4.0], [9.0, 4.0]]}
    polygon = [[0.0, 0.0], [10.0, 0.0], [10.0, 5.0]]
    lines = [
        _line(points=polygon, baseline=baseline, block_id=first.id),
        _line(points=polygon, baseline=baseline, block_id=second.id),
        _line(points=polygon, baseline=baseline, block_id=None),
    ]

    root, _, _ = _render(part, [first, second], lines)

    regions = [region.get("id") for region in root.findall(".//p:TextRegion", PAGE_NS)]
    assert regions[-1] == f"region_{part.id}_unassigned"


def test_an_empty_page_renders_one_empty_region_and_zero_counts() -> None:
    part = _part()

    root, counts, _ = _render(part, [], [])

    regions = root.findall(".//p:TextRegion", PAGE_NS)
    assert len(regions) == 1
    # A boundary but no lines: kraken drops a region that has no Coords at all.
    assert [child.tag.split("}")[-1] for child in regions[0]] == ["Coords"]
    assert counts.total == 0


def test_export_part_lines_omits_dimensions_when_the_part_has_none() -> None:
    part = SimpleNamespace(id=uuid4(), width=None, height=None, image_key="parts/x.webp")

    root, _, _ = _render(part, [], [], image_filename="Codex_page_2.img")
    page = root.find("p:Page", PAGE_NS)

    assert page.get("imageFilename") == "Codex_page_2.img"
    assert "imageWidth" not in page.attrib
    assert "imageHeight" not in page.attrib
