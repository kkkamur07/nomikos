"""Transcription PDF artifacts through the platform API."""

from __future__ import annotations

import io
from io import BytesIO
from xml.etree import ElementTree
from zipfile import ZipFile

from fastapi.testclient import TestClient
from PIL import Image
from pypdf import PdfReader

from backend.document.infrastructure.orm_models import (
    LineTranscription,
    Transcription,
    TranscriptionKind,
)
from infrastructure.db import sync_system_session
from tests.nomikos.integration.helpers import documents_url


def _png_bytes(width: int = 160, height: int = 90) -> bytes:
    buf = BytesIO()
    Image.new("RGB", (width, height), "white").save(buf, format="PNG")
    return buf.getvalue()


def _pdf_reader(pdf_bytes: bytes) -> PdfReader:
    return PdfReader(io.BytesIO(pdf_bytes))


def _create_document_part_with_segments(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> tuple[str, str, str, list[str]]:
    project_id = owner_project["id"]
    base = documents_url(project_id)
    create = client.post(base, headers=owner_headers, json={"name": "PDF codex"})
    assert create.status_code == 201
    document_id = create.json()["id"]

    upload = client.post(
        f"{base}/{document_id}/parts",
        headers=owner_headers,
        files={"file": ("page.png", _png_bytes(), "image/png")},
    )
    assert upload.status_code == 201
    part_id = upload.json()["id"]

    replace = client.put(
        f"{base}/{document_id}/parts/{part_id}/lines",
        headers=owner_headers,
        json={
            "lines": [
                {
                    "order": 0,
                    "kind": "polygon",
                    "points": [[10, 10], [120, 10], [120, 30], [10, 30]],
                    "source": "manual",
                },
                {
                    "order": 1,
                    "kind": "polygon",
                    "points": [[10, 45], [120, 45], [120, 65], [10, 65]],
                    "source": "manual",
                },
            ]
        },
    )
    assert replace.status_code == 200
    line_ids = [line["id"] for line in replace.json()]
    return project_id, document_id, part_id, line_ids


# --- Transcription PDF generation ---
# Tests PDF bytes from paired or empty pages. Does not run OCR.


def test_member_generates_transcription_pdf_from_paired_ground_truth(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    project_id, document_id, part_id, line_ids = _create_document_part_with_segments(
        client, owner_headers, owner_project
    )
    base = documents_url(project_id)
    import_response = client.put(
        f"{base}/{document_id}/parts/{part_id}/page-transcription",
        headers=owner_headers,
        json={"text": "Αθήνα\nunused"},
    )
    assert import_response.status_code == 200
    pair = client.post(
        f"{base}/{document_id}/parts/{part_id}/pairings",
        headers=owner_headers,
        json={"line_id": line_ids[0], "text_line_order": 0},
    )
    assert pair.status_code == 200

    response = client.post(
        f"{base}/{document_id}/parts/{part_id}/transcription-pdf",
        headers=owner_headers,
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF")
    reader = _pdf_reader(response.content)
    assert len(reader.pages) == 1
    assert "Αθήνα" in (reader.pages[0].extract_text() or "")
    assert reader.pages[0].mediabox.width == 160
    assert reader.pages[0].mediabox.height == 90


def test_member_generates_blank_same_size_transcription_pdf_without_paired_lines(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    project_id, document_id, part_id, _line_ids = _create_document_part_with_segments(
        client, owner_headers, owner_project
    )
    base = documents_url(project_id)

    response = client.post(
        f"{base}/{document_id}/parts/{part_id}/transcription-pdf",
        headers=owner_headers,
    )

    assert response.status_code == 200
    assert response.content.startswith(b"%PDF")
    reader = _pdf_reader(response.content)
    assert len(reader.pages) == 1
    assert (reader.pages[0].extract_text() or "").strip() == ""
    assert reader.pages[0].mediabox.width == 160
    assert reader.pages[0].mediabox.height == 90


def test_member_pdf_renders_unapproved_machine_text_when_nothing_is_approved(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    """A transcribed-but-unreviewed page is not blank; it shows the model's text.

    Approved text still wins on the line that has it. The two are drawn in different
    colours, which pypdf cannot read back, so the colour rule is held by the unit test
    on ``_pdf_lines``; here the check is that both texts reach the page at all.
    """
    project_id, document_id, part_id, line_ids = _create_document_part_with_segments(
        client, owner_headers, owner_project
    )
    base = documents_url(project_id)
    import_response = client.put(
        f"{base}/{document_id}/parts/{part_id}/page-transcription",
        headers=owner_headers,
        json={"text": "Αθήνα"},
    )
    assert import_response.status_code == 200
    pair = client.post(
        f"{base}/{document_id}/parts/{part_id}/pairings",
        headers=owner_headers,
        json={"line_id": line_ids[0], "text_line_order": 0},
    )
    assert pair.status_code == 200
    with sync_system_session() as session:
        model_layer = Transcription(
            document_id=document_id, name="Model run", kind=TranscriptionKind.model
        )
        session.add(model_layer)
        session.flush()
        session.add_all(
            [
                # Ignored: the line already has approved text.
                LineTranscription(
                    line_id=line_ids[0], transcription_id=model_layer.id, text="Αθηνα?"
                ),
                LineTranscription(
                    line_id=line_ids[1], transcription_id=model_layer.id, text="Σπάρτη"
                ),
            ]
        )
        session.commit()

    response = client.post(
        f"{base}/{document_id}/parts/{part_id}/transcription-pdf",
        headers=owner_headers,
    )

    assert response.status_code == 200
    text = _pdf_reader(response.content).pages[0].extract_text() or ""
    assert "Αθήνα" in text
    assert "Σπάρτη" in text
    assert "Αθηνα?" not in text


# --- PDF access control ---
# Tests outsiders cannot generate member-route PDFs. Does not test public download.


def test_outsider_cannot_generate_transcription_pdf(
    client: TestClient,
    owner_headers: dict[str, str],
    outsider_headers: dict[str, str],
    owner_project: dict,
) -> None:
    project_id, document_id, part_id, _line_ids = _create_document_part_with_segments(
        client, owner_headers, owner_project
    )
    base = documents_url(project_id)

    response = client.post(
        f"{base}/{document_id}/parts/{part_id}/transcription-pdf",
        headers=outsider_headers,
    )

    assert response.status_code in (403, 404)


# --- PAGE XML export ---
# Tests XML layout export with baseline and transcription. Does not validate against an XSD.


def test_member_exports_page_xml_with_a_derived_baseline_not_the_polygon(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    project_id, document_id, part_id, _line_ids = _create_document_part_with_segments(
        client, owner_headers, owner_project
    )
    base = documents_url(project_id)
    replace = client.put(
        f"{base}/{document_id}/parts/{part_id}/lines",
        headers=owner_headers,
        json={
            "lines": [
                {
                    "order": 0,
                    "kind": "polygon",
                    "points": [[10, 10], [120, 10], [120, 30], [10, 30]],
                    "source": "kraken",
                    "approved_text": "alpha",
                },
            ]
        },
    )
    assert replace.status_code == 200

    response = client.get(
        f"{base}/{document_id}/parts/{part_id}/page-xml",
        headers=owner_headers,
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/xml")
    # S314: the document being parsed is the response this test just asked the
    # app to render, in-process. There is no untrusted producer to defend against.
    root = ElementTree.fromstring(response.content)  # noqa: S314
    ns = {"page": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"}
    text_line = root.find(".//page:TextLine", ns)
    assert text_line is not None
    coords = text_line.find("page:Coords", ns)
    baseline = text_line.find("page:Baseline", ns)
    text = text_line.find("page:TextEquiv/page:Unicode", ns)
    assert coords is not None
    assert baseline is not None
    assert text is not None
    assert coords.attrib["points"] == "10,10 120,10 120,30 10,30"
    assert text.text == "alpha"

    # The bulk PUT stores a placeholder baseline along the polygon's bottom edge,
    # so the export derives one. It says so, in the grammar kraken parses.
    assert text_line.attrib["custom"] == "nomikos {source:kraken; kind:polygon; baseline:derived;}"

    # A Baseline is an open polyline under the letter bodies, never the closed
    # polygon: a trainer handed the polygon here learns nothing about where the
    # line sits. The value is deterministic for this rectangle, at 0.68 of its
    # height measured down from the top: 10 + 0.68 * 20 = 23.6, rounded to 24.
    assert baseline.attrib["points"] == "11,24 119,24"
    points = [[int(part) for part in pair.split(",")] for pair in baseline.attrib["points"].split()]
    assert points != [[10, 10], [120, 10], [120, 30], [10, 30]]
    assert len(points) >= 2
    assert all(later[0] > earlier[0] for earlier, later in zip(points, points[1:], strict=False))
    assert all(10 <= x <= 120 and 10 <= y <= 30 for x, y in points)


# --- PAGE XML access control ---
# Tests outsiders cannot export member-route XML. Does not test public XML download.


def test_outsider_cannot_export_page_xml(
    client: TestClient,
    owner_headers: dict[str, str],
    outsider_headers: dict[str, str],
    owner_project: dict,
) -> None:
    project_id, document_id, part_id, _line_ids = _create_document_part_with_segments(
        client, owner_headers, owner_project
    )
    base = documents_url(project_id)

    response = client.get(
        f"{base}/{document_id}/parts/{part_id}/page-xml",
        headers=outsider_headers,
    )

    assert response.status_code in (403, 404)


# --- PAGE XML bundle ---
# Tests the zip pairs the XML with the full-resolution page image under one shared
# stem, and that the XML points at that sibling file. Does not validate against an XSD.

PAGE_NS = {"page": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"}


def test_member_exports_page_xml_bundle_with_full_resolution_image(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    project_id, document_id, part_id, _line_ids = _create_document_part_with_segments(
        client, owner_headers, owner_project
    )
    base = documents_url(project_id)

    response = client.get(
        f"{base}/{document_id}/parts/{part_id}/page-xml-bundle",
        headers=owner_headers,
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    # The document was created as "PDF codex" above; the stem is the document name with
    # whitespace collapsed plus the 1-based page position, shared by every file.
    assert response.headers["content-disposition"] == 'attachment; filename="PDF_codex_page_1.zip"'

    with ZipFile(BytesIO(response.content)) as archive:
        assert sorted(archive.namelist()) == ["PDF_codex_page_1.webp", "PDF_codex_page_1.xml"]
        # S314: parsing the response this test just asked the app to render, in-process.
        root = ElementTree.fromstring(archive.read("PDF_codex_page_1.xml"))  # noqa: S314
        image_bytes = archive.read("PDF_codex_page_1.webp")

    page = root.find("page:Page", PAGE_NS)
    assert page is not None
    assert page.attrib["imageFilename"] == "PDF_codex_page_1.webp"
    assert page.attrib["imageWidth"] == "160"
    assert page.attrib["imageHeight"] == "90"
    assert root.find(".//page:TextLine", PAGE_NS) is not None

    # The stored full-resolution object, not a thumbnail: same pixel grid the XML
    # coordinates were drawn on.
    with Image.open(BytesIO(image_bytes)) as image:
        assert image.format == "WEBP"
        assert image.size == (160, 90)

    # The bare XML export names the same sibling basename, never the storage key.
    bare = client.get(f"{base}/{document_id}/parts/{part_id}/page-xml", headers=owner_headers)
    assert bare.status_code == 200
    bare_page = ElementTree.fromstring(bare.content).find("page:Page", PAGE_NS)  # noqa: S314
    assert bare_page is not None
    assert bare_page.attrib["imageFilename"] == "PDF_codex_page_1.webp"


def test_page_xml_bundle_numbers_pages_by_position_not_by_order_value(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    project_id, document_id, first_part_id, _line_ids = _create_document_part_with_segments(
        client, owner_headers, owner_project
    )
    base = documents_url(project_id)
    upload = client.post(
        f"{base}/{document_id}/parts",
        headers=owner_headers,
        files={"file": ("page2.png", _png_bytes(), "image/png")},
    )
    assert upload.status_code == 201
    second_part_id = upload.json()["id"]

    second = client.get(
        f"{base}/{document_id}/parts/{second_part_id}/page-xml-bundle", headers=owner_headers
    )
    assert second.status_code == 200
    assert second.headers["content-disposition"] == 'attachment; filename="PDF_codex_page_2.zip"'

    # Deleting the first page leaves the second with order=1 but position 1: the name
    # follows what the reader sees in the page list, not the stored order value.
    deleted = client.delete(f"{base}/{document_id}/parts/{first_part_id}", headers=owner_headers)
    assert deleted.status_code == 204

    renumbered = client.get(
        f"{base}/{document_id}/parts/{second_part_id}/page-xml-bundle", headers=owner_headers
    )
    assert renumbered.status_code == 200
    assert (
        renumbered.headers["content-disposition"] == 'attachment; filename="PDF_codex_page_1.zip"'
    )


def test_outsider_cannot_export_page_xml_bundle(
    client: TestClient,
    owner_headers: dict[str, str],
    outsider_headers: dict[str, str],
    owner_project: dict,
) -> None:
    project_id, document_id, part_id, _line_ids = _create_document_part_with_segments(
        client, owner_headers, owner_project
    )
    base = documents_url(project_id)

    response = client.get(
        f"{base}/{document_id}/parts/{part_id}/page-xml-bundle",
        headers=outsider_headers,
    )

    assert response.status_code in (403, 404)
