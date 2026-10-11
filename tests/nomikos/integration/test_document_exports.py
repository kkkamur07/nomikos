"""Whole-document exports through the platform API.

The per-part exports are covered by ``test_transcription_pdf_artifact.py``. What is under
test here is what only exists at document scale: which pages an export covers, how they
are named and numbered inside it, and what happens when it covers none.
"""

from __future__ import annotations

import io
from io import BytesIO
from xml.etree import ElementTree
from zipfile import ZipFile

from fastapi.testclient import TestClient
from PIL import Image
from pypdf import PdfReader

from tests.nomikos.integration.helpers import documents_url

PAGE_NS = {"page": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"}


def _png_bytes(width: int = 160, height: int = 90) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def _create_document(
    client: TestClient, headers: dict[str, str], project_id: str, *, name: str
) -> str:
    created = client.post(documents_url(project_id), headers=headers, json={"name": name})
    assert created.status_code == 201, created.text
    return created.json()["id"]


def _add_page(
    client: TestClient,
    headers: dict[str, str],
    project_id: str,
    document_id: str,
    *,
    texts: list[str],
    reviewed: bool = False,
    size: tuple[int, int] = (160, 90),
) -> str:
    """One uploaded page carrying one approved line per entry in ``texts``."""
    base = f"{documents_url(project_id)}/{document_id}"
    upload = client.post(
        f"{base}/parts",
        headers=headers,
        files={"file": ("page.png", _png_bytes(*size), "image/png")},
    )
    assert upload.status_code == 201, upload.text
    part_id = upload.json()["id"]

    if texts:
        replace = client.put(
            f"{base}/parts/{part_id}/lines",
            headers=headers,
            json={
                "lines": [
                    {
                        "order": order,
                        "kind": "polygon",
                        "points": [
                            [10, 10 + order * 25],
                            [120, 10 + order * 25],
                            [120, 30 + order * 25],
                            [10, 30 + order * 25],
                        ],
                        "source": "manual",
                        "approved_text": text,
                    }
                    for order, text in enumerate(texts)
                ]
            },
        )
        assert replace.status_code == 200, replace.text

    if reviewed:
        marked = client.patch(f"{base}/parts/{part_id}", headers=headers, json={"reviewed": True})
        assert marked.status_code == 200, marked.text
    return part_id


def _chapter(
    client: TestClient,
    owner_headers: dict[str, str],
    owner_project: dict,
    *,
    name: str = "Chapter Four",
) -> tuple[str, str, list[str]]:
    """Three pages: the first and third reviewed, the second not."""
    project_id = owner_project["id"]
    document_id = _create_document(client, owner_headers, project_id, name=name)
    parts = [
        _add_page(
            client, owner_headers, project_id, document_id, texts=["alpha", "beta"], reviewed=True
        ),
        _add_page(client, owner_headers, project_id, document_id, texts=["gamma"]),
        _add_page(client, owner_headers, project_id, document_id, texts=["delta"], reviewed=True),
    ]
    return project_id, document_id, parts


def _export_url(project_id: str, document_id: str, artifact: str) -> str:
    return f"{documents_url(project_id)}/{document_id}/export/{artifact}"


# --- PAGE XML archive ---
# Tests the whole document arrives as one zip, page-numbered, with each XML pointing at
# the image shipped beside it. Does not validate against an XSD - the per-page bundle
# tests already cover the XML's shape.


def test_member_exports_every_page_as_page_xml_and_image(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    project_id, document_id, _parts = _chapter(client, owner_headers, owner_project)

    response = client.get(_export_url(project_id, document_id, "page-xml"), headers=owner_headers)

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    # The document was created as "Chapter Four" above: the archive is named for the
    # document, the files inside it for the pages.
    assert (
        response.headers["content-disposition"]
        == 'attachment; filename="chapter-four-page-xml.zip"'
    )

    with ZipFile(BytesIO(response.content)) as archive:
        assert sorted(archive.namelist()) == [
            "0001.webp",
            "0001.xml",
            "0002.webp",
            "0002.xml",
            "0003.webp",
            "0003.xml",
        ]
        # S314: parsing the response this test just asked the app to render, in-process.
        root = ElementTree.fromstring(archive.read("0002.xml"))  # noqa: S314
        image_bytes = archive.read("0002.webp")

    page = root.find("page:Page", PAGE_NS)
    assert page is not None
    # The XML names the sibling it actually ships with, not the per-page download's stem.
    assert page.attrib["imageFilename"] == "0002.webp"
    assert page.attrib["imageWidth"] == "160"
    assert page.attrib["imageHeight"] == "90"
    texts = [node.text for node in root.findall(".//page:TextEquiv/page:Unicode", PAGE_NS)]
    assert texts == ["gamma"]

    with Image.open(BytesIO(image_bytes)) as image:
        assert image.format == "WEBP"
        assert image.size == (160, 90)


def test_page_xml_archive_keeps_document_page_numbers_when_filtered_to_reviewed(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    """A reviewed-only archive names pages 1 and 3, not 1 and 2.

    Renumbering would make the same page cite differently depending on a query
    parameter, which is exactly what a page number is for.
    """
    project_id, document_id, _parts = _chapter(client, owner_headers, owner_project)

    response = client.get(
        _export_url(project_id, document_id, "page-xml"),
        headers=owner_headers,
        params={"reviewed_only": "true"},
    )

    assert response.status_code == 200
    with ZipFile(BytesIO(response.content)) as archive:
        assert sorted(archive.namelist()) == ["0001.webp", "0001.xml", "0003.webp", "0003.xml"]
        root = ElementTree.fromstring(archive.read("0003.xml"))  # noqa: S314
    texts = [node.text for node in root.findall(".//page:TextEquiv/page:Unicode", PAGE_NS)]
    assert texts == ["delta"]


def test_page_xml_archive_streams_rather_than_buffering_the_whole_zip(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    """The body arrives chunked, with no up-front ``Content-Length``.

    Not a style point: it is the observable difference between rendering a chapter of
    scans into one ``bytes`` and writing it out a page at a time.
    """
    project_id, document_id, _parts = _chapter(client, owner_headers, owner_project)

    with client.stream(
        "GET", _export_url(project_id, document_id, "page-xml"), headers=owner_headers
    ) as response:
        assert response.status_code == 200
        assert "content-length" not in response.headers
        body = b"".join(response.iter_bytes())

    with ZipFile(BytesIO(body)) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == 6


# --- Transcription PDF ---
# Tests one PDF for the whole document, one page per part, in part order.


def test_member_exports_one_transcription_pdf_for_the_whole_document(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    project_id, document_id, _parts = _chapter(client, owner_headers, owner_project)

    response = client.get(
        _export_url(project_id, document_id, "transcription-pdf"), headers=owner_headers
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert (
        response.headers["content-disposition"]
        == 'attachment; filename="chapter-four-transcription.pdf"'
    )
    assert response.content.startswith(b"%PDF")
    reader = PdfReader(io.BytesIO(response.content))
    assert len(reader.pages) == 3
    rendered = [page.extract_text() or "" for page in reader.pages]
    assert "alpha" in rendered[0]
    assert "gamma" in rendered[1]
    assert "delta" in rendered[2]


def test_document_pdf_page_matches_what_the_per_part_pdf_draws(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    """One page of the chapter PDF is the page the per-part route already serves.

    The document renderer drives one fpdf2 document so that every page lands in one
    file; this pins it to the single-page renderer's output so the two cannot drift.
    """
    project_id = owner_project["id"]
    document_id = _create_document(client, owner_headers, project_id, name="One page")
    part_id = _add_page(
        client, owner_headers, project_id, document_id, texts=["Αθήνα"], size=(200, 120)
    )

    document_pdf = client.get(
        _export_url(project_id, document_id, "transcription-pdf"), headers=owner_headers
    )
    part_pdf = client.get(
        f"{documents_url(project_id)}/{document_id}/parts/{part_id}/transcription-pdf",
        headers=owner_headers,
    )

    assert document_pdf.status_code == 200
    assert part_pdf.status_code == 200
    document_page = PdfReader(io.BytesIO(document_pdf.content)).pages[0]
    part_page = PdfReader(io.BytesIO(part_pdf.content)).pages[0]
    assert document_page.extract_text() == part_page.extract_text()
    assert document_page.mediabox.width == part_page.mediabox.width == 200
    assert document_page.mediabox.height == part_page.mediabox.height == 120


def test_document_pdf_keeps_a_page_that_has_no_transcription(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    project_id = owner_project["id"]
    document_id = _create_document(client, owner_headers, project_id, name="Gappy")
    _add_page(client, owner_headers, project_id, document_id, texts=["first"])
    _add_page(client, owner_headers, project_id, document_id, texts=[])
    _add_page(client, owner_headers, project_id, document_id, texts=["third"])

    response = client.get(
        _export_url(project_id, document_id, "transcription-pdf"), headers=owner_headers
    )

    assert response.status_code == 200
    reader = PdfReader(io.BytesIO(response.content))
    # Three pages, not two: dropping the blank one would silently make "page 3" page 2.
    assert len(reader.pages) == 3
    assert (reader.pages[1].extract_text() or "").strip() == ""
    assert "third" in (reader.pages[2].extract_text() or "")


def test_document_pdf_covers_only_reviewed_pages_when_asked(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    project_id, document_id, _parts = _chapter(client, owner_headers, owner_project)

    response = client.get(
        _export_url(project_id, document_id, "transcription-pdf"),
        headers=owner_headers,
        params={"reviewed_only": "true"},
    )

    assert response.status_code == 200
    reader = PdfReader(io.BytesIO(response.content))
    assert len(reader.pages) == 2
    rendered = " ".join(page.extract_text() or "" for page in reader.pages)
    assert "alpha" in rendered
    assert "delta" in rendered
    # The unreviewed middle page is the whole point of the filter.
    assert "gamma" not in rendered


# --- Plain text ---
# Tests the transcription as text, page markers included.


def test_member_exports_document_text_with_page_markers(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    project_id, document_id, _parts = _chapter(client, owner_headers, owner_project)

    response = client.get(_export_url(project_id, document_id, "text"), headers=owner_headers)

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/plain; charset=utf-8"
    assert (
        response.headers["content-disposition"]
        == 'attachment; filename="chapter-four-transcription.txt"'
    )
    assert response.text == "[p.1]\nalpha\nbeta\n\n[p.2]\ngamma\n\n[p.3]\ndelta\n"


def test_document_text_marks_a_page_that_has_no_transcription(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    project_id = owner_project["id"]
    document_id = _create_document(client, owner_headers, project_id, name="Gappy")
    _add_page(client, owner_headers, project_id, document_id, texts=["first"])
    _add_page(client, owner_headers, project_id, document_id, texts=[])

    response = client.get(_export_url(project_id, document_id, "text"), headers=owner_headers)

    assert response.status_code == 200
    assert response.text == "[p.1]\nfirst\n\n[p.2]\n"


def test_document_text_covers_only_reviewed_pages_when_asked(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    project_id, document_id, _parts = _chapter(client, owner_headers, owner_project)

    response = client.get(
        _export_url(project_id, document_id, "text"),
        headers=owner_headers,
        params={"reviewed_only": "true"},
    )

    assert response.status_code == 200
    # Markers stay at the document's own numbering, and the unreviewed page is gone.
    assert response.text == "[p.1]\nalpha\nbeta\n\n[p.3]\ndelta\n"


def test_document_text_carries_a_non_ascii_transcription_intact(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    project_id = owner_project["id"]
    document_id = _create_document(client, owner_headers, project_id, name="Greek")
    _add_page(client, owner_headers, project_id, document_id, texts=["μῆνιν ἄειδε θεά"])

    response = client.get(_export_url(project_id, document_id, "text"), headers=owner_headers)

    assert response.status_code == 200
    assert response.content.decode("utf-8") == "[p.1]\nμῆνιν ἄειδε θεά\n"


def test_export_of_a_greek_titled_document_names_the_download_in_greek(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    """RFC 6266: an ASCII fallback the header can carry, plus the real name beside it."""
    project_id = owner_project["id"]
    document_id = _create_document(client, owner_headers, project_id, name="Ιλιάς Α")
    _add_page(client, owner_headers, project_id, document_id, texts=["alpha"])

    response = client.get(_export_url(project_id, document_id, "text"), headers=owner_headers)

    assert response.status_code == 200
    disposition = response.headers["content-disposition"]
    assert disposition.startswith('attachment; filename="_-_-transcription.txt"')
    assert disposition.endswith(
        "filename*=UTF-8''%CE%B9%CE%BB%CE%B9%CE%AC%CF%82-%CE%B1-transcription.txt"
    )


# --- Nothing to export ---
# Tests all three endpoints refuse rather than hand back an empty artifact.


def test_exports_are_404_for_a_document_with_no_pages(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    project_id = owner_project["id"]
    document_id = _create_document(client, owner_headers, project_id, name="Empty")

    for artifact in ("page-xml", "transcription-pdf", "text"):
        response = client.get(_export_url(project_id, document_id, artifact), headers=owner_headers)
        assert response.status_code == 404, f"{artifact}: {response.text}"


def test_exports_are_404_when_reviewed_only_matches_no_pages(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    """An empty archive would download as a success and read as corruption.

    A 404 is the only answer that distinguishes "nothing is reviewed yet" from "the
    export broke", and it is the same answer for all three artifacts.
    """
    project_id = owner_project["id"]
    document_id = _create_document(client, owner_headers, project_id, name="Unreviewed")
    _add_page(client, owner_headers, project_id, document_id, texts=["alpha"])

    for artifact in ("page-xml", "transcription-pdf", "text"):
        response = client.get(
            _export_url(project_id, document_id, artifact),
            headers=owner_headers,
            params={"reviewed_only": "true"},
        )
        assert response.status_code == 404, f"{artifact}: {response.text}"
        # Not a stream that opens and then dies: the refusal is the whole response.
        assert response.json()["error"]["code"] == "NOT_FOUND"


# --- Access control ---
# Tests the document-level exports are gated exactly like the per-part ones.


def test_outsider_cannot_export_a_document(
    client: TestClient,
    owner_headers: dict[str, str],
    outsider_headers: dict[str, str],
    owner_project: dict,
) -> None:
    project_id, document_id, _parts = _chapter(client, owner_headers, owner_project)

    for artifact in ("page-xml", "transcription-pdf", "text"):
        response = client.get(
            _export_url(project_id, document_id, artifact), headers=outsider_headers
        )
        assert response.status_code in (403, 404), f"{artifact}: {response.text}"


def test_anonymous_caller_cannot_export_a_document(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    project_id, document_id, _parts = _chapter(client, owner_headers, owner_project)

    for artifact in ("page-xml", "transcription-pdf", "text"):
        response = client.get(_export_url(project_id, document_id, artifact))
        assert response.status_code == 401, f"{artifact}: {response.text}"
