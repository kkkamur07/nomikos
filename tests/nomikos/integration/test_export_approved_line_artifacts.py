"""Export approved Line artifacts through the platform API."""

from __future__ import annotations

import base64
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tests.nomikos.integration.helpers import documents_url

pytestmark = pytest.mark.integration


def _png_bytes(width: int = 80, height: int = 40) -> bytes:
    buf = BytesIO()
    Image.new("RGB", (width, height), "white").save(buf, format="PNG")
    return buf.getvalue()


def _create_document_part_with_segments(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> tuple[str, str, str, list[str]]:
    project_id = owner_project["id"]
    base = documents_url(project_id)
    create = client.post(base, headers=owner_headers, json={"name": "Export codex"})
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
                    "points": [[5, 5], [45, 5], [45, 15], [5, 15]],
                    "source": "manual",
                },
                {
                    "order": 1,
                    "kind": "polygon",
                    "points": [[5, 20], [45, 20], [45, 30], [5, 30]],
                    "source": "manual",
                },
            ]
        },
    )
    assert replace.status_code == 200
    line_ids = [line["id"] for line in replace.json()]
    return project_id, document_id, part_id, line_ids


# --- Line artifact export ---
# Tests paired line crops and transcriptions export with warnings. Does not run ML.


def test_member_exports_paired_line_image_and_transcription_with_warnings(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    project_id, document_id, part_id, line_ids = _create_document_part_with_segments(
        client, owner_headers, owner_project
    )
    base = documents_url(project_id)
    import_response = client.put(
        f"{base}/{document_id}/parts/{part_id}/page-transcription",
        headers=owner_headers,
        json={"text": "alpha\nunused"},
    )
    assert import_response.status_code == 200
    pair = client.post(
        f"{base}/{document_id}/parts/{part_id}/pairings",
        headers=owner_headers,
        json={"line_id": line_ids[0], "text_line_order": 0},
    )
    assert pair.status_code == 200

    response = client.post(
        f"{base}/{document_id}/parts/{part_id}/export",
        headers=owner_headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["exported_count"] == 1
    assert body["warnings"] == {
        "unpaired_segments": [2],
        "unused_text_lines": [1],
    }
    assert body["artifacts"] == [
        {
            "line_id": line_ids[0],
            "segment_number": 1,
            "image_filename": "page_1.jpg",
            "transcription_filename": "page_1.txt",
            "transcription_text": "alpha",
            "image_base64": body["artifacts"][0]["image_base64"],
        }
    ]
    image_bytes = base64.b64decode(body["artifacts"][0]["image_base64"])
    assert image_bytes.startswith(b"\xff\xd8")
    assert Image.open(BytesIO(image_bytes)).size == (40, 10)


# --- Export access control ---
# Tests outsiders cannot export. Does not test published/public export routes.


def test_outsider_cannot_export_approved_line_artifacts(
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
        f"{base}/{document_id}/parts/{part_id}/export",
        headers=outsider_headers,
    )

    assert response.status_code in (403, 404)
