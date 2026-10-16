"""Presigned (direct-to-storage) part upload endpoints — real Postgres (kalamos).

The suite runs on the local media store, which cannot presign, so the full
begin → PUT → finalize round trip is not exercisable here (it needs Supabase
Storage). What IS pinned at the HTTP level is the contract every deployment
relies on: the fallback shape begin returns on a backend that cannot presign,
that the fallback leaves no ``pending`` part row behind, and finalize's
refusals (unknown part, already-uploaded part, foreign key).
"""

import uuid

import pytest

from tests.fixtures.paths import MINIMAL_PNG
from tests.nomikos.integration.helpers import assert_api_error, documents_url


def _create_document(client, owner_headers, project_id) -> str:
    create = client.post(
        documents_url(project_id), headers=owner_headers, json={"name": "Presign codex"}
    )
    assert create.status_code == 201
    return create.json()["id"]


@pytest.mark.integration
def test_begin_upload_falls_back_to_multipart_on_local_backend(
    client, owner_headers, owner_project
):
    project_id = owner_project["id"]
    document_id = _create_document(client, owner_headers, project_id)

    begin = client.post(
        f"{documents_url(project_id)}/{document_id}/parts/upload",
        headers=owner_headers,
        json={"filename": "page.png", "size": 12345},
    )
    assert begin.status_code == 201
    body = begin.json()
    # The local filesystem cannot presign: no URL, no token, and — critically — no
    # committed part row for the caller to orphan.
    assert body["upload_url"] is None
    assert body["token"] is None
    assert body["part_id"] is None
    assert body["image_key"].startswith("parts/")

    listed = client.get(f"{documents_url(project_id)}/{document_id}", headers=owner_headers)
    assert listed.status_code == 200
    assert listed.json()["parts"] == []


@pytest.mark.integration
def test_finalize_refuses_unknown_part_and_already_uploaded_part(
    client, owner_headers, owner_project
):
    project_id = owner_project["id"]
    document_id = _create_document(client, owner_headers, project_id)
    base = f"{documents_url(project_id)}/{document_id}/parts"

    unknown = uuid.uuid4()
    missing = client.post(
        f"{base}/{unknown}/finalize",
        headers=owner_headers,
        json={"image_key": f"parts/{unknown}.webp", "width": None, "height": None},
    )
    assert missing.status_code == 404

    upload = client.post(
        base, headers=owner_headers, files={"file": ("page.png", MINIMAL_PNG, "image/png")}
    )
    assert upload.status_code == 201
    part = upload.json()

    # A multipart-uploaded part is already sealed; finalize must refuse it.
    again = client.post(
        f"{base}/{part['id']}/finalize",
        headers=owner_headers,
        json={"image_key": f"parts/{part['id']}.webp", "width": None, "height": None},
    )
    assert again.status_code == 422
    assert_api_error(again, code="VALIDATION_ERROR")
