"""Page transcription Pairing workflow API tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.fixtures.paths import MINIMAL_PNG
from tests.nomikos.integration.helpers import documents_url

pytestmark = pytest.mark.integration


def _create_document_part_with_segments(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> tuple[str, str, str, list[str]]:
    project_id = owner_project["id"]
    base = documents_url(project_id)
    create = client.post(base, headers=owner_headers, json={"name": "Pairing codex"})
    assert create.status_code == 201
    document_id = create.json()["id"]

    upload = client.post(
        f"{base}/{document_id}/parts",
        headers=owner_headers,
        files={"file": ("page.png", MINIMAL_PNG, "image/png")},
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
                    "points": [[0, 0], [10, 0], [10, 5], [0, 5]],
                    "source": "manual",
                },
                {
                    "order": 1,
                    "kind": "polygon",
                    "points": [[0, 10], [10, 10], [10, 15], [0, 15]],
                    "source": "manual",
                },
            ]
        },
    )
    assert replace.status_code == 200
    line_ids = [line["id"] for line in replace.json()]
    return project_id, document_id, part_id, line_ids


# --- Page transcription import ---
# Tests text splits into candidate lines with zero progress. Does not create ground truth yet.


def test_import_page_transcription_splits_candidate_lines_without_ground_truth(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    project_id, document_id, part_id, _line_ids = _create_document_part_with_segments(
        client, owner_headers, owner_project
    )
    base = documents_url(project_id)

    import_response = client.put(
        f"{base}/{document_id}/parts/{part_id}/page-transcription",
        headers=owner_headers,
        json={"text": "alpha\n\nbeta\n"},
    )

    assert import_response.status_code == 200
    assert import_response.json()["text_lines"] == [
        {"order": 0, "text": "alpha", "paired_line_id": None},
        {"order": 1, "text": "beta", "paired_line_id": None},
    ]
    assert import_response.json()["pairing_progress"] == {
        "paired_lines": 0,
        "total_lines": 2,
        "percent": 0,
    }

    lines = client.get(f"{base}/{document_id}/parts/{part_id}/lines", headers=owner_headers)
    assert lines.status_code == 200
    assert [line["line_transcriptions"] for line in lines.json()] == [[], []]

    reloaded = client.get(
        f"{base}/{document_id}/parts/{part_id}/pairing",
        headers=owner_headers,
    )
    assert reloaded.status_code == 200
    assert reloaded.json() == import_response.json()


# --- Pairing candidate lines ---
# Tests pairing creates ground truth and updates progress. Does not test OCR import.


def test_pair_candidate_text_line_creates_ground_truth_and_updates_progress(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    project_id, document_id, part_id, line_ids = _create_document_part_with_segments(
        client, owner_headers, owner_project
    )
    base = documents_url(project_id)
    import_response = client.put(
        f"{base}/{document_id}/parts/{part_id}/page-transcription",
        headers=owner_headers,
        json={"text": "alpha\nbeta"},
    )
    assert import_response.status_code == 200

    pair = client.post(
        f"{base}/{document_id}/parts/{part_id}/pairings",
        headers=owner_headers,
        json={"line_id": line_ids[0], "text_line_order": 1},
    )

    assert pair.status_code == 200
    assert pair.json()["text_lines"] == [
        {"order": 0, "text": "alpha", "paired_line_id": None},
        {"order": 1, "text": "beta", "paired_line_id": line_ids[0]},
    ]
    assert pair.json()["pairing_progress"] == {
        "paired_lines": 1,
        "total_lines": 2,
        "percent": 50,
    }

    lines = client.get(f"{base}/{document_id}/parts/{part_id}/lines", headers=owner_headers)
    assert lines.status_code == 200
    assert lines.json()[0]["line_transcriptions"][0]["text"] == "beta"
    assert lines.json()[0]["line_transcriptions"][0]["transcription_kind"] == "ground_truth"
    assert lines.json()[1]["line_transcriptions"] == []


# --- Re-pairing and reimport ---
# Tests moving or replacing pairings clears stale ground truth. Does not test ML transcriptions.


def test_repairing_candidate_text_line_clears_previous_segment_ground_truth(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    project_id, document_id, part_id, line_ids = _create_document_part_with_segments(
        client, owner_headers, owner_project
    )
    base = documents_url(project_id)
    import_response = client.put(
        f"{base}/{document_id}/parts/{part_id}/page-transcription",
        headers=owner_headers,
        json={"text": "alpha\nbeta"},
    )
    assert import_response.status_code == 200

    first_pair = client.post(
        f"{base}/{document_id}/parts/{part_id}/pairings",
        headers=owner_headers,
        json={"line_id": line_ids[0], "text_line_order": 0},
    )
    assert first_pair.status_code == 200

    moved_pair = client.post(
        f"{base}/{document_id}/parts/{part_id}/pairings",
        headers=owner_headers,
        json={"line_id": line_ids[1], "text_line_order": 0},
    )

    assert moved_pair.status_code == 200
    assert moved_pair.json()["text_lines"] == [
        {"order": 0, "text": "alpha", "paired_line_id": line_ids[1]},
        {"order": 1, "text": "beta", "paired_line_id": None},
    ]
    assert moved_pair.json()["pairing_progress"] == {
        "paired_lines": 1,
        "total_lines": 2,
        "percent": 50,
    }

    lines = client.get(f"{base}/{document_id}/parts/{part_id}/lines", headers=owner_headers)
    assert lines.status_code == 200
    assert lines.json()[0]["line_transcriptions"] == []
    assert lines.json()[1]["line_transcriptions"][0]["text"] == "alpha"


def test_reimport_page_transcription_clears_stale_pairing_ground_truth(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    project_id, document_id, part_id, line_ids = _create_document_part_with_segments(
        client, owner_headers, owner_project
    )
    base = documents_url(project_id)
    import_response = client.put(
        f"{base}/{document_id}/parts/{part_id}/page-transcription",
        headers=owner_headers,
        json={"text": "alpha\nbeta"},
    )
    assert import_response.status_code == 200
    pair = client.post(
        f"{base}/{document_id}/parts/{part_id}/pairings",
        headers=owner_headers,
        json={"line_id": line_ids[0], "text_line_order": 0},
    )
    assert pair.status_code == 200

    reimport = client.put(
        f"{base}/{document_id}/parts/{part_id}/page-transcription",
        headers=owner_headers,
        json={"text": "gamma\ndelta"},
    )

    assert reimport.status_code == 200
    assert reimport.json()["text_lines"] == [
        {"order": 0, "text": "gamma", "paired_line_id": None},
        {"order": 1, "text": "delta", "paired_line_id": None},
    ]
    assert reimport.json()["pairing_progress"] == {
        "paired_lines": 0,
        "total_lines": 2,
        "percent": 0,
    }
    lines = client.get(f"{base}/{document_id}/parts/{part_id}/lines", headers=owner_headers)
    assert lines.status_code == 200
    assert [line["line_transcriptions"] for line in lines.json()] == [[], []]


# --- Direct ground-truth edits ---
# Tests manual text edits count toward pairing progress. Does not test review workflow gates.


def test_direct_ground_truth_text_edit_counts_toward_pairing_progress(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    project_id, document_id, part_id, line_ids = _create_document_part_with_segments(
        client, owner_headers, owner_project
    )
    base = documents_url(project_id)
    ground_truth_id = client.get(
        f"{base}/{document_id}/transcriptions", headers=owner_headers
    ).json()[0]["id"]

    edit = client.patch(
        f"{base}/{document_id}/transcriptions/{ground_truth_id}/lines/{line_ids[0]}",
        headers=owner_headers,
        json={"text": "typed approved text"},
    )
    assert edit.status_code == 200

    pairing = client.get(
        f"{base}/{document_id}/parts/{part_id}/pairing",
        headers=owner_headers,
    )
    assert pairing.status_code == 200
    assert pairing.json()["pairing_progress"] == {
        "paired_lines": 1,
        "total_lines": 2,
        "percent": 50,
    }


# --- Pairing access control ---
# Tests outsiders cannot toggle review status. Does not test read-only collaborators.


def test_outsider_cannot_change_part_review_status(
    client: TestClient,
    owner_headers: dict[str, str],
    outsider_headers: dict[str, str],
    owner_project: dict,
) -> None:
    project_id, document_id, part_id, _line_ids = _create_document_part_with_segments(
        client, owner_headers, owner_project
    )
    base = documents_url(project_id)

    denied = client.patch(
        f"{base}/{document_id}/parts/{part_id}",
        headers=outsider_headers,
        json={"reviewed": True},
    )

    assert denied.status_code in (403, 404)
    reloaded = client.get(f"{base}/{document_id}", headers=owner_headers)
    assert reloaded.status_code == 200
    assert reloaded.json()["parts"][0]["reviewed"] is False
