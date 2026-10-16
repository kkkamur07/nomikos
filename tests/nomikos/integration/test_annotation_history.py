"""Annotation history API tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.fixtures.paths import MINIMAL_PNG
from tests.nomikos.integration.helpers import documents_url

pytestmark = pytest.mark.integration


def _create_part_with_lines(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> tuple[str, str, str, list[str]]:
    project_id = owner_project["id"]
    base = documents_url(project_id)
    create = client.post(base, headers=owner_headers, json={"name": "Recoverable codex"})
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
                    "approved_text": "alpha",
                },
                {
                    "order": 1,
                    "kind": "rectangle",
                    "points": [[0, 10], [10, 10], [10, 15], [0, 15]],
                    "source": "kraken",
                    "source_metadata": {"model": "blla"},
                    "kraken_ceiling": [[-1, 9], [11, 9], [11, 16], [-1, 16]],
                },
            ]
        },
    )
    assert replace.status_code == 200
    line_ids = [line["id"] for line in replace.json()]
    return project_id, document_id, part_id, line_ids


# --- Snapshot create and list ---
# Tests compact history payloads without bulky fields. Does not test retention pruning here.


def test_member_can_create_and_list_compact_history_snapshots(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    project_id, document_id, part_id, line_ids = _create_part_with_lines(
        client, owner_headers, owner_project
    )
    base = documents_url(project_id)
    history_url = f"{base}/{document_id}/parts/{part_id}/history"

    created = client.post(history_url, headers=owner_headers)

    assert created.status_code == 201
    snapshot = created.json()
    assert snapshot["part_id"] == part_id
    assert snapshot["line_count"] == 2
    assert snapshot["paired_line_count"] == 1
    assert snapshot["state"]["lines"] == [
        {
            "id": line_ids[0],
            "block_id": None,
            "order": 0,
            "kind": "polygon",
            "points": [[0, 0], [10, 0], [10, 5], [0, 5]],
            "source": "manual",
            "source_metadata": None,
            "kraken_ceiling": None,
            "approved_text": "alpha",
        },
        {
            "id": line_ids[1],
            "block_id": None,
            "order": 1,
            "kind": "rectangle",
            "points": [[0, 10], [10, 10], [10, 15], [0, 15]],
            "source": "kraken",
            "source_metadata": {"model": "blla"},
            "kraken_ceiling": [[-1, 9], [11, 9], [11, 16], [-1, 16]],
            "approved_text": None,
        },
    ]
    assert "image_key" not in snapshot["state"]
    assert "image_url" not in snapshot["state"]
    assert "exports" not in snapshot["state"]
    assert "events" not in snapshot["state"]

    listed = client.get(history_url, headers=owner_headers)

    assert listed.status_code == 200
    assert listed.json() == [
        {
            "id": snapshot["id"],
            "part_id": snapshot["part_id"],
            "line_count": snapshot["line_count"],
            "paired_line_count": snapshot["paired_line_count"],
            "created_at": snapshot["created_at"],
        }
    ]


# --- Snapshot restore ---
# Tests restore replaces current lines and transcriptions. Does not test concurrent edits.


def test_member_can_restore_snapshot_to_replace_current_page_annotations(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    project_id, document_id, part_id, line_ids = _create_part_with_lines(
        client, owner_headers, owner_project
    )
    base = documents_url(project_id)
    history_url = f"{base}/{document_id}/parts/{part_id}/history"
    snapshot_id = client.post(history_url, headers=owner_headers).json()["id"]

    replace_current = client.put(
        f"{base}/{document_id}/parts/{part_id}/lines",
        headers=owner_headers,
        json={
            "lines": [
                {
                    "id": line_ids[0],
                    "order": 0,
                    "kind": "polygon",
                    "points": [[100, 100], [110, 100], [110, 105], [100, 105]],
                    "source": "manual",
                    "approved_text": "changed",
                },
                {
                    "order": 1,
                    "kind": "polygon",
                    "points": [[200, 200], [210, 200], [210, 205], [200, 205]],
                    "source": "manual",
                    "approved_text": "new line",
                },
            ]
        },
    )
    assert replace_current.status_code == 200

    restored = client.post(
        f"{history_url}/{snapshot_id}/restore",
        headers=owner_headers,
    )

    assert restored.status_code == 200
    reloaded = client.get(
        f"{base}/{document_id}/parts/{part_id}/lines",
        headers=owner_headers,
    )
    assert reloaded.status_code == 200
    restored_lines = reloaded.json()
    assert [line["id"] for line in restored_lines] == line_ids
    assert [line["points"] for line in restored_lines] == [
        [[0, 0], [10, 0], [10, 5], [0, 5]],
        [[0, 10], [10, 10], [10, 15], [0, 15]],
    ]
    assert restored_lines[0]["line_transcriptions"][0]["text"] == "alpha"
    assert restored_lines[1]["line_transcriptions"] == []
    assert restored.json() == restored_lines


# --- Restore preserves block links ---
# Tests block_id associations survive restore. Does not test block geometry edits.


def test_restore_snapshot_preserves_line_block_associations(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    project_id, document_id, part_id, line_ids = _create_part_with_lines(
        client, owner_headers, owner_project
    )
    base = documents_url(project_id)
    history_url = f"{base}/{document_id}/parts/{part_id}/history"
    first_block = client.post(
        f"{base}/{document_id}/parts/{part_id}/blocks",
        headers=owner_headers,
        json={"order": 0, "box": {"points": [[0, 0], [10, 0], [10, 10], [0, 10]]}},
    )
    second_block = client.post(
        f"{base}/{document_id}/parts/{part_id}/blocks",
        headers=owner_headers,
        json={"order": 1, "box": {"points": [[20, 0], [30, 0], [30, 10], [20, 10]]}},
    )
    assert first_block.status_code == 201
    assert second_block.status_code == 201

    assign_first = client.patch(
        f"{base}/{document_id}/parts/{part_id}/lines/{line_ids[0]}",
        headers=owner_headers,
        json={"block_id": first_block.json()["id"]},
    )
    assert assign_first.status_code == 200
    snapshot_id = client.post(history_url, headers=owner_headers).json()["id"]

    assign_second = client.patch(
        f"{base}/{document_id}/parts/{part_id}/lines/{line_ids[0]}",
        headers=owner_headers,
        json={"block_id": second_block.json()["id"]},
    )
    assert assign_second.status_code == 200

    restored = client.post(
        f"{history_url}/{snapshot_id}/restore",
        headers=owner_headers,
    )

    assert restored.status_code == 200
    restored_first_line = next(line for line in restored.json() if line["id"] == line_ids[0])
    assert restored_first_line["block_id"] == first_block.json()["id"]


# --- History access control ---
# Tests non-members cannot create, list, or restore. Does not test published read-only access.


def test_non_member_cannot_create_list_or_restore_history_snapshots(
    client: TestClient,
    owner_headers: dict[str, str],
    outsider_headers: dict[str, str],
    owner_project: dict,
) -> None:
    project_id, document_id, part_id, _line_ids = _create_part_with_lines(
        client, owner_headers, owner_project
    )
    base = documents_url(project_id)
    history_url = f"{base}/{document_id}/parts/{part_id}/history"
    snapshot_id = client.post(history_url, headers=owner_headers).json()["id"]

    create = client.post(history_url, headers=outsider_headers)
    listed = client.get(history_url, headers=outsider_headers)
    restored = client.post(
        f"{history_url}/{snapshot_id}/restore",
        headers=outsider_headers,
    )

    assert create.status_code in (403, 404)
    assert listed.status_code in (403, 404)
    assert restored.status_code in (403, 404)


# --- Snapshot retention ---
# Tests only the latest five snapshots are kept. Does not test restore of pruned snapshots.


def test_history_snapshot_retention_keeps_only_latest_five(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    project_id, document_id, part_id, _line_ids = _create_part_with_lines(
        client, owner_headers, owner_project
    )
    base = documents_url(project_id)
    history_url = f"{base}/{document_id}/parts/{part_id}/history"

    snapshot_ids = []
    for _ in range(6):
        created = client.post(history_url, headers=owner_headers)
        assert created.status_code == 201
        snapshot_ids.append(created.json()["id"])

    listed = client.get(history_url, headers=owner_headers)

    assert listed.status_code == 200
    listed_ids = [snapshot["id"] for snapshot in listed.json()]
    assert len(listed_ids) == 5
    assert snapshot_ids[0] not in listed_ids
    assert snapshot_ids[-1] in listed_ids

    pruned_restore = client.post(
        f"{history_url}/{snapshot_ids[0]}/restore",
        headers=owner_headers,
    )
    assert pruned_restore.status_code == 404
