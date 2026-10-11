"""Regions (ADR 0011): drawing one assigns the lines inside it and renumbers the page."""

from __future__ import annotations

from fastapi.testclient import TestClient

from tests.fixtures.paths import MINIMAL_PNG
from tests.nomikos.integration.helpers import documents_url


def _box(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _spread(client: TestClient, headers: dict[str, str], project_id: str) -> tuple[str, str]:
    """A two-folio page whose lines are numbered row by row across both folios."""
    base = documents_url(project_id)
    document_id = client.post(base, headers=headers, json={"name": "Spread"}).json()["id"]
    upload = client.post(
        f"{base}/{document_id}/parts",
        headers=headers,
        files={"file": ("page.png", MINIMAL_PNG, "image/png")},
    )
    assert upload.status_code == 201
    part_id = upload.json()["id"]
    lines = []
    for row in range(2):
        for x0, x1 in ((50, 650), (750, 1400)):
            lines.append(
                {
                    "order": len(lines),
                    "kind": "polygon",
                    "points": _box(x0, 100 + 50 * row, x1, 140 + 50 * row),
                    "source": "manual",
                }
            )
    replace = client.put(
        f"{base}/{document_id}/parts/{part_id}/lines", headers=headers, json={"lines": lines}
    )
    assert replace.status_code == 200
    return document_id, part_id


def test_a_region_takes_the_lines_inside_it_and_they_are_numbered_first(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    project_id = owner_project["id"]
    document_id, part_id = _spread(client, owner_headers, project_id)
    part_url = f"{documents_url(project_id)}/{document_id}/parts/{part_id}"

    created = client.post(
        f"{part_url}/blocks",
        headers=owner_headers,
        json={"order": 0, "box": {"points": _box(700, 0, 1450, 400)}, "label": "5r"},
    )

    assert created.status_code == 201
    block = created.json()
    assert block["label"] == "5r"
    lines = client.get(f"{part_url}/lines", headers=owner_headers).json()
    by_order = sorted(lines, key=lambda line: line["order"])
    assert [line["order"] for line in by_order] == [0, 1, 2, 3]
    assert [line["block_id"] for line in by_order] == [block["id"], block["id"], None, None]
    assert all(line["points"][0][0] == 750 for line in by_order[:2])

    relabelled = client.patch(
        f"{part_url}/blocks/{block['id']}", headers=owner_headers, json={"label": None}
    )
    assert relabelled.status_code == 200
    assert relabelled.json()["label"] is None

    deleted = client.delete(f"{part_url}/blocks/{block['id']}", headers=owner_headers)
    assert deleted.status_code == 204
    after = client.get(f"{part_url}/lines", headers=owner_headers).json()
    assert all(line["block_id"] is None for line in after)
    assert sorted(line["order"] for line in after) == [0, 1, 2, 3]


def test_assign_contained_false_creates_an_empty_region(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    project_id = owner_project["id"]
    document_id, part_id = _spread(client, owner_headers, project_id)
    part_url = f"{documents_url(project_id)}/{document_id}/parts/{part_id}"

    created = client.post(
        f"{part_url}/blocks",
        headers=owner_headers,
        json={"order": 0, "box": {"points": _box(0, 0, 1450, 400)}, "assign_contained": False},
    )

    assert created.status_code == 201
    lines = client.get(f"{part_url}/lines", headers=owner_headers).json()
    assert all(line["block_id"] is None for line in lines)
