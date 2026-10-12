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


def _segment_like_blla(part_id: str, width: int, height: int) -> None:
    """Run a segment merge with blla's shape: one block covering the whole image."""
    import uuid

    from sqlalchemy import text
    from sqlalchemy.orm import Session

    from backend.document.application.segment_merge_service import SegmentMergeService
    from backend.ml.domain.segment import CanonicalBlock, CanonicalLine, CanonicalSegmentResult
    from infrastructure.db import sync_engine

    lines = []
    for row in range(3):
        for x0, x1 in ((50, 650), (750, 1400)):
            y0 = 100 + 50 * row
            lines.append(
                CanonicalLine(
                    external_id=f"blla-line-{len(lines) + 1}",
                    order=len(lines),
                    block_external_id="blla-block-1",
                    baseline={"points": [[x0, y0 + 35], [x1, y0 + 35]]},
                    points=_box(x0, y0, x1, y0 + 40),
                )
            )
    segment = CanonicalSegmentResult(
        blocks=[
            CanonicalBlock(
                external_id="blla-block-1", order=0, box={"points": _box(0, 0, width, height)}
            )
        ],
        lines=lines,
    )
    with Session(sync_engine) as session:
        session.execute(
            text("update document_parts set width = :w, height = :h where id = :p"),
            {"w": width, "h": height, "p": part_id},
        )
        session.commit()
        SegmentMergeService().apply_sync(
            session, part_id=uuid.UUID(part_id), canonical_segment=segment, job_id=uuid.uuid4()
        )


def test_on_a_blla_segmented_page_regions_and_drawn_lines_work(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    project_id = owner_project["id"]
    base = documents_url(project_id)
    document_id = client.post(base, headers=owner_headers, json={"name": "Seg"}).json()["id"]
    part_id = client.post(
        f"{base}/{document_id}/parts",
        headers=owner_headers,
        files={"file": ("page.png", MINIMAL_PNG, "image/png")},
    ).json()["id"]
    part_url = f"{base}/{document_id}/parts/{part_id}"
    _segment_like_blla(part_id, 1450, 400)

    layout = client.get(f"{part_url}/layout", headers=owner_headers).json()
    assert layout["blocks"] == []
    assert all(line["block_id"] is None for line in layout["lines"])

    # Recto drawn first, verso second, neither with an order: the verso still reads first.
    recto = client.post(
        f"{part_url}/blocks",
        headers=owner_headers,
        json={"box": {"points": _box(700, 0, 1450, 400)}},
    ).json()
    verso = client.post(
        f"{part_url}/blocks", headers=owner_headers, json={"box": {"points": _box(0, 0, 700, 400)}}
    ).json()
    blocks = {
        b["id"]: b["order"]
        for b in client.get(f"{part_url}/layout", headers=owner_headers).json()["blocks"]
    }
    assert (blocks[verso["id"]], blocks[recto["id"]]) == (0, 1)

    # A line drawn between recto rows 0 and 1, sent with no block_id, joins the recto there.
    created = client.post(
        f"{part_url}/lines",
        headers=owner_headers,
        json={"kind": "polygon", "points": _box(760, 142, 1390, 148)},
    )
    assert created.status_code == 201
    assert created.json()["block_id"] == recto["id"]
    lines = sorted(
        client.get(f"{part_url}/lines", headers=owner_headers).json(), key=lambda l: l["order"]
    )
    recto_run = [line["id"] for line in lines if line["block_id"] == recto["id"]]
    assert recto_run.index(created.json()["id"]) == 1
    assert [line["order"] for line in lines] == list(range(7))
