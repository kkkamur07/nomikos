"""Segment health against Postgres: the two statements a fake cannot check.

The service's unit tests hand it a fake repository, so they prove it *asks* for
the pairings and takes the part lock, and nothing at all about whether the SQL
underneath returns the right rows. Both queries are new, and between them they
decide whether a line may be deleted, so they are checked here against the real
database rather than against a stub that returns whatever the test set.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from tests.nomikos.integration.helpers import (  # noqa: F401
    documents_url,
    make_part,
    return_pooled_connections_before_leaving,
)

pytestmark = pytest.mark.integration

TWO_LINES = [
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


def _page_with_lines(
    client: TestClient, headers: dict[str, str], project: dict, *, name: str
) -> tuple[str, str, list[str]]:
    document_id, part_id = make_part(client, headers, project, name=name)
    base = documents_url(project["id"])
    replace = client.put(
        f"{base}/{document_id}/parts/{part_id}/lines",
        headers=headers,
        json={"lines": TWO_LINES},
    )
    assert replace.status_code == 200, replace.text
    return document_id, part_id, [line["id"] for line in replace.json()]


def _pair_first_line(
    client: TestClient, headers: dict[str, str], project: dict, ids: tuple[str, str, list[str]]
) -> str:
    document_id, part_id, line_ids = ids
    base = f"{documents_url(project['id'])}/{document_id}/parts/{part_id}"
    imported = client.put(
        f"{base}/page-transcription", headers=headers, json={"text": "alpha\nbeta"}
    )
    assert imported.status_code == 200, imported.text
    paired = client.post(
        f"{base}/pairings",
        headers=headers,
        json={"line_id": line_ids[0], "text_line_order": 0},
    )
    assert paired.status_code == 200, paired.text
    return line_ids[0]


def _paired_line_ids(client: TestClient, part_id: str) -> set[uuid.UUID]:
    """Call the repository on the loop that owns the async pool.

    ``asyncio.run`` would open a second loop and leave the pool's connections
    bound to the first, which is the failure ``return_pooled_connections_before_leaving``
    exists to clean up after. ``client.portal`` is the loop the app itself runs on.
    """
    from backend.document.infrastructure.document_repository import DocumentRepository
    from infrastructure.db import system_session

    async def run() -> set[uuid.UUID]:
        async with system_session() as session:
            return await DocumentRepository().paired_line_ids(session, uuid.UUID(part_id))

    return client.portal.call(run)


def test_paired_line_ids_finds_the_line_a_human_paired(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    ids = _page_with_lines(client, owner_headers, owner_project, name="Paired page")
    part_id = ids[1]
    assert _paired_line_ids(client, part_id) == set()

    line_id = _pair_first_line(client, owner_headers, owner_project, ids)

    assert _paired_line_ids(client, part_id) == {uuid.UUID(line_id)}


def test_paired_line_ids_does_not_leak_a_pairing_from_another_part(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    """The part filter, which is the easy half of this query to get wrong.

    Without it every page in the database would look fully paired, and segment
    health would quietly stop offering to delete anything at all: a bug that
    makes the feature useless rather than dangerous, and so one that no
    destructive-path test would ever catch.
    """
    paired_page = _page_with_lines(client, owner_headers, owner_project, name="Paired page")
    other_page = _page_with_lines(client, owner_headers, owner_project, name="Untouched page")
    _pair_first_line(client, owner_headers, owner_project, paired_page)

    assert _paired_line_ids(client, other_page[1]) == set()


def test_lock_part_is_a_statement_postgres_accepts(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    """``SELECT ... FOR UPDATE`` on a bare column is the kind of thing that
    compiles in SQLAlchemy and is refused by the server. Run it once for real."""
    from backend.document.infrastructure.document_repository import DocumentRepository
    from infrastructure.db import system_session

    _document_id, part_id, _line_ids = _page_with_lines(
        client, owner_headers, owner_project, name="Locked page"
    )

    async def run() -> None:
        async with system_session() as session:
            await DocumentRepository().lock_part(session, uuid.UUID(part_id))

    client.portal.call(run)


def test_segment_health_reports_a_real_page(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    """The route, wired to the real service, over a page that has no problems."""
    document_id, part_id, _line_ids = _page_with_lines(
        client, owner_headers, owner_project, name="Healthy page"
    )
    response = client.get(
        f"{documents_url(owner_project['id'])}/{document_id}/parts/{part_id}/segment-health",
        headers=owner_headers,
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["part_id"] == part_id
    assert body["line_count"] == 2
    assert body["finding_count"] == 0
    assert body["suspects"] == []
