"""The free merge route against Postgres.

The service's unit tests hand it a fake repository, so they prove which row it
decides to keep and nothing about what the database then does with the rows it
deletes. Three things can only be checked here: that a pairing pointing at a
deleted line is released rather than raising a foreign key error, that a caller
without access to the project is refused, and that ids from another page are a
404 instead of a merge across pages.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from tests.nomikos.integration.helpers import (  # noqa: F401
    assert_api_error,
    documents_url,
    make_part,
    return_pooled_connections_before_leaving,
)


def _row(order: int, top: float) -> dict:
    return {
        "order": order,
        "kind": "polygon",
        "points": [[0, top], [100, top], [100, top + 40], [0, top + 40]],
        "source": "manual",
    }


THREE_LINES = [_row(0, 0), _row(1, 60), _row(2, 120)]


def _page(
    client: TestClient, headers: dict[str, str], project: dict, *, name: str
) -> tuple[str, str, list[str]]:
    document_id, part_id = make_part(client, headers, project, name=name)
    replace = client.put(
        f"{documents_url(project['id'])}/{document_id}/parts/{part_id}/lines",
        headers=headers,
        json={"lines": THREE_LINES},
    )
    assert replace.status_code == 200, replace.text
    return document_id, part_id, [line["id"] for line in replace.json()]


def _merge_url(project: dict, document_id: str, part_id: str) -> str:
    return f"{documents_url(project['id'])}/{document_id}/parts/{part_id}/lines/merge"


def _pair(
    client: TestClient,
    headers: dict[str, str],
    project: dict,
    document_id: str,
    part_id: str,
    line_id: str,
    order: int,
) -> None:
    base = f"{documents_url(project['id'])}/{document_id}/parts/{part_id}"
    imported = client.put(
        f"{base}/page-transcription", headers=headers, json={"text": "alpha\nbeta\ngamma"}
    )
    assert imported.status_code == 200, imported.text
    paired = client.post(
        f"{base}/pairings",
        headers=headers,
        json={"line_id": line_id, "text_line_order": order},
    )
    assert paired.status_code == 200, paired.text


def test_two_lines_merge_into_one_and_the_rest_of_the_page_closes_up(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    document_id, part_id, line_ids = _page(client, owner_headers, owner_project, name="Merge page")

    response = client.post(
        _merge_url(owner_project, document_id, part_id),
        headers=owner_headers,
        json={"line_ids": [line_ids[0], line_ids[1]]},
    )

    assert response.status_code == 200, response.text
    lines = response.json()
    assert [line["id"] for line in lines] == [line_ids[0], line_ids[2]]
    assert [line["order"] for line in lines] == [0, 1]
    merged = lines[0]
    assert merged["manual_geometry"] is True
    ys = [point[1] for point in merged["points"]]
    assert min(ys) == 0 and max(ys) == 100


def test_the_line_carrying_the_pairing_survives_and_the_others_release_theirs(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    """A foreign key with ``ON DELETE SET NULL`` behind it, exercised for real."""
    document_id, part_id, line_ids = _page(
        client, owner_headers, owner_project, name="Paired merge page"
    )
    _pair(client, owner_headers, owner_project, document_id, part_id, line_ids[1], 1)

    response = client.post(
        _merge_url(owner_project, document_id, part_id),
        headers=owner_headers,
        json={"line_ids": [line_ids[0], line_ids[1]]},
    )

    assert response.status_code == 200, response.text
    assert [line["id"] for line in response.json()] == [line_ids[1], line_ids[2]]
    pairing = client.get(
        f"{documents_url(owner_project['id'])}/{document_id}/parts/{part_id}/pairing",
        headers=owner_headers,
    )
    assert pairing.status_code == 200, pairing.text
    paired_to = {
        text_line["order"]: text_line["paired_line_id"]
        for text_line in pairing.json()["text_lines"]
    }
    assert paired_to[1] == line_ids[1]


def test_two_paired_lines_are_refused_with_both_ids_until_one_is_named(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    document_id, part_id, line_ids = _page(
        client, owner_headers, owner_project, name="Two paired page"
    )
    _pair(client, owner_headers, owner_project, document_id, part_id, line_ids[0], 0)
    client.post(
        f"{documents_url(owner_project['id'])}/{document_id}/parts/{part_id}/pairings",
        headers=owner_headers,
        json={"line_id": line_ids[1], "text_line_order": 1},
    )

    refused = client.post(
        _merge_url(owner_project, document_id, part_id),
        headers=owner_headers,
        json={"line_ids": [line_ids[0], line_ids[1]]},
    )

    assert refused.status_code == 409, refused.text
    error = refused.json()["error"]
    assert error["code"] == "MERGE_TEXT_CONFLICT"
    assert set(error["line_ids"]) == {line_ids[0], line_ids[1]}

    answered = client.post(
        _merge_url(owner_project, document_id, part_id),
        headers=owner_headers,
        json={"line_ids": [line_ids[0], line_ids[1]], "keep_text_from": line_ids[1]},
    )
    assert answered.status_code == 200, answered.text
    assert [line["id"] for line in answered.json()] == [line_ids[1], line_ids[2]]


def test_a_line_from_another_page_is_a_404_and_neither_page_changes(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    document_id, part_id, line_ids = _page(client, owner_headers, owner_project, name="This page")
    _other_document, _other_part, other_ids = _page(
        client, owner_headers, owner_project, name="Other page"
    )

    response = client.post(
        _merge_url(owner_project, document_id, part_id),
        headers=owner_headers,
        json={"line_ids": [line_ids[0], other_ids[0]]},
    )

    assert response.status_code == 404, response.text
    assert_api_error(response, code="NOT_FOUND")
    still_there = client.get(
        f"{documents_url(owner_project['id'])}/{document_id}/parts/{part_id}/lines",
        headers=owner_headers,
    )
    assert [line["id"] for line in still_there.json()] == line_ids


def test_a_signed_in_outsider_cannot_merge_lines_on_someone_elses_page(
    client: TestClient,
    owner_headers: dict[str, str],
    owner_project: dict,
    outsider_headers: dict[str, str],
) -> None:
    document_id, part_id, line_ids = _page(
        client, owner_headers, owner_project, name="Private page"
    )

    response = client.post(
        _merge_url(owner_project, document_id, part_id),
        headers=outsider_headers,
        json={"line_ids": [line_ids[0], line_ids[1]]},
    )

    assert response.status_code in (403, 404), response.text
    still_there = client.get(
        f"{documents_url(owner_project['id'])}/{document_id}/parts/{part_id}/lines",
        headers=owner_headers,
    )
    assert len(still_there.json()) == 3


def test_one_id_is_refused_by_the_schema_before_anything_is_read(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    document_id, part_id, line_ids = _page(client, owner_headers, owner_project, name="One id page")

    response = client.post(
        _merge_url(owner_project, document_id, part_id),
        headers=owner_headers,
        json={"line_ids": [line_ids[0]]},
    )

    assert response.status_code == 422, response.text
