"""Project CRUD and sharing integration tests — real Postgres (kalamos)."""

import uuid

import pytest

from tests.nomikos.integration.helpers import assert_api_error

# --- Project list and CRUD ---
# Tests owner create/read/update/delete. Does not test documents or sharing.


@pytest.mark.integration
def test_list_projects_empty_for_new_user(client, auth_headers):
    response = client.get("/projects", headers=auth_headers)
    assert response.status_code == 200
    assert response.json() == {"items": [], "next_cursor": None}


@pytest.mark.integration
@pytest.mark.parametrize("cursor", ["%%%", "x" * 2048])
def test_malformed_project_cursor_is_bounded_and_sanitized(client, auth_headers, cursor):
    response = client.get(f"/projects?cursor={cursor}", headers=auth_headers)

    assert response.status_code == 422
    error = assert_api_error(response, code="VALIDATION_ERROR")
    assert error["message"] in {"Invalid request", "Invalid pagination cursor"}
    assert cursor not in response.text
    assert response.headers["x-error-id"]


@pytest.mark.integration
def test_owner_create_read_update_delete(client, owner_headers):
    slug = f"proj-{uuid.uuid4().hex[:8]}"
    create = client.post(
        "/projects",
        headers=owner_headers,
        json={"slug": slug, "name": "My Manuscripts", "guidelines": "Use polytonic Greek"},
    )
    assert create.status_code == 201
    body = create.json()
    project_id = body["id"]
    assert body["slug"] == slug
    assert body["name"] == "My Manuscripts"
    assert body["guidelines"] == "Use polytonic Greek"
    assert body["document_count"] == 0
    assert "owner_id" in body

    read = client.get(f"/projects/{project_id}", headers=owner_headers)
    assert read.status_code == 200
    assert read.json()["slug"] == slug

    update = client.patch(
        f"/projects/{project_id}",
        headers=owner_headers,
        json={"name": "Renamed", "guidelines": None},
    )
    assert update.status_code == 200
    assert update.json()["name"] == "Renamed"
    assert update.json()["guidelines"] is None

    listed = client.get("/projects", headers=owner_headers)
    assert listed.status_code == 200
    ids = [p["id"] for p in listed.json()["items"]]
    assert project_id in ids

    delete = client.delete(f"/projects/{project_id}", headers=owner_headers)
    assert delete.status_code == 204

    gone = client.get(f"/projects/{project_id}", headers=owner_headers)
    assert gone.status_code == 404


# --- Slug uniqueness ---
# Tests duplicate slugs return 409. Does not test slug format validation.


@pytest.mark.integration
def test_create_duplicate_slug_conflict(client, owner_headers):
    slug = f"dup-{uuid.uuid4().hex[:8]}"
    first = client.post(
        "/projects",
        headers=owner_headers,
        json={"slug": slug, "name": "First"},
    )
    assert first.status_code == 201
    second = client.post(
        "/projects",
        headers=owner_headers,
        json={"slug": slug, "name": "Second"},
    )
    assert second.status_code == 409


# --- Project sharing ---
# Tests share/unshare and collaborator list access. Does not test collaborator mutations.


@pytest.mark.integration
def test_share_and_collaborator_list_read(
    client, owner_headers, collaborator_headers, collaborator_user
):
    slug = f"share-{uuid.uuid4().hex[:8]}"
    create = client.post(
        "/projects",
        headers=owner_headers,
        json={"slug": slug, "name": "Shared workspace"},
    )
    assert create.status_code == 201
    project_id = create.json()["id"]

    share = client.post(
        f"/projects/{project_id}/share",
        headers=owner_headers,
        json={"username": collaborator_user["username"]},
    )
    assert share.status_code == 204

    collab_list = client.get("/projects", headers=collaborator_headers)
    assert collab_list.status_code == 200
    assert any(p["id"] == project_id for p in collab_list.json()["items"])

    collab_read = client.get(f"/projects/{project_id}", headers=collaborator_headers)
    assert collab_read.status_code == 200
    assert collab_read.json()["slug"] == slug

    collaborator_id = client.get(f"/projects/{project_id}/share", headers=owner_headers).json()[0][
        "id"
    ]
    unshare = client.delete(
        f"/projects/{project_id}/share/{collaborator_id}",
        headers=owner_headers,
    )
    assert unshare.status_code == 204

    collab_list_after = client.get("/projects", headers=collaborator_headers)
    assert collab_list_after.status_code == 200
    assert not any(p["id"] == project_id for p in collab_list_after.json()["items"])

    collab_read_after = client.get(f"/projects/{project_id}", headers=collaborator_headers)
    assert collab_read_after.status_code in (403, 404)


# --- Sharing by email ---
# Tests the email path and the collaborator list. Does not test username sharing.


@pytest.mark.integration
def test_share_by_email_and_list_collaborators(
    client, owner_headers, collaborator_headers, collaborator_user, outsider_headers
):
    slug = f"share-email-{uuid.uuid4().hex[:8]}"
    create = client.post(
        "/projects",
        headers=owner_headers,
        json={"slug": slug, "name": "Shared by email"},
    )
    assert create.status_code == 201
    project_id = create.json()["id"]

    empty = client.get(f"/projects/{project_id}/share", headers=owner_headers)
    assert empty.status_code == 200
    assert empty.json() == []

    share = client.post(
        f"/projects/{project_id}/share",
        headers=owner_headers,
        json={"email": collaborator_user["email"].upper()},
    )
    assert share.status_code == 204

    listed = client.get(f"/projects/{project_id}/share", headers=owner_headers)
    assert listed.status_code == 200
    assert [c["username"] for c in listed.json()] == [collaborator_user["username"]]
    assert listed.json()[0]["email"] == collaborator_user["email"]

    # The list carries emails, so only the owner may read it.
    as_collaborator = client.get(f"/projects/{project_id}/share", headers=collaborator_headers)
    assert as_collaborator.status_code == 403
    as_outsider = client.get(f"/projects/{project_id}/share", headers=outsider_headers)
    assert as_outsider.status_code in (403, 404)

    collab_read = client.get(f"/projects/{project_id}", headers=collaborator_headers)
    assert collab_read.status_code == 200

    again = client.post(
        f"/projects/{project_id}/share",
        headers=owner_headers,
        json={"email": collaborator_user["email"]},
    )
    assert again.status_code == 409

    unknown = client.post(
        f"/projects/{project_id}/share",
        headers=owner_headers,
        json={"email": f"nobody-{uuid.uuid4().hex[:6]}@test.kalamos"},
    )
    assert unknown.status_code == 404

    both = client.post(
        f"/projects/{project_id}/share",
        headers=owner_headers,
        json={"email": collaborator_user["email"], "username": collaborator_user["username"]},
    )
    assert both.status_code == 422
    neither = client.post(f"/projects/{project_id}/share", headers=owner_headers, json={})
    assert neither.status_code == 422


# --- Removing a collaborator ---
# Tests removal by id, including names a URL path cannot carry. Does not test sharing.


@pytest.mark.integration
def test_a_collaborator_whose_username_contains_a_slash_can_be_removed(client, owner_headers):
    """Registration constrains only a username's length, so `scribe/anna` is a
    real account name. While removal addressed the collaborator by username in
    the path, that name split into two path segments, matched no route and
    answered 404 no matter how the client encoded it: the person could be
    shared with and then never removed."""
    suffix = uuid.uuid4().hex[:8]
    odd = {
        "email": f"scribe-{suffix}@test.kalamos",
        "username": f"scribe/{suffix}",
        "password": "test-pass-123",
    }
    assert client.post("/auth/register", json=odd).status_code == 201

    project_id = client.post(
        "/projects", headers=owner_headers, json={"slug": f"slash-{suffix}", "name": "Slash"}
    ).json()["id"]
    assert (
        client.post(
            f"/projects/{project_id}/share",
            headers=owner_headers,
            json={"identifier": odd["username"]},
        ).status_code
        == 204
    )

    listed = client.get(f"/projects/{project_id}/share", headers=owner_headers).json()
    assert [c["username"] for c in listed] == [odd["username"]]

    removed = client.delete(
        f"/projects/{project_id}/share/{listed[0]['id']}", headers=owner_headers
    )
    assert removed.status_code == 204
    assert client.get(f"/projects/{project_id}/share", headers=owner_headers).json() == []


@pytest.mark.integration
def test_removing_someone_who_is_not_a_collaborator_is_404(
    client, owner_headers, outsider_headers, collaborator_user
):
    suffix = uuid.uuid4().hex[:8]
    project_id = client.post(
        "/projects", headers=owner_headers, json={"slug": f"rm-{suffix}", "name": "Remove"}
    ).json()["id"]

    stranger = client.delete(f"/projects/{project_id}/share/{uuid.uuid4()}", headers=owner_headers)
    assert stranger.status_code == 404

    # A user who exists but was never shared with reads the same way.
    client.post(
        f"/projects/{project_id}/share",
        headers=owner_headers,
        json={"username": collaborator_user["username"]},
    )
    collaborator_id = client.get(f"/projects/{project_id}/share", headers=owner_headers).json()[0][
        "id"
    ]
    assert (
        client.delete(
            f"/projects/{project_id}/share/{collaborator_id}", headers=owner_headers
        ).status_code
        == 204
    )
    assert (
        client.delete(
            f"/projects/{project_id}/share/{collaborator_id}", headers=owner_headers
        ).status_code
        == 404
    )

    # Not the owner's to remove, and not a UUID at all.
    assert client.delete(
        f"/projects/{project_id}/share/{uuid.uuid4()}", headers=outsider_headers
    ).status_code in (403, 404)
    assert (
        client.delete(f"/projects/{project_id}/share/not-a-uuid", headers=owner_headers).status_code
        == 422
    )


# --- Non-member access ---
# Tests outsiders cannot read or mutate projects. Does not test anonymous access.


@pytest.mark.integration
def test_non_member_cannot_read_or_mutate(client, owner_headers, outsider_headers, outsider_user):
    slug = f"private-{uuid.uuid4().hex[:8]}"
    create = client.post(
        "/projects",
        headers=owner_headers,
        json={"slug": slug, "name": "Private"},
    )
    assert create.status_code == 201
    project_id = create.json()["id"]

    read = client.get(f"/projects/{project_id}", headers=outsider_headers)
    assert read.status_code in (403, 404)

    patch = client.patch(
        f"/projects/{project_id}",
        headers=outsider_headers,
        json={"name": "Hijacked"},
    )
    assert patch.status_code in (403, 404)

    delete = client.delete(f"/projects/{project_id}", headers=outsider_headers)
    assert delete.status_code in (403, 404)

    share = client.post(
        f"/projects/{project_id}/share",
        headers=outsider_headers,
        json={"username": outsider_user["username"]},
    )
    assert share.status_code in (403, 404)


# --- Collaborator restrictions ---
# Tests collaborators can read but not update or delete. Does not test document access.


@pytest.mark.integration
def test_collaborator_cannot_update_or_delete(
    client, owner_headers, collaborator_headers, collaborator_user
):
    slug = f"collab-{uuid.uuid4().hex[:8]}"
    create = client.post(
        "/projects",
        headers=owner_headers,
        json={"slug": slug, "name": "Team project"},
    )
    assert create.status_code == 201
    project_id = create.json()["id"]
    share = client.post(
        f"/projects/{project_id}/share",
        headers=owner_headers,
        json={"username": collaborator_user["username"]},
    )
    assert share.status_code == 204

    patch = client.patch(
        f"/projects/{project_id}",
        headers=collaborator_headers,
        json={"name": "Not allowed"},
    )
    assert patch.status_code == 403

    delete = client.delete(f"/projects/{project_id}", headers=collaborator_headers)
    assert delete.status_code == 403
