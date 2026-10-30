"""Public access policy - anonymous read of published documents, gated by share token."""

import pytest

from tests.fixtures.paths import MINIMAL_PNG


@pytest.fixture
def published_document(client, owner_headers, owner_project):
    project_id = owner_project["id"]
    base = f"/projects/{project_id}/documents"
    create = client.post(base, headers=owner_headers, json={"name": "Public codex"})

    assert create.status_code == 201

    document_id = create.json()["id"]
    publish = client.patch(
        f"{base}/{document_id}",
        headers=owner_headers,
        json={"workflow": "published"},
    )
    assert publish.status_code == 200
    token = publish.json()["public_share_token"]
    assert token
    upload = client.post(
        f"{base}/{document_id}/parts",
        headers=owner_headers,
        files={"file": ("page.png", MINIMAL_PNG, "image/png")},
    )
    assert upload.status_code == 201
    part_id = upload.json()["id"]
    return {
        "project_id": project_id,
        "document_id": document_id,
        "part_id": part_id,
        "token": token,
    }


# --- Anonymous read of published documents, correct token required ---
# Tests public document detail for published workflow, with and without the share
# token. Does not allow draft access.


@pytest.mark.integration
def test_anonymous_can_read_published_document_with_the_right_token(client, published_document):
    project_id = published_document["project_id"]
    document_id = published_document["document_id"]
    token = published_document["token"]
    url = f"/public/projects/{project_id}/documents/{document_id}"

    response = client.get(url, params={"t": token})
    assert response.status_code == 200
    body = response.json()
    assert body["workflow"] == "published"
    assert len(body["parts"]) == 1
    assert body["parts"][0]["image_url"] == f"/public/media/parts/{published_document['part_id']}"
    # The owner-only secret never rides along in a response anyone anonymous can reach.
    assert "public_share_token" not in body


@pytest.mark.integration
def test_anonymous_read_with_the_wrong_token_is_not_found(client, published_document):
    project_id = published_document["project_id"]
    document_id = published_document["document_id"]
    url = f"/public/projects/{project_id}/documents/{document_id}"

    response = client.get(url, params={"t": "not-the-real-token"})
    assert response.status_code == 404


# --- Anonymous mutations blocked ---
# Tests unauthenticated users cannot change published documents. Does not test member routes.


@pytest.mark.integration
def test_anonymous_cannot_mutate_published_document(client, published_document):
    project_id = published_document["project_id"]
    document_id = published_document["document_id"]
    member_url = f"/projects/{project_id}/documents/{document_id}"

    patch = client.patch(member_url, json={"name": "Hacked"})
    assert patch.status_code == 401

    upload = client.post(
        f"{member_url}/parts",
        files={"file": ("x.png", MINIMAL_PNG, "image/png")},
    )
    assert upload.status_code == 401


# --- Outsider vs public route ---
# Tests outsiders use /public (with the token) for read access, and never see the
# token itself. Does not grant member-route access.


@pytest.mark.integration
def test_outsider_can_read_published_via_public_route(client, outsider_headers, published_document):
    project_id = published_document["project_id"]
    document_id = published_document["document_id"]
    token = published_document["token"]
    member_url = f"/projects/{project_id}/documents/{document_id}"
    public_url = f"/public/projects/{project_id}/documents/{document_id}"

    denied = client.get(member_url, headers=outsider_headers)
    assert denied.status_code == 403

    allowed = client.get(public_url, params={"t": token})
    assert allowed.status_code == 200

    # An authenticated outsider is not a member either - the token is still required.
    no_token = client.get(public_url, headers=outsider_headers)
    assert no_token.status_code == 404


# --- Rotating the share token invalidates the old link ---
# Tests the rotate endpoint mints a new secret and the old one stops working immediately.


@pytest.mark.integration
def test_rotating_the_share_token_invalidates_the_old_link_and_the_new_one_works(
    client, owner_headers, published_document
):
    project_id = published_document["project_id"]
    document_id = published_document["document_id"]
    old_token = published_document["token"]
    public_url = f"/public/projects/{project_id}/documents/{document_id}"
    rotate_url = f"/projects/{project_id}/documents/{document_id}/share-token/rotate"

    assert client.get(public_url, params={"t": old_token}).status_code == 200

    rotate = client.post(rotate_url, headers=owner_headers)
    assert rotate.status_code == 200
    new_token = rotate.json()["public_share_token"]
    assert new_token
    assert new_token != old_token

    stale = client.get(public_url, params={"t": old_token})
    assert stale.status_code == 404

    fresh = client.get(public_url, params={"t": new_token})
    assert fresh.status_code == 200


@pytest.mark.integration
def test_rotating_the_share_token_is_owner_only(
    client, owner_headers, collaborator_user, collaborator_headers, published_document
):
    """A collaborator reaches past ``require_document`` where an outsider never does,
    so this is the request that actually exercises the ``is_owner`` guard."""
    project_id = published_document["project_id"]
    document_id = published_document["document_id"]
    rotate_url = f"/projects/{project_id}/documents/{document_id}/share-token/rotate"

    share = client.post(
        f"/projects/{project_id}/share",
        headers=owner_headers,
        json={"username": collaborator_user["username"]},
    )
    assert share.status_code == 204

    denied = client.post(rotate_url, headers=collaborator_headers)
    assert denied.status_code == 403


# --- Public layout and transcriptions require the token too ---
# Tests anonymous access to layout and transcription layers. Public part media is
# covered end to end in test_part_media_variants.py. Does not test export zip.


@pytest.mark.integration
def test_anonymous_gets_layout_and_transcriptions_with_the_token(client, published_document):
    project_id = published_document["project_id"]
    document_id = published_document["document_id"]
    token = published_document["token"]
    base = f"/public/projects/{project_id}/documents/{document_id}"

    layout = client.get(f"{base}/layout", params={"t": token})
    assert layout.status_code == 200
    # `blocks_truncated` is part of the public layout contract as of ce74fdc: an
    # anonymous caller is told when the block list was cut short rather than
    # silently receiving a partial page. False here because this document has none.
    assert layout.json() == {
        "blocks": [],
        "blocks_truncated": False,
        "lines": [],
        "next_cursor": None,
    }

    no_token = client.get(f"{base}/layout")
    assert no_token.status_code == 404

    over_limit = client.get(f"{base}/layout", params={"t": token, "limit": 10_001})
    assert over_limit.status_code == 422

    layers = client.get(f"{base}/transcriptions", params={"t": token})
    assert layers.status_code == 200
    assert layers.json()[0]["kind"] == "ground_truth"

    layers_no_token = client.get(f"{base}/transcriptions")
    assert layers_no_token.status_code == 404


# --- Public artifact downloads require the token too ---
# Tests anonymous PDF/XML on public routes; member routes still require auth.


@pytest.mark.integration
def test_anonymous_can_download_published_part_artifacts_with_the_token(
    client, published_document, owner_headers
):
    project_id = published_document["project_id"]
    document_id = published_document["document_id"]
    part_id = published_document["part_id"]
    token = published_document["token"]
    base = f"/public/projects/{project_id}/documents/{document_id}/parts/{part_id}"

    pdf = client.get(f"{base}/transcription-pdf", params={"t": token})
    assert pdf.status_code == 200
    assert pdf.headers["content-type"] == "application/pdf"
    assert pdf.content.startswith(b"%PDF")
    assert client.get(f"{base}/transcription-pdf").status_code == 404

    xml = client.get(f"{base}/page-xml", params={"t": token})
    assert xml.status_code == 200
    assert xml.headers["content-type"] == "application/xml"
    assert xml.content.startswith(b"<?xml")
    assert client.get(f"{base}/page-xml").status_code == 404

    bundle = client.get(f"{base}/page-xml-bundle", params={"t": token})
    assert bundle.status_code == 200
    assert bundle.headers["content-type"] == "application/zip"
    assert bundle.content.startswith(b"PK")
    assert bundle.headers["content-disposition"].endswith('_page_1.zip"')
    assert client.get(f"{base}/page-xml-bundle").status_code == 404

    draft_pdf = client.get(
        f"/projects/{project_id}/documents/{document_id}/parts/{part_id}/transcription-pdf"
    )
    assert draft_pdf.status_code == 401


# --- Per-page publishing: a page can be held back from an otherwise public document ---
# Tests the bulk publish-flag endpoint and every public read path it must gate: the
# document-with-parts response, the layout listing, and public media. Owner reads are
# unaffected - the toggle exists so the owner can see and flip it.


@pytest.mark.integration
def test_unpublished_part_is_hidden_from_every_public_surface_but_visible_to_the_owner(
    client, owner_headers, published_document
):
    project_id = published_document["project_id"]
    document_id = published_document["document_id"]
    part_id = published_document["part_id"]
    token = published_document["token"]
    parts_url = f"/projects/{project_id}/documents/{document_id}/parts/published"

    hold_back = client.patch(
        parts_url,
        headers=owner_headers,
        json={"parts": [{"part_id": part_id, "published": False}]},
    )
    assert hold_back.status_code == 200
    assert hold_back.json()[0]["published"] is False

    public_doc = client.get(
        f"/public/projects/{project_id}/documents/{document_id}", params={"t": token}
    )
    assert public_doc.status_code == 200
    assert public_doc.json()["parts"] == []

    public_media = client.get(f"/public/media/parts/{part_id}", params={"t": token})
    assert public_media.status_code == 404

    owner_doc = client.get(f"/projects/{project_id}/documents/{document_id}", headers=owner_headers)
    assert owner_doc.status_code == 200
    owner_parts = owner_doc.json()["parts"]
    assert len(owner_parts) == 1
    assert owner_parts[0]["published"] is False

    # Flip it back: the public surface recovers immediately, no republish needed.
    restore = client.patch(
        parts_url,
        headers=owner_headers,
        json={"parts": [{"part_id": part_id, "published": True}]},
    )
    assert restore.status_code == 200
    assert restore.json()[0]["published"] is True

    public_media_again = client.get(f"/public/media/parts/{part_id}", params={"t": token})
    assert public_media_again.status_code == 200


@pytest.mark.integration
def test_setting_published_flag_is_refused_to_a_collaborator(
    client, owner_headers, collaborator_user, collaborator_headers, published_document
):
    """The owner check, not the membership check.

    An outsider is already turned away by ``require_document`` before ownership is
    ever consulted - delete the ``is_owner`` guard in ``DocumentPartService`` and an
    outsider-only check still passes. The share below is what makes this one
    different: the caller is a member, so the request gets past ``require_project``
    and reaches the guard, and this is the only test that fails if the guard goes.
    The editor hides the control from
    non-owners on the strength of it being enforced.
    """
    project_id = published_document["project_id"]
    document_id = published_document["document_id"]
    part_id = published_document["part_id"]

    share = client.post(
        f"/projects/{project_id}/share",
        headers=owner_headers,
        json={"username": collaborator_user["username"]},
    )
    assert share.status_code == 204

    # Asserted, not assumed. No fixture makes the collaborator a member, so without
    # the share above this reads 403 exactly like the outsider does, and the PATCH
    # below would prove nothing about ownership.
    member_read = client.get(
        f"/projects/{project_id}/documents/{document_id}", headers=collaborator_headers
    )
    assert member_read.status_code == 200

    denied = client.patch(
        f"/projects/{project_id}/documents/{document_id}/parts/published",
        headers=collaborator_headers,
        json={"parts": [{"part_id": part_id, "published": False}]},
    )
    assert denied.status_code == 403


@pytest.mark.integration
def test_republishing_keeps_the_share_link_that_was_already_handed_out(
    client, owner_headers, published_document
):
    """A draft round trip must not rotate the token.

    ``update_document`` mints only when the token is null, and the comment there
    promises exactly this. Nothing tested it: mutation testing flipped that
    ``is None`` to ``is not None`` and every test still passed. If it ever
    regressed, every link already sent would 404 the next time an owner toggled a
    document off and back on, which is the one failure this whole feature exists to
    prevent.
    """
    project_id = published_document["project_id"]
    document_id = published_document["document_id"]
    url = f"/projects/{project_id}/documents/{document_id}"
    original = published_document["token"]

    unpublish = client.patch(url, headers=owner_headers, json={"workflow": "draft"})
    assert unpublish.status_code == 200

    republish = client.patch(url, headers=owner_headers, json={"workflow": "published"})
    assert republish.status_code == 200
    assert republish.json()["public_share_token"] == original

    # And the link itself still opens the document, not just the column.
    still_live = client.get(
        f"/public/projects/{project_id}/documents/{document_id}", params={"t": original}
    )
    assert still_live.status_code == 200


# --- The share token is the owner's to hand out, not every collaborator's ---
# Tests that a member who is not the owner never sees the secret on any owner-facing
# read, while the owner does. Does not test rotation, which is covered above.


@pytest.mark.integration
def test_a_collaborator_never_sees_the_share_token(
    client, owner_headers, collaborator_user, collaborator_headers, published_document
):
    """Anyone holding the token can hand an anonymous link to the whole document to
    anyone at all, and the owner has no way to see that it happened - the only remedy
    left is rotation, which breaks every link already sent. Publishing and rotating are
    owner-only for exactly that reason, so reading the secret has to be too.
    """
    project_id = published_document["project_id"]
    document_id = published_document["document_id"]
    share = client.post(
        f"/projects/{project_id}/share",
        headers=owner_headers,
        json={"username": collaborator_user["username"]},
    )
    assert share.status_code == 204

    base = f"/projects/{project_id}/documents"
    detail = client.get(f"{base}/{document_id}", headers=collaborator_headers)
    assert detail.status_code == 200
    assert detail.json()["public_share_token"] is None

    listing = client.get(base, headers=collaborator_headers)
    assert listing.status_code == 200
    assert all(item["public_share_token"] is None for item in listing.json()["items"])

    # A write the collaborator *is* allowed to make must not leak it on the way back.
    renamed = client.patch(
        f"{base}/{document_id}",
        headers=collaborator_headers,
        json={"name": "Renamed by collaborator"},
    )
    assert renamed.status_code == 200
    assert renamed.json()["public_share_token"] is None

    # The owner still gets it, or there would be no way to share the document at all.
    owner_detail = client.get(f"{base}/{document_id}", headers=owner_headers)
    assert owner_detail.status_code == 200
    assert owner_detail.json()["public_share_token"] == published_document["token"]


# --- The bulk published flag is bounded work ---
# A foreign id is refused before any write.


@pytest.mark.integration
def test_a_foreign_part_id_in_the_batch_writes_nothing_at_all(
    client, owner_headers, published_document
):
    project_id = published_document["project_id"]
    document_id = published_document["document_id"]
    part_id = published_document["part_id"]

    response = client.patch(
        f"/projects/{project_id}/documents/{document_id}/parts/published",
        headers=owner_headers,
        json={
            "parts": [
                {"part_id": part_id, "published": False},
                {"part_id": "00000000-0000-0000-0000-000000000001", "published": False},
            ]
        },
    )
    assert response.status_code == 404

    # The valid half of the batch must not have landed.
    detail = client.get(f"/projects/{project_id}/documents/{document_id}", headers=owner_headers)
    assert detail.json()["parts"][0]["published"] is True


# --- The public contract says what it means ---
# Tests that the token is absent from the public schema, not merely stripped at
# serialisation time.


@pytest.mark.integration
def test_the_public_document_schema_does_not_advertise_the_share_token(client):
    """``response_model_exclude`` kept the value off the wire but left the field in the
    OpenAPI, so every generated client was told a public body carries a secret it can
    never contain.
    """
    schema = client.get("/openapi.json").json()
    route = schema["paths"]["/public/projects/{project_id}/documents/{document_id}"]
    ref = route["get"]["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]
    model = schema["components"]["schemas"][ref.rsplit("/", 1)[-1]]
    assert "public_share_token" not in model["properties"]
