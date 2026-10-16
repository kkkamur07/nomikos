"""The signed page image link, end to end against the real app and real storage.

Everything here goes through ``create_app()`` and the configured media store.
There is no stub storage client and no local FastAPI app - ADR 0001 records what
a test-local app cost this project the last time, and a faked media store would
make the one claim under test ("the link reaches real bytes in the real store")
unfalsifiable.

The media store exercised here is the **local** backend, which is what this
repository runs outside production. Its signature is minted and checked by the
platform. The Supabase backend signs with Storage's own key, and there is no
Supabase running in this environment - the platform's refusal on that backend is
covered by ``test_the_platform_refuses_to_serve_objects_on_the_supabase_backend``.

**No test here sleeps.** Expiry is proven by asking the real signer for a link
whose deadline is already in the past, and by showing that a deadline moved
forward in the URL no longer verifies. A ``sleep(61)`` would assert the same
thing sixty-one seconds more slowly and flake on a loaded machine.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from urllib.parse import parse_qs, unquote, urlsplit

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

import infrastructure.models  # noqa: F401 - register all ORM mappers
from backend.core.settings import reset_settings_caches
from backend.document.infrastructure.media_store import (
    SIGNED_MEDIA_PREFIX,
    get_media_store,
    sign_object_path,
)
from backend.document.infrastructure.orm_models import DocumentPart
from backend.ml.application.device_auth import DEVICE_TOKEN_HEADER
from infrastructure.db import sync_system_session
from tests.nomikos.integration.helpers import (
    MINIMAL_PNG,
    claim_page,
    device_headers,
    documents_url,
    stored_minimal_page_bytes,
)
from tests.nomikos.integration.helpers import prefer_local as _prefer_local
from tests.nomikos.integration.helpers import running_agent as _running_agent

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def serve_objects_from_the_local_filesystem(monkeypatch):
    """Pin the storage backend rather than inherit whatever is configured.

    Every test here but one is about the path where *the platform* signs and
    serves the object, which is `STORAGE_BACKEND=local`. Settings resolve from
    `backend/core/.env`, falling back to `.env.supabase` - both gitignored, so
    whether this module talks to the local filesystem or to a real Supabase
    project depended on which untracked file happened to exist. It passed in a
    fresh worktree and, in a checkout with Supabase credentials, uploaded
    manuscript pages to live Storage and then asserted against its URLs.

    The one test that wants the production profile monkeypatches it back.
    """
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    reset_settings_caches()
    yield
    reset_settings_caches()


# ---------------------------------------------------------------------------
# Live fixtures: real pairing, real upload, real claim
# ---------------------------------------------------------------------------


def _claim(client: TestClient, device_token: str) -> dict:
    """The claim body, already asserted: no test in this file is about the status
    code - they are all about the **signed link** that comes back on the page."""
    response = claim_page(client, device_headers(device_token))
    assert response.status_code == 200, response.text
    return response.json()


def _document_with_parts(
    client: TestClient, headers: dict[str, str], project: dict, *, pages: int = 1
) -> tuple[str, list[str]]:
    created = client.post(documents_url(project["id"]), headers=headers, json={"name": "Codex"})
    assert created.status_code == 201, created.text
    document_id = created.json()["id"]
    base = f"{documents_url(project['id'])}/{document_id}/parts"
    part_ids = []
    for page in range(pages):
        upload = client.post(
            base, headers=headers, files={"file": (f"page{page}.png", MINIMAL_PNG, "image/png")}
        )
        assert upload.status_code == 201, upload.text
        part_ids.append(upload.json()["id"])
    return document_id, part_ids


def _submit_segment(
    client: TestClient, headers: dict[str, str], project: dict, document_id: str, part_id: str
) -> str:
    response = client.post(
        f"{documents_url(project['id'])}/{document_id}/parts/{part_id}/segment", headers=headers
    )
    assert response.status_code == 202, response.text
    return response.json()["job_id"]


def _claimed_page(client: TestClient, owner_headers: dict[str, str], project: dict, **kw) -> dict:
    """Pair an agent, submit one page locally, and claim it. Returns the page."""
    agent = _running_agent(client, owner_headers)
    _prefer_local(client, owner_headers)
    document_id, part_ids = _document_with_parts(client, owner_headers, project, **kw)
    _submit_segment(client, owner_headers, project, document_id, part_ids[0])
    body = _claim(client, agent["device_token"])
    page = body["page"]
    assert page is not None, body
    page["_server_time"] = body["server_time"]
    page["_part_ids"] = part_ids
    return page


def _image_key(part_id: str) -> str:
    with sync_system_session() as session:
        return session.execute(
            select(DocumentPart.image_key).where(DocumentPart.id == uuid.UUID(part_id))
        ).scalar_one()


def _link_parts(url: str) -> tuple[str, str, str]:
    """Split a signed link into (image_key, expires, signature)."""
    split = urlsplit(url)
    assert split.path.startswith(f"{SIGNED_MEDIA_PREFIX}/"), url
    query = parse_qs(split.query)
    return (
        unquote(split.path[len(SIGNED_MEDIA_PREFIX) + 1 :]),
        query["expires"][0],
        query["signature"][0],
    )


def _link(image_key: str, expires: str, signature: str) -> str:
    return f"{SIGNED_MEDIA_PREFIX}/{image_key}?expires={expires}&signature={signature}"


def _seconds_until(moment: str, *, of: str) -> float:
    return (datetime.fromisoformat(moment) - datetime.fromisoformat(of)).total_seconds()


# ---------------------------------------------------------------------------
# The link points at exactly this page's image
# ---------------------------------------------------------------------------


def test_the_claim_carries_a_link_to_exactly_this_pages_image(
    client: TestClient, owner_user, owner_headers, owner_project
) -> None:
    page = _claimed_page(client, owner_headers, owner_project)

    signed_key, _, _ = _link_parts(page["page_image_url"])

    assert signed_key == _image_key(page["_part_ids"][0])
    # One object, spelled out in full - not a bucket, not the ``parts/`` prefix,
    # not the document. The key ends at a file.
    assert signed_key.count("/") >= 1 and not signed_key.endswith("/")
    assert page["page_image_expires_at"]


def test_the_link_fetches_the_real_image_bytes_with_no_device_credential(
    client: TestClient, owner_user, owner_headers, owner_project
) -> None:
    """The signature *is* the authorization (ADR 0002).

    No ``Authorization``, no ``X-Nomikos-Device-Token``, no cookie - and the
    bytes that come back are the same bytes the authenticated media route serves,
    read from the same store.
    """
    page = _claimed_page(client, owner_headers, owner_project)

    fetched = client.get(page["page_image_url"])

    assert fetched.status_code == 200, fetched.text
    # Proof about the request that was actually sent, not about what we meant.
    sent = fetched.request.headers
    assert DEVICE_TOKEN_HEADER.lower() not in {name.lower() for name in sent}
    assert "authorization" not in {name.lower() for name in sent}
    assert "cookie" not in {name.lower() for name in sent}

    signed_key, _, _ = _link_parts(page["page_image_url"])
    assert fetched.content == stored_minimal_page_bytes()
    assert fetched.content == get_media_store().read(signed_key)
    assert fetched.headers["content-type"].startswith("image/")


# ---------------------------------------------------------------------------
# The link dies
# ---------------------------------------------------------------------------


def test_an_already_expired_signature_is_refused(
    client: TestClient, owner_user, owner_headers, owner_project
) -> None:
    """Expiry proven by controlling the issuing deadline, not by sleeping.

    The link below is minted by the platform's own signer for the same object,
    with a deadline a minute in the past - which is exactly the state a real link
    reaches sixty seconds after a claim.
    """
    page = _claimed_page(client, owner_headers, owner_project)
    signed_key, _, _ = _link_parts(page["page_image_url"])
    assert client.get(page["page_image_url"]).status_code == 200

    expired = sign_object_path(signed_key, expires_at=datetime.now(UTC) - timedelta(seconds=60))

    assert client.get(expired).status_code == 403


def test_the_deadline_cannot_be_pushed_forward_by_the_holder(
    client: TestClient, owner_user, owner_headers, owner_project
) -> None:
    """The expiry is inside the signed message, so it is not a suggestion."""
    page = _claimed_page(client, owner_headers, owner_project)
    signed_key, expires, signature = _link_parts(page["page_image_url"])

    extended = _link(signed_key, str(int(expires) + 86_400), signature)

    assert client.get(extended).status_code == 403


# ---------------------------------------------------------------------------
# The link reaches one object, and only that one
# ---------------------------------------------------------------------------


def test_the_signature_does_not_carry_over_to_a_sibling_page(
    client: TestClient, owner_user, owner_headers, owner_project
) -> None:
    """A second page of the same document, in the same store, under the same
    prefix, owned by the same researcher - and the link still will not open it."""
    page = _claimed_page(client, owner_headers, owner_project, pages=2)
    claimed_key, expires, signature = _link_parts(page["page_image_url"])
    sibling_key = _image_key(page["_part_ids"][1])

    assert sibling_key != claimed_key
    # The sibling really is there: the refusal below is about authorization, not
    # about a missing object.
    assert get_media_store().read(sibling_key)

    assert client.get(_link(sibling_key, expires, signature)).status_code == 403
    # And the claimed page still opens, so nothing about the request shape broke.
    assert client.get(page["page_image_url"]).status_code == 200


def test_the_signature_does_not_open_the_prefix_or_a_directory(
    client: TestClient, owner_user, owner_headers, owner_project
) -> None:
    page = _claimed_page(client, owner_headers, owner_project)
    claimed_key, expires, signature = _link_parts(page["page_image_url"])

    for reach in (
        "parts/",
        "parts",
        claimed_key.rsplit("/", 1)[0] + "/",
        "",
        # Percent-encoded so the client cannot normalize the traversal away
        # before the platform ever sees it.
        "%2e%2e/%2e%2e/secrets.env",
    ):
        assert client.get(_link(reach, expires, signature)).status_code == 403, reach


# ---------------------------------------------------------------------------
# The link's lifetime is not the lease's
# ---------------------------------------------------------------------------


def test_the_link_expires_long_before_the_lease_does(
    client: TestClient, owner_user, owner_headers, owner_project
) -> None:
    page = _claimed_page(client, owner_headers, owner_project)

    image_window = _seconds_until(page["page_image_expires_at"], of=page["_server_time"])
    lease_window = _seconds_until(page["lease_expires_at"], of=page["_server_time"])

    assert 0 < image_window < lease_window
    # The configured values, not a measured elapsed time: 60s against the 600s
    # lease. The tolerance absorbs the round trip, nothing else.
    assert 55 <= image_window <= 65
    assert 590 <= lease_window <= 610


# ---------------------------------------------------------------------------
# The platform never becomes the thing that streams scans in production
# ---------------------------------------------------------------------------


def test_the_platform_refuses_to_serve_objects_on_the_supabase_backend(
    client: TestClient, owner_user, owner_headers, owner_project, monkeypatch
) -> None:
    """On the production storage profile, Storage checks its own signature and
    hands the bytes over directly. A serverless function streaming a manuscript
    scan is the cost ADR 0002 rejected, so this route stops answering."""
    page = _claimed_page(client, owner_headers, owner_project)
    assert client.get(page["page_image_url"]).status_code == 200

    monkeypatch.setenv("STORAGE_BACKEND", "supabase")
    reset_settings_caches()

    assert client.get(page["page_image_url"]).status_code == 404
