"""End-to-end device pairing over real HTTP against Postgres.

These tests run against the application the deployment builds, through the
session-scoped ``client`` fixture. They used to assemble a local app instead,
because the device routers construct their service at import time and the poll
cadence had to be collapsed first; that env var now lives in the integration
conftest, which is imported earlier still, so the local app is unnecessary.

Removing it fixed four tests. A second ``TestClient`` runs its own event loop,
and the asyncpg pool is bound to whichever loop created it, so every query from
the second client failed with "attached to a different loop" (issue #63).

It also removes a hazard worth naming: an app assembled by the test module is
never evidence that anything is reachable in the deployed application. That is
how this phase once shipped with its routers unmounted and a green suite.
``test_device_routes_are_mounted_on_the_real_app`` in the unit suite holds that
line over the full route set; every test here reaches those routes on the app
the deployment builds, so the module carries it too.

Requires migration ``003_helper_devices``.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from backend.ml.application.device_auth import DEVICE_TOKEN_HEADER
from infrastructure.db import sync_engine
from tests.nomikos.integration.helpers import pair_device_over_http

pytestmark = pytest.mark.integration


def _start_pairing(client: TestClient, name: str = "Researcher laptop") -> tuple[dict, str]:
    response = client.post(
        "/device/v1/pairings",
        json={
            "device_name": name,
            "platform": "darwin-arm64",
            "helper_version": "0.2.0",
            "capabilities": {"runtime": "torch"},
        },
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert "?" not in body["verification_url"]
    # The helper must be able to show this next to the consent screen's copy.
    assert body["confirmation_code"]
    verification_token = body["verification_url"].split("#", 1)[1]
    return body, verification_token


def _approve(client: TestClient, headers: dict, pairing_id: str, verification_token: str) -> dict:
    response = client.post(
        f"/devices/pairings/{pairing_id}/approve",
        headers=headers,
        json={"verification_token": verification_token},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _collect(client: TestClient, pairing_id: str, device_code: str) -> dict:
    response = client.post(
        "/device/v1/pairings/token",
        json={"pairing_id": pairing_id, "device_code": device_code},
    )
    # Protocol states always ride in a 200 body: the error envelope replaces
    # HTTPException.detail with a fixed public string.
    assert response.status_code == 200, response.text
    return response.json()


def test_mint_redeem_then_authenticate(client: TestClient, owner_headers: dict[str, str]) -> None:
    headers = owner_headers
    started, verification_token = _start_pairing(client)

    pending = _collect(client, started["pairing_id"], started["device_code"])
    assert pending["status"] == "authorization_pending"
    assert pending["device_token"] is None

    lookup = client.post(
        "/devices/pairings/lookup",
        headers=headers,
        json={"verification_token": verification_token},
    )
    assert lookup.status_code == 200, lookup.text
    assert lookup.json()["device_name"] == "Researcher laptop"
    assert lookup.json()["pairing_id"] == started["pairing_id"]
    # The consent screen shows the code the helper showed, and no IP-derived
    # signal at all - behind an unallowlisted proxy those were the same value
    # for every user on the platform.
    assert lookup.json()["confirmation_code"] == started["confirmation_code"]
    assert "same_network" not in lookup.json()
    assert "request_ip" not in lookup.json()

    device = _approve(client, headers, started["pairing_id"], verification_token)
    assert device["status"] == "pairing"
    assert device["token_prefix"] == ""

    approved = _collect(client, started["pairing_id"], started["device_code"])
    assert approved["status"] == "approved"
    token = approved["device_token"]
    assert token.startswith("nmd1.")

    me = client.get("/device/v1/self", headers={DEVICE_TOKEN_HEADER: token})
    assert me.status_code == 200, me.text
    assert me.json()["device_id"] == approved["device_id"]

    listed = client.get("/devices", headers=headers)
    assert listed.status_code == 200
    entries = listed.json()
    assert len(entries) == 1
    assert entries[0]["status"] == "online"
    assert entries[0]["token_prefix"].startswith("nmd1.")


def test_device_token_cannot_reach_another_users_devices(
    client: TestClient, owner_headers: dict[str, str], outsider_headers: dict[str, str]
) -> None:
    approved = pair_device_over_http(client, owner_headers, name="Owner laptop")
    token = approved["device_token"]
    device_id = approved["device_id"]

    # The outsider sees nothing and cannot revoke it.
    assert client.get("/devices", headers=outsider_headers).json() == []
    denied = client.delete(f"/devices/{device_id}", headers=outsider_headers)
    assert denied.status_code == 403, denied.text

    # A device token is not a Bearer credential and cannot impersonate a user.
    as_bearer = client.get("/devices", headers={"Authorization": f"Bearer {token}"})
    assert as_bearer.status_code == 401

    # A user's access token is not a device credential either.
    owner_access_token = owner_headers["Authorization"].split(" ", 1)[1]
    as_device = client.get("/device/v1/self", headers={DEVICE_TOKEN_HEADER: owner_access_token})
    assert as_device.status_code == 401

    # The owner still has it.
    assert [entry["id"] for entry in client.get("/devices", headers=owner_headers).json()] == [
        device_id
    ]


def test_expired_pairing_cannot_be_approved(
    client: TestClient, owner_headers: dict[str, str]
) -> None:
    headers = owner_headers
    started, verification_token = _start_pairing(client)

    with sync_engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE helper_pairings SET expires_at = now() - interval '1 minute' "
                "WHERE id = :pairing_id"
            ),
            {"pairing_id": started["pairing_id"]},
        )

    expired = _collect(client, started["pairing_id"], started["device_code"])
    assert expired["status"] == "expired"
    assert expired["device_token"] is None

    # And the browser can no longer see it at all.
    lookup = client.post(
        "/devices/pairings/lookup",
        headers=headers,
        json={"verification_token": verification_token},
    )
    assert lookup.status_code == 404

    approve = client.post(
        f"/devices/pairings/{started['pairing_id']}/approve",
        headers=headers,
        json={"verification_token": verification_token},
    )
    assert approve.status_code == 409


def test_pairing_routes_require_authentication(client: TestClient) -> None:
    started, verification_token = _start_pairing(client)
    assert client.get("/devices").status_code == 401
    assert (
        client.post(
            "/devices/pairings/lookup", json={"verification_token": verification_token}
        ).status_code
        == 401
    )
    assert (
        client.post(
            f"/devices/pairings/{started['pairing_id']}/approve",
            json={"verification_token": verification_token},
        ).status_code
        == 401
    )
    assert client.get("/device/v1/self").status_code == 401
    assert client.post("/device/v1/token/renew").status_code == 401


def test_raw_secrets_are_absent_from_the_database(
    client: TestClient, owner_headers: dict[str, str]
) -> None:
    headers = owner_headers
    approved = pair_device_over_http(client, headers)
    started = approved["pairing"]
    verification_token = approved["verification_token"]
    secret = approved["device_token"].split(".", 2)[2]

    with sync_engine.begin() as connection:
        device_rows = [
            dict(row)
            for row in connection.execute(
                text("SELECT * FROM helper_devices WHERE id = :device_id"),
                {"device_id": approved["device_id"]},
            ).mappings()
        ]
        pairing_rows = [
            dict(row)
            for row in connection.execute(
                text("SELECT * FROM helper_pairings WHERE id = :pairing_id"),
                {"pairing_id": started["pairing_id"]},
            ).mappings()
        ]

    assert device_rows and pairing_rows
    for row in device_rows + pairing_rows:
        for column, value in row.items():
            rendered = str(value)
            assert secret not in rendered, f"raw device secret found in {column}"
            assert started["device_code"] not in rendered, f"raw device_code found in {column}"
            assert verification_token not in rendered, f"raw verification token found in {column}"

    # And no read endpoint hands any of it back.
    listed = client.get("/devices", headers=headers).text
    for leaked in (secret, started["device_code"], verification_token):
        assert leaked not in listed


def test_finished_pairings_are_deleted(client: TestClient, owner_headers: dict[str, str]) -> None:
    """Unauthenticated writes with no cleanup grow forever. Also proves the grant."""
    headers = owner_headers
    approved = pair_device_over_http(client, headers, name="Sweepable laptop")
    consumed_id = approved["pairing"]["pairing_id"]

    with sync_engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE helper_pairings SET created_at = now() - interval '90 days', "
                "expires_at = now() - interval '90 days' WHERE id = :pairing_id"
            ),
            {"pairing_id": consumed_id},
        )

    # The sweep runs from the one endpoint that inserts into this table.
    _start_pairing(client, "Fresh laptop")

    with sync_engine.begin() as connection:
        remaining = connection.execute(
            text("SELECT count(*) FROM helper_pairings WHERE id = :pairing_id"),
            {"pairing_id": consumed_id},
        ).scalar_one()
        devices = connection.execute(
            text("SELECT count(*) FROM helper_devices WHERE id = :device_id"),
            {"device_id": approved["device_id"]},
        ).scalar_one()

    assert remaining == 0
    # Sweeping a pairing must never cascade into the device it created.
    assert devices == 1
