"""Unhandled route exceptions return the JSON 500 with CORS headers.

Without them the browser turns the 500 into a network error and the frontend
only sees a generic failure (production incident 2026-11-04).
"""

from __future__ import annotations

import logging
import os

os.environ.setdefault("JWT_SECRET", "test-secret-not-for-production-at-least-32-bytes")

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.core.app import create_app
from backend.core.exceptions import ConflictError
from backend.core.settings import get_app_settings

DISALLOWED_ORIGIN = "https://attacker.example"


@pytest.fixture
def app() -> FastAPI:
    app = create_app()

    @app.get("/__test__/boom")
    async def boom() -> None:
        raise RuntimeError("database exploded")

    @app.get("/__test__/conflict")
    async def conflict() -> None:
        raise ConflictError("already exists")

    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def origin() -> str:
    return get_app_settings().cors_origin_list[0]


def test_unhandled_500_carries_cors_headers_and_envelope(
    client: TestClient, origin: str, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.ERROR):
        response = client.get("/__test__/boom", headers={"Origin": origin})

    assert response.status_code == 500
    assert response.headers["access-control-allow-origin"] == origin
    assert response.headers["access-control-allow-credentials"] == "true"
    assert "x-error-id" in response.headers["access-control-expose-headers"].lower()
    assert response.headers["x-content-type-options"] == "nosniff"
    error = response.json()["error"]
    assert error["code"] == "INTERNAL_SERVER_ERROR"
    assert error["message"] == "Internal server error"
    assert error["ref"] == response.headers["x-error-id"]

    records = [r for r in caplog.records if "unhandled_api_error" in r.getMessage()]
    assert len(records) == 1
    assert error["ref"] in records[0].getMessage()
    assert records[0].exc_info is not None


def test_unhandled_500_gives_no_cors_to_disallowed_origin(client: TestClient) -> None:
    response = client.get("/__test__/boom", headers={"Origin": DISALLOWED_ORIGIN})

    assert response.status_code == 500
    assert "access-control-allow-origin" not in response.headers
    assert response.json()["error"]["code"] == "INTERNAL_SERVER_ERROR"


def test_conflict_path_is_unchanged(client: TestClient, origin: str) -> None:
    response = client.get("/__test__/conflict", headers={"Origin": origin})

    assert response.status_code == 409
    assert response.headers["access-control-allow-origin"] == origin
    error = response.json()["error"]
    assert error["code"] == "CONFLICT"
    assert error["ref"] == response.headers["x-error-id"]
