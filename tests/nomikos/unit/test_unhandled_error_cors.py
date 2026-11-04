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
from sqlalchemy.exc import ProgrammingError

from backend.core.app import SUMMARY_MAX_CHARS, create_app, error_summary
from backend.core.exceptions import ConflictError
from backend.core.settings import get_app_settings

DISALLOWED_ORIGIN = "https://attacker.example"


@pytest.fixture
def app() -> FastAPI:
    app = create_app()

    @app.get("/__test__/boom")
    async def boom() -> None:
        raise RuntimeError("database exploded")

    @app.get("/__test__/projects/{project_id}/sql")
    async def sql_failure(project_id: str) -> None:
        raise ProgrammingError(
            "INSERT INTO transcriptions (name) VALUES (%s)",
            {"name": "secret-value"},
            Exception('column "x" does not exist'),
        )

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
    assert error["summary"] == "RuntimeError: database exploded"

    records = [r for r in caplog.records if "unhandled_api_error" in r.getMessage()]
    assert len(records) == 1
    line = records[0].getMessage()
    assert f"ref={error['ref']} " in line
    assert "method=GET path=/__test__/boom route=/__test__/boom user_id=- " in line
    assert "exception=RuntimeError summary=RuntimeError: database exploded" in line
    assert records[0].exc_info is not None


def test_sql_failure_summary_has_the_reason_but_no_statement_or_parameters(
    client: TestClient, origin: str, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.ERROR):
        response = client.get("/__test__/projects/p1/sql", headers={"Origin": origin})

    assert response.status_code == 500
    assert response.headers["access-control-allow-origin"] == origin
    summary = response.json()["error"]["summary"]
    assert summary == 'Exception: column "x" does not exist'
    for leaked in ("INSERT", "secret-value", "Background", "[SQL", "[parameters"):
        assert leaked not in summary

    records = [r for r in caplog.records if "unhandled_api_error" in r.getMessage()]
    assert len(records) == 1
    line = records[0].getMessage()
    assert "route=/__test__/projects/{project_id}/sql " in line
    assert "exception=ProgrammingError " in line
    assert "secret-value" not in line


def test_summary_follows_the_asyncpg_adapter_to_the_driver_error() -> None:
    class UndefinedColumnError(Exception):
        pass

    class AdaptedProgrammingError(Exception):
        pass

    driver = UndefinedColumnError('column "model_name" of relation "transcriptions" does not exist')
    adapted = AdaptedProgrammingError(f"{type(driver)}: {driver}")
    adapted.__cause__ = driver
    exc = ProgrammingError("INSERT ...", {"a": 1}, adapted)

    assert error_summary(exc) == (
        'UndefinedColumnError: column "model_name" of relation "transcriptions" does not exist'
    )


def test_summary_keeps_only_the_first_line_masks_paths_and_caps_length() -> None:
    multiline = RuntimeError("first line\nTraceback (most recent call last): secret")
    assert error_summary(multiline) == "RuntimeError: first line"

    missing = FileNotFoundError("No such file: /srv/app/data/key.pem")
    assert "/srv" not in error_summary(missing)
    assert error_summary(missing).endswith("<path>")

    assert len(error_summary(RuntimeError("x" * 1000))) == SUMMARY_MAX_CHARS
    assert error_summary(RuntimeError()) == "RuntimeError"


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
    assert "summary" not in error
