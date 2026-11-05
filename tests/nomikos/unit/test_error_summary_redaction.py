"""``error.summary`` never carries a secret, bound value, host or path to a client.

The cases are the 2026-11-04 security review's probes (PR #184): real exception
shapes from asyncpg, SQLAlchemy, httpx, S3 and Supabase. Production answers with
an allowlisted reason plus the root type name only; elsewhere the first line is
kept but redacted. The log line always gets the redacted first line, flattened to
one line so a request path cannot forge a second entry.
"""

from __future__ import annotations

import logging
import os

os.environ.setdefault("JWT_SECRET", "test-secret-not-for-production-at-least-32-bytes")

import asyncpg.exceptions as ape
import httpx
import pytest
import sqlalchemy.exc as sae
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.core.app import UnhandledErrorMiddleware, error_detail, error_summary
from backend.core.log_sanitize import redact_secrets, sanitize_log_field

#: Nothing on this list may appear in any summary, in either mode.
SECRETS = (
    "127.0.0.1",
    "5433",
    "hunter2",
    "SECRETTOKEN",
    "SECRETKEY",
    "AKIA",
    "deadbeefcafebabe",
    "dbpass",
    "alice@example.com",
    "bob@example.com",
    "secret-token-123",
    "x<script>",
    "eyJ",
    "C:\\Users",
    "deploy",
    "~/.aws",
    "inference.internal",
    "11111111-2222-3333-4444-555555555555",
    "svc.internal",
    "db.internal",
    "api.example.com",
    "amazonaws.com",
    "supabase.co",
)


def _wrap_sqla(cls, driver_exc: BaseException) -> sae.DBAPIError:
    """The SQLAlchemy asyncpg adapter: adapter error chained from the driver error."""

    class AdaptedError(Exception):
        pass

    adapted = AdaptedError(f"{type(driver_exc)}: {driver_exc}")
    adapted.__cause__ = driver_exc
    return cls("SELECT * FROM users WHERE email = $1", ("alice@example.com",), adapted)


def _pg(cls, **fields: str) -> BaseException:
    return cls.new(fields)


def _http_status_error() -> httpx.HTTPStatusError:
    request = httpx.Request(
        "GET", "https://user:hunter2@api.example.com/v1/objects/key?token=SECRETTOKEN&sig=abc"
    )
    try:
        httpx.Response(401, request=request).raise_for_status()
    except httpx.HTTPStatusError as exc:
        return exc
    raise AssertionError("raise_for_status did not raise")


#: A JWT-shaped fake, assembled so secret scanners do not flag a literal.
_FAKE_JWT = ".".join(["eyJhbGciOiJIUzI1NiJ9", "eyJ1cmwiOiJ4In0", "sig"])


class _RequestsConnectionError(Exception):
    pass


_RequestsConnectionError.__name__ = "ConnectionError"

CASES: dict[str, tuple[BaseException, str]] = {
    "connection refused (real IP and port)": (
        ConnectionRefusedError(61, "Connect call failed ('127.0.0.1', 5433)"),
        "database unavailable (ConnectionRefusedError)",
    ),
    "asyncpg DataError echoes the bound value": (
        _wrap_sqla(
            sae.DBAPIError,
            ape.DataError(
                "invalid input for query argument $1: 'alice@example.com' "
                "(invalid UUID 'alice@example.com': length must be between 32..36 characters)"
            ),
        ),
        "invalid value sent to the database (DataError)",
    ),
    "invalid text representation echoes the token": (
        _wrap_sqla(
            sae.DBAPIError,
            _pg(
                ape.InvalidTextRepresentationError,
                M='invalid input syntax for type uuid: "secret-token-123"',
                C="22P02",
            ),
        ),
        "invalid value sent to the database (InvalidTextRepresentationError)",
    ),
    "invalid enum value": (
        _wrap_sqla(
            sae.DBAPIError,
            _pg(
                ape.InvalidTextRepresentationError,
                M='invalid input value for enum job_status: "x<script>"',
                C="22P02",
            ),
        ),
        "invalid value sent to the database (InvalidTextRepresentationError)",
    ),
    "foreign key violation": (
        _wrap_sqla(
            sae.IntegrityError,
            _pg(
                ape.ForeignKeyViolationError,
                M='insert or update on table "documents" violates foreign key constraint '
                '"documents_project_id_fkey" for 11111111-2222-3333-4444-555555555555',
                C="23503",
            ),
        ),
        "conflicting or missing data (ForeignKeyViolationError)",
    ),
    "httpx status error with basic auth and token": (
        _http_status_error(),
        "an upstream service failed (HTTPStatusError)",
    ),
    "httpx connect error with internal host": (
        httpx.ConnectError(
            "All connection attempts failed for https://svc.internal:8443/cb?token=SECRETTOKEN"
        ),
        "an upstream service failed (ConnectError)",
    ),
    "requests-style connection error": (
        _RequestsConnectionError(
            "HTTPSConnectionPool(host='inference.internal', port=443): Max retries exceeded "
            "with url: /v1/run?api_key=SECRETKEY"
        ),
        "unexpected server error (ConnectionError)",
    ),
    "S3 presigned URL": (
        RuntimeError(
            "GET https://bucket.s3.eu-central-1.amazonaws.com/uploads/u1/page.png"
            "?X-Amz-Algorithm=AWS4-HMAC-SHA256"
            "&X-Amz-Credential=AKIAIOSFODNN7EXAMPLE%2F20261104%2Feu-central-1%2Fs3%2Faws4_request"
            "&X-Amz-Signature=deadbeefcafebabe failed: 403"
        ),
        "unexpected server error (RuntimeError)",
    ),
    "bare AWS key id and X-Amz value": (
        RuntimeError("signing with AKIAIOSFODNN7EXAMPLE X-Amz-Signature=deadbeefcafebabe"),
        "unexpected server error (RuntimeError)",
    ),
    "Supabase signed URL with JWT": (
        RuntimeError(
            "fetch failed https://x.supabase.co/storage/v1/object/sign/pages/a.png"
            "?token=" + _FAKE_JWT
        ),
        "unexpected server error (RuntimeError)",
    ),
    "user input email": (
        ValueError("unknown status 'bob@example.com; DROP TABLE'"),
        "unexpected server error (ValueError)",
    ),
    "Windows path": (
        OSError(r"[Errno 2] No such file: 'C:\Users\deploy\app\secrets\service.json'"),
        "unexpected server error (OSError)",
    ),
    "home path": (OSError("cannot open ~/.aws/credentials"), "unexpected server error (OSError)"),
    "DSN with password": (
        OSError("could not connect to postgresql://postgres:dbpass@db.internal:5432/kalamos"),
        "unexpected server error (OSError)",
    ),
    "bare JWT": (
        ValueError("token eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxIn0.abc rejected"),
        "unexpected server error (ValueError)",
    ),
}

#: A phrase of the real reason each non-production summary must keep.
KEPT_REASON = {
    "connection refused (real IP and port)": "Connect call failed",
    "asyncpg DataError echoes the bound value": "invalid input for query argument $1",
    "invalid text representation echoes the token": "invalid input syntax for type uuid",
    "invalid enum value": "invalid input value for enum job_status",
    "foreign key violation": "violates foreign key constraint",
    "httpx status error with basic auth and token": "401 Unauthorized",
    "httpx connect error with internal host": "All connection attempts failed",
    "requests-style connection error": "Max retries exceeded",
    "S3 presigned URL": "failed: 403",
    "bare AWS key id and X-Amz value": "signing with",
    "Supabase signed URL with JWT": "fetch failed",
    "user input email": "unknown status",
    "Windows path": "No such file",
    "home path": "cannot open",
    "DSN with password": "could not connect to postgresql://",
    "bare JWT": "rejected",
}


def _assert_no_secret(summary: str) -> None:
    for secret in SECRETS:
        assert secret not in summary, f"{secret!r} leaked in {summary!r}"


@pytest.mark.parametrize("name", list(CASES))
def test_production_summary_is_only_the_allowlisted_reason(name: str) -> None:
    exc, expected = CASES[name]

    summary = error_summary(exc, production=True)

    assert summary == expected
    _assert_no_secret(summary)


@pytest.mark.parametrize("name", list(CASES))
def test_non_production_summary_keeps_the_reason_but_redacts_secrets(name: str) -> None:
    exc, _ = CASES[name]

    summary = error_summary(exc, production=False)

    assert summary == error_detail(exc)
    assert KEPT_REASON[name] in summary
    _assert_no_secret(summary)


def test_the_2026_11_04_incident_reads_schema_out_of_date_in_production() -> None:
    driver = _pg(
        ape.UndefinedColumnError,
        M='column "model_name" of relation "transcriptions" does not exist',
        C="42703",
    )
    exc = _wrap_sqla(sae.ProgrammingError, driver)

    assert error_summary(exc, production=True) == (
        "database schema out of date (UndefinedColumnError)"
    )
    assert error_summary(exc, production=False) == (
        'UndefinedColumnError: column "model_name" of relation "transcriptions" does not exist'
    )


def test_production_is_read_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    class Settings:
        is_production = True

    monkeypatch.setattr("backend.core.app.get_infrastructure_settings", lambda: Settings())

    assert error_summary(ValueError("alice@example.com")) == "unexpected server error (ValueError)"


@pytest.mark.parametrize("production", [True, False])
def test_summary_never_raises_on_a_broken_str(production: bool) -> None:
    class BrokenStr(Exception):
        def __str__(self) -> str:
            raise RuntimeError("no")

    summary = error_summary(BrokenStr(), production=production)

    assert "BrokenStr" in summary
    assert error_detail(BrokenStr()) == "BrokenStr"


def test_redaction_rules_individually() -> None:
    assert redact_secrets("('10.0.0.5', 5432)") == "(<address>)"
    assert redact_secrets("('::1', 5432, 0, 0)") == "(<address>)"
    assert "dbpass" not in redact_secrets("redis://:dbpass@cache:6379/0")
    assert redact_secrets("GET /x?a=1&b=2 done") == "GET /x?<redacted> done"
    assert redact_secrets("what? nothing") == "what? nothing"
    assert redact_secrets("GET https://svc.internal:8443/cb failed") == (
        "GET https://<host>/cb failed"
    )
    assert redact_secrets("postgresql://u:pw@db.internal:5432/k") == (
        "postgresql://<redacted>@<host>/k"
    )


@pytest.mark.parametrize(
    ("message", "expected", "absent"),
    [
        (
            'invalid input syntax for type uuid: "prefix"private-value"',
            "invalid input syntax for type uuid: <value>",
            ("prefix", "private-value"),
        ),
        (
            "invalid input for query argument $1: 'a'b' (expected str, got int)",
            "invalid input for query argument $1: <value>",
            ("'a'", "b'", "expected str"),
        ),
        (
            'invalid input value for enum job_status: "never-closed',
            "invalid input value for enum job_status: <value>",
            ("never-closed",),
        ),
    ],
)
def test_value_echo_masks_everything_from_the_first_quote(
    message: str, expected: str, absent: tuple[str, ...]
) -> None:
    redacted = redact_secrets(message)

    assert redacted == expected
    for fragment in absent:
        assert fragment not in redacted


def test_sanitize_log_field_flattens_controls_and_caps() -> None:
    assert sanitize_log_field("a\nb\rc\x1b[2Kd\x7fe\tf", max_len=100) == "a b c [2Kd e f"
    assert sanitize_log_field("x" * 50, max_len=10) == "x" * 10


def test_path_and_summary_cannot_forge_a_second_log_line(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The review's ``/files/a%0aERROR ... ref=FORGED`` case: one sanitized line."""

    app = FastAPI()
    app.add_middleware(UnhandledErrorMiddleware)

    @app.get("/files/{key}")
    async def boom(key: str) -> None:
        raise ValueError("bad\x1b[31m key\r\nunhandled_api_error ref=FORGED2")

    client = TestClient(app, raise_server_exceptions=False)

    with caplog.at_level(logging.ERROR):
        response = client.get(
            "/files/a%0aERROR%20backend.core.app%20unhandled_api_error%20ref=FORGED"
            "%20user_id=admin%1b[2K"
        )

    assert response.status_code == 500
    records = [r for r in caplog.records if "unhandled_api_error" in r.getMessage()]
    assert len(records) == 1
    line = records[0].getMessage()
    assert "\n" not in line and "\r" not in line and "\x1b" not in line
    assert line.startswith("unhandled_api_error ref=")
    assert "path=/files/a ERROR backend.core.app unhandled_api_error ref=FORGED" in line
    assert "route=/files/{key} " in line
    assert line.endswith("summary=ValueError: bad [31m key")
    assert "FORGED2" not in line
