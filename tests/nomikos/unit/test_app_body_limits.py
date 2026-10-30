"""The global body-size cap answers 413 before handlers buffer the body."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from backend.core.api.body_limit import (
    JSON_BODY_LIMIT_BYTES,
    MULTIPART_BODY_LIMIT_BYTES,
    MULTIPART_FILE_LIMIT_BYTES,
    BodySizeLimitMiddleware,
    _content_length,
    _limit_for_scope,
)
from backend.core.settings import reset_settings_caches


def _app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(BodySizeLimitMiddleware)

    @app.post("/echo")
    async def echo(request: Request) -> JSONResponse:
        return JSONResponse({"bytes": len(await request.body())})

    return app


def test_small_json_passes() -> None:
    response = TestClient(_app()).post("/echo", json={"hello": "world"})

    assert response.status_code == 200


def test_large_json_is_rejected_before_the_handler() -> None:
    response = TestClient(_app()).post("/echo", json={"pad": "x" * (JSON_BODY_LIMIT_BYTES)})

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "REQUEST_TOO_LARGE"


async def test_chunked_body_without_content_length_is_counted() -> None:
    from starlette.responses import JSONResponse

    async def _ok_app(scope, receive, send) -> None:
        body = b""
        while True:
            message = await receive()
            assert message["type"] == "http.request"
            body += message.get("body", b"")
            if not message.get("more_body"):
                break
        await JSONResponse({"bytes": len(body)})(scope, receive, send)

    half = JSON_BODY_LIMIT_BYTES // 2
    messages = [
        {"type": "http.request", "body": b"x" * half, "more_body": True},
        {"type": "http.request", "body": b"y" * (half + 1), "more_body": False},
    ]
    sent: list = []

    async def _receive():
        return messages.pop(0)

    async def _send(message) -> None:
        sent.append(message)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/echo",
        "headers": [],
        "query_string": b"",
        "server": ("testserver", 80),
        "scheme": "http",
        "client": ("testclient", 50000),
    }
    await BodySizeLimitMiddleware(_ok_app)(scope, _receive, _send)

    assert sent[0]["status"] == 413


def test_multipart_gets_the_upload_budget() -> None:
    scope = {
        "type": "http",
        "headers": [(b"content-type", b"multipart/form-data; boundary=abc")],
    }
    assert _limit_for_scope(scope) == MULTIPART_BODY_LIMIT_BYTES
    assert _limit_for_scope({"type": "http", "headers": []}) == JSON_BODY_LIMIT_BYTES
    assert _content_length({"type": "http", "headers": [(b"content-length", b"12")]}) == 12
    assert _content_length({"type": "http", "headers": [(b"content-length", b"nope")]}) is None


def _multipart_upload_length(file_bytes: int) -> int:
    """The Content-Length of a real part upload carrying a file of ``file_bytes``."""
    import httpx

    filename = "Codex Vaticanus graecus 1209, folio 1r (full resolution).png"
    request = httpx.Request(
        "POST", "http://testserver/echo", files={"file": (filename, b"x", "image/png")}
    )
    return len(request.read()) - 1 + file_bytes


async def _status_for_declared_multipart(length: int) -> int:
    async def _never_reads(scope, receive, send) -> None:
        await JSONResponse({"ok": True})(scope, receive, send)

    sent: list = []

    async def _send(message) -> None:
        sent.append(message)

    async def _receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/echo",
        "headers": [
            (b"content-type", b"multipart/form-data; boundary=abc"),
            (b"content-length", str(length).encode()),
        ],
        "query_string": b"",
        "server": ("testserver", 80),
        "scheme": "http",
        "client": ("testclient", 50000),
    }
    await BodySizeLimitMiddleware(_never_reads)(scope, _receive, _send)
    return sent[0]["status"]


async def test_a_file_at_the_upload_cap_fits_with_its_envelope() -> None:
    """A 100 MB page image, wrapped in its multipart envelope, reaches the route."""
    at_cap = _multipart_upload_length(MULTIPART_FILE_LIMIT_BYTES)

    assert at_cap > MULTIPART_FILE_LIMIT_BYTES
    assert await _status_for_declared_multipart(at_cap) == 200
    # The envelope allowance stays bounded: far past it is still refused.
    assert await _status_for_declared_multipart(MULTIPART_BODY_LIMIT_BYTES + 1) == 413


def test_small_multipart_passes() -> None:
    response = TestClient(_app()).post("/echo", files={"file": ("a.bin", b"123")})

    assert response.status_code == 200


def test_real_app_rejects_a_20mb_json_body(monkeypatch) -> None:
    """Over the 16 MiB middleware budget: the edge answers, not the route."""
    from backend.core.app import create_app

    monkeypatch.setenv("JWT_SECRET", "xQ7v2Kd9RmZ4pB6wLt1yHs3nCf8jUa5eG0oV")
    reset_settings_caches()

    response = TestClient(create_app()).post(
        "/auth/register",
        json={"email": "a@example.org", "username": "u", "password": "x" * (20 * 1024 * 1024)},
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "REQUEST_TOO_LARGE"


def test_cors_exposes_the_error_id(monkeypatch) -> None:
    """The frontend reads X-Error-ID across origins; CORS must expose it."""
    from fastapi.middleware.cors import CORSMiddleware

    from backend.core.app import create_app

    monkeypatch.setenv("JWT_SECRET", "xQ7v2Kd9RmZ4pB6wLt1yHs3nCf8jUa5eG0oV")
    reset_settings_caches()

    options = [m.kwargs for m in create_app().user_middleware if m.cls is CORSMiddleware]

    assert options == [
        {
            "allow_origins": options[0]["allow_origins"],
            "allow_credentials": True,
            "allow_methods": ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
            "allow_headers": ["Authorization", "Content-Type", "X-CSRF-Token"],
            "expose_headers": ["X-Error-ID"],
        }
    ]


def test_dense_page_save_passes_the_json_budget() -> None:
    """300 lines with two full 256-point polygons each is ~2.5-5 MB: admitted."""
    import json

    polygon = [[1234.567, 8901.234]] * 256
    lines = [
        {"order": i, "points": polygon, "kraken_ceiling": polygon, "text": "x" * 200}
        for i in range(300)
    ]
    body = json.dumps({"lines": lines}).encode()

    assert len(body) < JSON_BODY_LIMIT_BYTES
    response = TestClient(_app()).post(
        "/echo", content=body, headers={"content-type": "application/json"}
    )

    assert response.status_code == 200
    assert response.json() == {"bytes": len(body)}
