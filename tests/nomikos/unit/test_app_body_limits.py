"""The global body-size cap answers 413 before handlers buffer the body."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from backend.core.api.body_limit import (
    JSON_BODY_LIMIT_BYTES,
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
    assert _limit_for_scope(scope) == 100 * 1024 * 1024
    assert _limit_for_scope({"type": "http", "headers": []}) == JSON_BODY_LIMIT_BYTES
    assert _content_length({"type": "http", "headers": [(b"content-length", b"12")]}) == 12
    assert _content_length({"type": "http", "headers": [(b"content-length", b"nope")]}) is None


def test_small_multipart_passes() -> None:
    response = TestClient(_app()).post("/echo", files={"file": ("a.bin", b"123")})

    assert response.status_code == 200


def test_real_app_rejects_a_5mb_json_body(monkeypatch) -> None:
    from backend.core.app import create_app

    monkeypatch.setenv("JWT_SECRET", "xQ7v2Kd9RmZ4pB6wLt1yHs3nCf8jUa5eG0oV")
    reset_settings_caches()

    response = TestClient(create_app()).post(
        "/auth/register",
        json={"email": "a@example.org", "username": "u", "password": "x" * (5 * 1024 * 1024)},
    )

    assert response.status_code == 413
