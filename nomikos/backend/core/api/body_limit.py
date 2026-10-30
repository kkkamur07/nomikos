"""Cap request body sizes before handlers buffer them.

JSON routes accept a body of any size by default, and several of them take
unbounded maps; the multipart upload route legitimately takes up to 100 MB.
The middleware answers 413 from the edge of the app: immediately when
``Content-Length`` already exceeds the budget, and by counting bytes off the
receive channel when it does not (chunked uploads).
"""

from __future__ import annotations

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

#: Ceiling for every non-multipart body. Sized against the biggest legitimate
#: JSON payload: a page/part line save. Each line carries two bounded
#: geometries (``points`` and ``kraken_ceiling``, at most 256 points each)
#: plus up to 10,000 chars of text. A JSON point ``[1234.567, 8901.234]`` is
#: ~20 bytes, budgeted at 32:
#:
#:   300 lines x 2 geometries x 256 points x 32 B ≈ 4.7 MB,
#:   plus typical line text (~300 x 200 B ≈ 60 KB).
#:
#: A dense 300-line editor save is therefore ~2.5-5 MB; 16 MiB admits it with
#: ~3x headroom while still refusing absurd bodies before handlers buffer
#: them. (The 10,000-line theoretical maximum would be hundreds of MB - no
#: legitimate client sends that; the schema bounds validate content, this
#: constant caps transport.)
JSON_BODY_LIMIT_BYTES = 16 * 1024 * 1024
#: The largest file a multipart body may carry. Only the part-upload route
#: speaks multipart, and it enforces this same 100 MB cap on the decoded file
#: (``MAX_PART_UPLOAD_BYTES`` in ``document/application/part_service.py``).
MULTIPART_FILE_LIMIT_BYTES = 100 * 1024 * 1024
#: Room for the multipart envelope around that file: the boundaries and the
#: part headers, filename included. The middleware counts the whole request, so
#: without this a file exactly at the cap was refused before the route saw it.
MULTIPART_ENVELOPE_BYTES = 64 * 1024
#: Ceiling for multipart bodies: the file budget plus its envelope.
MULTIPART_BODY_LIMIT_BYTES = MULTIPART_FILE_LIMIT_BYTES + MULTIPART_ENVELOPE_BYTES


class _BodyTooLarge(Exception):
    pass


def _limit_for_scope(scope: Scope) -> int:
    headers = dict(scope.get("headers", []))
    content_type = headers.get(b"content-type", b"").decode("latin-1").lower()
    if content_type.startswith("multipart/"):
        return MULTIPART_BODY_LIMIT_BYTES
    return JSON_BODY_LIMIT_BYTES


def _content_length(scope: Scope) -> int | None:
    headers = dict(scope.get("headers", []))
    raw = headers.get(b"content-length")
    if raw is None:
        return None
    try:
        return int(raw.decode("ascii"))
    except ValueError:
        return None


def _too_large_response() -> JSONResponse:
    return JSONResponse(
        status_code=413,
        content={"error": {"code": "REQUEST_TOO_LARGE", "message": "Request body too large"}},
    )


class BodySizeLimitMiddleware:
    """Reject bodies over budget with 413 before the handler reads them."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        limit = _limit_for_scope(scope)
        declared = _content_length(scope)
        if declared is not None and declared > limit:
            await _too_large_response()(scope, receive, send)
            return
        received = 0

        async def bounded_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise _BodyTooLarge
            return message

        try:
            await self.app(scope, bounded_receive, send)
        except _BodyTooLarge:
            await _too_large_response()(scope, receive, send)
