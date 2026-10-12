"""Process-local request counters and the recent client-failure buffer.

Everything here lives in this process's memory and is gone on restart, on purpose.
The database is metered, and what the admin dashboard is meant to help reduce is
work per request; writing a row per request to measure that would add exactly the
load it is trying to find. The cost is that the numbers cover one API process since
it started (``started_at`` says when) and are not summed across replicas.

Two things are kept:

* per ``(method, route template, status class)`` a request count and latency totals,
  filled by :class:`RequestMetricsMiddleware`. The route *template*
  (``/documents/{document_id}``), never the raw path, so the table stays the size of
  the API rather than the size of the URL space.
* the last :data:`CLIENT_FAILURE_BUFFER` frontend failure beacons plus a per-day count,
  filled by ``POST /client-failures``. That route only ever logged, so there is no
  table to read them back from.

Read through :func:`snapshot` and :func:`client_failures_snapshot`, which the admin
router exposes; nothing else reads them.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, date, datetime

from starlette.types import ASGIApp, Message, Receive, Scope, Send

#: Paths polled by infrastructure or fetched by tooling; counting them would bury
#: the traffic people generate under probe noise.
SKIPPED_PATHS = frozenset({"/health", "/", "/docs", "/redoc", "/openapi.json", "/favicon.ico"})
UNMATCHED_ROUTE = "<unmatched>"
CLIENT_FAILURE_BUFFER = 200
#: Per-day client failure counts older than this are dropped, matching the widest
#: window the dashboard offers.
CLIENT_FAILURE_DAYS_KEPT = 90


@dataclass
class _RouteCounter:
    count: int = 0
    total_ms: float = 0.0
    max_ms: float = 0.0


@dataclass
class _ClientFailure:
    at: datetime
    path: str | None
    status: int | None
    source: str | None
    message: str


@dataclass
class _State:
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    routes: dict[tuple[str, str, str], _RouteCounter] = field(default_factory=dict)
    failures: deque[_ClientFailure] = field(
        default_factory=lambda: deque(maxlen=CLIENT_FAILURE_BUFFER)
    )
    failures_per_day: dict[date, int] = field(default_factory=dict)


_state = _State()


def reset() -> None:
    """Forget everything; for tests."""
    global _state
    _state = _State()


def _status_class(status_code: int) -> str:
    return f"{status_code // 100}xx"


def record_request(method: str, route: str, status_code: int, elapsed_ms: float) -> None:
    key = (method, route, _status_class(status_code))
    counter = _state.routes.get(key)
    if counter is None:
        counter = _state.routes[key] = _RouteCounter()
    counter.count += 1
    counter.total_ms += elapsed_ms
    counter.max_ms = max(counter.max_ms, elapsed_ms)


def record_client_failure(
    *, path: str | None, status: int | None, source: str | None, message: str
) -> None:
    now = datetime.now(UTC)
    _state.failures.append(
        _ClientFailure(at=now, path=path, status=status, source=source, message=message)
    )
    today = now.date()
    _state.failures_per_day[today] = _state.failures_per_day.get(today, 0) + 1
    if len(_state.failures_per_day) > CLIENT_FAILURE_DAYS_KEPT:
        del _state.failures_per_day[min(_state.failures_per_day)]


def snapshot() -> dict:
    """Per-route counters, busiest first."""
    routes = [
        {
            "method": method,
            "route": route,
            "status_class": status_class,
            "count": counter.count,
            "avg_ms": round(counter.total_ms / counter.count, 1) if counter.count else 0.0,
            "max_ms": round(counter.max_ms, 1),
        }
        for (method, route, status_class), counter in _state.routes.items()
    ]
    routes.sort(key=lambda row: (-row["count"], row["route"], row["method"]))
    return {
        "started_at": _state.started_at,
        "total_requests": sum(row["count"] for row in routes),
        "routes": routes,
    }


def client_failures_snapshot(*, days: int, recent: int = 20) -> dict:
    """Per-day counts inside the window and the newest ``recent`` beacons."""
    cutoff = datetime.now(UTC).date().toordinal() - days + 1
    per_day = [
        {"day": day, "count": count}
        for day, count in sorted(_state.failures_per_day.items())
        if day.toordinal() >= cutoff
    ]
    latest = list(_state.failures)[-recent:][::-1]
    return {
        "started_at": _state.started_at,
        "per_day": per_day,
        "recent": [
            {
                "at": failure.at,
                "path": failure.path,
                "status": failure.status,
                "source": failure.source,
                "message": failure.message,
            }
            for failure in latest
        ],
    }


class RequestMetricsMiddleware:
    """Pure ASGI middleware: count each HTTP request under its route template.

    Pure ASGI rather than ``BaseHTTPMiddleware`` so streaming responses (job
    events over SSE) pass through untouched; latency is measured to the end of
    the response body, so background tasks that run after it are not charged.
    The route is read from ``scope["route"]`` after the router has run, which
    Starlette fills in for the matched route.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("path") in SKIPPED_PATHS:
            await self.app(scope, receive, send)
            return

        status_code = 500
        started = time.perf_counter()
        recorded = False

        def record() -> None:
            nonlocal recorded
            if recorded:
                return
            recorded = True
            route = scope.get("route")
            template = getattr(route, "path", None) or UNMATCHED_ROUTE
            record_request(
                scope.get("method", "GET"),
                template,
                status_code,
                (time.perf_counter() - started) * 1000.0,
            )

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)
            # Stop the clock once the last body chunk is out: background tasks
            # (the Resend mail after login) run after it inside the same call.
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                record()

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            record()
