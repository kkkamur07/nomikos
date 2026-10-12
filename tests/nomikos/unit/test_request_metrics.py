"""Per-route request counters and the client-failure buffer behind ``/admin``."""

from __future__ import annotations

import time

import pytest
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.testclient import TestClient

from backend.core.api import request_metrics
from backend.core.api.request_metrics import RequestMetricsMiddleware


@pytest.fixture(autouse=True)
def _fresh_counters():
    request_metrics.reset()
    yield
    request_metrics.reset()


def _app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(RequestMetricsMiddleware)

    @app.get("/documents/{document_id}")
    async def read_document(document_id: str) -> dict:
        if document_id == "missing":
            raise HTTPException(status_code=404)
        return {"id": document_id}

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    return app


def _route(snapshot: dict, route: str, status_class: str) -> dict:
    return next(
        row
        for row in snapshot["routes"]
        if row["route"] == route and row["status_class"] == status_class
    )


def test_requests_are_counted_under_the_route_template_not_the_raw_path() -> None:
    client = TestClient(_app())
    client.get("/documents/a")
    client.get("/documents/b")
    client.get("/documents/missing")

    snapshot = request_metrics.snapshot()

    ok = _route(snapshot, "/documents/{document_id}", "2xx")
    assert ok["count"] == 2
    assert ok["method"] == "GET"
    assert ok["max_ms"] >= ok["avg_ms"] >= 0
    assert _route(snapshot, "/documents/{document_id}", "4xx")["count"] == 1
    assert snapshot["total_requests"] == 3
    assert not any(row["route"] == "/documents/a" for row in snapshot["routes"])


def test_probe_paths_are_not_counted() -> None:
    client = TestClient(_app())
    client.get("/health")

    assert request_metrics.snapshot()["total_requests"] == 0


def test_a_path_no_route_matches_shares_one_bucket() -> None:
    client = TestClient(_app())
    client.get("/nope/1")
    client.get("/nope/2")

    unmatched = _route(request_metrics.snapshot(), request_metrics.UNMATCHED_ROUTE, "4xx")
    assert unmatched["count"] == 2


def test_busiest_route_comes_first() -> None:
    request_metrics.record_request("GET", "/a", 200, 1.0)
    for _ in range(3):
        request_metrics.record_request("POST", "/b", 201, 2.0)

    routes = request_metrics.snapshot()["routes"]
    assert [row["route"] for row in routes] == ["/b", "/a"]
    assert routes[0]["avg_ms"] == 2.0


def test_client_failures_keep_the_newest_first_and_count_per_day() -> None:
    for index in range(25):
        request_metrics.record_client_failure(
            path=f"/p/{index}", status=500, source="ui", message=f"boom {index}"
        )

    snapshot = request_metrics.client_failures_snapshot(days=7)

    assert len(snapshot["recent"]) == 20
    assert snapshot["recent"][0]["message"] == "boom 24"
    assert sum(day["count"] for day in snapshot["per_day"]) == 25


def test_client_failure_buffer_is_bounded() -> None:
    for index in range(request_metrics.CLIENT_FAILURE_BUFFER + 10):
        request_metrics.record_client_failure(
            path=None, status=None, source=None, message=str(index)
        )

    recent = request_metrics.client_failures_snapshot(
        days=1, recent=request_metrics.CLIENT_FAILURE_BUFFER + 10
    )["recent"]
    assert len(recent) == request_metrics.CLIENT_FAILURE_BUFFER


def test_background_tasks_after_the_response_are_not_charged_to_the_route() -> None:
    app = FastAPI()
    app.add_middleware(RequestMetricsMiddleware)

    @app.post("/auth/login")
    async def login(background: BackgroundTasks) -> dict:
        background.add_task(time.sleep, 0.3)
        return {"ok": True}

    TestClient(app).post("/auth/login")

    row = _route(request_metrics.snapshot(), "/auth/login", "2xx")
    assert row["count"] == 1
    assert row["max_ms"] < 250
