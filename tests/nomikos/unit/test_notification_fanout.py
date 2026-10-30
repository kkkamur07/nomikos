"""Unit tests for platform job status notifications: NOTIFY emission and SSE fan-out.

A recording session stands in for Postgres and a fake asyncpg connection for the
LISTEN side, so batching, payload parsing and the listener's lifecycle are
checked without a database.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import contextmanager

import pytest

from backend.jobs.infrastructure import notifications
from backend.jobs.infrastructure.notifications import (
    JobStatusBroadcaster,
    JobStatusNotification,
    notify_platform_job_status_changed,
    notify_platform_job_statuses_changed,
    platform_job_notification_loop,
)
from backend.jobs.infrastructure.orm_models import JobStatus


class _RecordingSession:
    def __init__(self) -> None:
        self.params: list[dict] = []
        self.commits = 0

    def execute(self, _statement, params: dict) -> None:
        self.params.append(params)

    def commit(self) -> None:
        self.commits += 1


def _use_sessions(monkeypatch: pytest.MonkeyPatch) -> list[_RecordingSession]:
    sessions: list[_RecordingSession] = []

    @contextmanager
    def _factory():
        session = _RecordingSession()
        sessions.append(session)
        yield session

    monkeypatch.setattr(notifications, "sync_system_session", _factory)
    return sessions


# --- NOTIFY emission ---
# Tests a batch of status changes costs one session and one commit.
# Does not test delivery, which is Postgres's job.


def test_a_batch_of_changes_uses_one_session_and_one_commit(monkeypatch: pytest.MonkeyPatch):
    sessions = _use_sessions(monkeypatch)
    first, second = uuid.uuid4(), uuid.uuid4()

    notify_platform_job_statuses_changed([(first, JobStatus.failed), (second, "pending")])

    assert len(sessions) == 1
    assert sessions[0].commits == 1
    payloads = [json.loads(params["payload"]) for params in sessions[0].params]
    assert payloads == [
        {"job_id": str(first), "status": "failed"},
        {"job_id": str(second), "status": "pending"},
    ]
    channel = notifications.get_job_settings().platform_job_notify_channel
    assert {params["channel"] for params in sessions[0].params} == {channel}


def test_an_empty_batch_opens_no_session(monkeypatch: pytest.MonkeyPatch):
    sessions = _use_sessions(monkeypatch)

    notify_platform_job_statuses_changed(iter(()))

    assert sessions == []


def test_a_single_change_is_a_batch_of_one(monkeypatch: pytest.MonkeyPatch):
    sessions = _use_sessions(monkeypatch)
    job_id = uuid.uuid4()

    notify_platform_job_status_changed(job_id, JobStatus.done)

    assert [json.loads(params["payload"]) for params in sessions[0].params] == [
        {"job_id": str(job_id), "status": "done"}
    ]


def test_a_failed_notify_is_logged_not_raised(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    @contextmanager
    def _broken():
        raise RuntimeError("database is down")
        yield

    monkeypatch.setattr(notifications, "sync_system_session", _broken)

    notify_platform_job_status_changed(uuid.uuid4(), JobStatus.failed)

    assert "failed to notify 1 platform job status change(s)" in caplog.text


# --- Payload parsing and local fan-out ---
# Tests what reaches an SSE subscriber's queue.


def test_a_valid_payload_is_parsed_to_the_notification_shape():
    job_id = uuid.uuid4()

    parsed = notifications._parse_notification_payload(
        json.dumps({"job_id": str(job_id), "status": "done", "extra": 1})
    )

    assert parsed == (job_id, {"job_id": str(job_id), "status": "done"})


@pytest.mark.parametrize("raw", ["not json", "[1]", '{"status": "done"}', '{"job_id": "nope"}'])
def test_an_invalid_payload_is_ignored(raw: str):
    assert notifications._parse_notification_payload(raw) is None


async def test_a_full_queue_drops_the_oldest_notification():
    broadcaster = JobStatusBroadcaster()
    job_id = uuid.uuid4()
    queue = await broadcaster.subscribe(job_id)

    for index in range(queue.maxsize + 1):
        await broadcaster.publish(job_id, JobStatusNotification(job_id=str(index), status="x"))

    assert queue.qsize() == queue.maxsize
    assert queue.get_nowait()["job_id"] == "1"


async def test_unsubscribing_the_last_queue_forgets_the_job():
    broadcaster = JobStatusBroadcaster()
    job_id = uuid.uuid4()
    queue = await broadcaster.subscribe(job_id)

    await broadcaster.unsubscribe(job_id, queue)
    await broadcaster.unsubscribe(job_id, queue)
    await broadcaster.publish(job_id, JobStatusNotification(job_id=str(job_id), status="done"))

    assert queue.empty()


# --- LISTEN loop ---
# Tests the listener forwards NOTIFY payloads, keeps its publish tasks alive,
# and cleans up its connection. Does not test asyncpg itself.


class _FakeConnection:
    def __init__(self) -> None:
        self.listeners: list = []
        self.closed = False

    async def add_listener(self, channel: str, callback) -> None:
        self.listeners.append((channel, callback))

    async def remove_listener(self, channel: str, callback) -> None:
        self.listeners.remove((channel, callback))

    async def close(self) -> None:
        self.closed = True


async def test_the_listener_forwards_a_notification_and_holds_its_task(
    monkeypatch: pytest.MonkeyPatch,
):
    connection = _FakeConnection()

    async def _connect(_url: str) -> _FakeConnection:
        return connection

    monkeypatch.setattr(notifications.asyncpg, "connect", _connect)
    broadcaster = JobStatusBroadcaster()
    job_id = uuid.uuid4()
    queue = await broadcaster.subscribe(job_id)
    stop_event = asyncio.Event()
    loop_task = asyncio.create_task(platform_job_notification_loop(stop_event, broadcaster))
    while not connection.listeners:
        await asyncio.sleep(0)

    channel, callback = connection.listeners[0]
    callback(connection, 1, channel, json.dumps({"job_id": str(job_id), "status": "done"}))
    # Held strongly until it has run, so the loop cannot collect it mid-flight.
    assert len(notifications._publish_tasks) == 1
    received = await asyncio.wait_for(queue.get(), timeout=1.0)
    await asyncio.sleep(0)

    assert received == {"job_id": str(job_id), "status": "done"}
    assert notifications._publish_tasks == set()
    assert channel == notifications.get_job_settings().platform_job_notify_channel

    stop_event.set()
    await asyncio.wait_for(loop_task, timeout=1.0)
    assert connection.listeners == []
    assert connection.closed


async def test_a_failed_connect_backs_off_until_shutdown(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    stop_event = asyncio.Event()
    attempts: list[str] = []

    async def _connect(url: str):
        attempts.append(url)
        stop_event.set()
        raise OSError("connection refused")

    monkeypatch.setattr(notifications.asyncpg, "connect", _connect)

    await asyncio.wait_for(platform_job_notification_loop(stop_event), timeout=1.0)

    assert len(attempts) == 1
    assert "platform job notification listener failed" in caplog.text


async def test_an_unsafe_channel_name_is_refused(monkeypatch: pytest.MonkeyPatch):
    settings = notifications.get_job_settings().model_copy(
        update={"platform_job_notify_channel": "jobs; DROP TABLE jobs"}
    )
    monkeypatch.setattr(notifications, "get_job_settings", lambda: settings)

    with pytest.raises(ValueError, match="invalid PostgreSQL identifier"):
        await platform_job_notification_loop(asyncio.Event())
