"""Platform job status notifications: Postgres NOTIFY detection + local SSE fan-out."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import uuid
from collections.abc import Iterable
from typing import TypedDict

import asyncpg
from sqlalchemy import text

from backend.core.settings import get_infrastructure_settings
from backend.core.settings.job import get_job_settings
from backend.jobs.infrastructure.orm_models import JobStatus
from infrastructure.db import sync_system_session

logger = logging.getLogger(__name__)

_POSTGRES_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _quote_postgres_identifier(identifier: str) -> str:
    if not _POSTGRES_IDENTIFIER.fullmatch(identifier):
        raise ValueError(f"invalid PostgreSQL identifier: {identifier!r}")
    return f'"{identifier}"'


def _status_value(status: JobStatus | str) -> str:
    return status.value if isinstance(status, JobStatus) else status


class JobStatusNotification(TypedDict):
    """What one NOTIFY carries and what an SSE subscriber's queue receives."""

    job_id: str
    status: str


def notify_platform_job_statuses_changed(
    changes: Iterable[tuple[uuid.UUID, JobStatus | str]],
) -> None:
    """Emit committed platform job status changes to Postgres listeners.

    One session and one commit for the whole batch: the sweeps announce every
    row a bulk update moved, and a session per row turned one sweep into a
    round trip per job. Postgres delivers the batch when the commit lands.
    """
    payloads = [
        json.dumps(JobStatusNotification(job_id=str(job_id), status=_status_value(status)))
        for job_id, status in changes
    ]
    if not payloads:
        return
    try:
        with sync_system_session() as session:
            channel = get_job_settings().platform_job_notify_channel
            for payload in payloads:
                session.execute(
                    text("SELECT pg_notify(:channel, :payload)"),
                    {"channel": channel, "payload": payload},
                )
            session.commit()
    except Exception:
        logger.exception("failed to notify %s platform job status change(s)", len(payloads))


def notify_platform_job_status_changed(
    job_id: uuid.UUID,
    status: JobStatus | str,
) -> None:
    """Emit one committed platform job status change to Postgres listeners."""
    notify_platform_job_statuses_changed([(job_id, status)])


class JobStatusBroadcaster:
    """Per-process registry of SSE queues waiting for job status changes."""

    def __init__(self) -> None:
        self._subscribers: dict[uuid.UUID, set[asyncio.Queue[JobStatusNotification]]] = {}
        self._lock = asyncio.Lock()

    async def subscribe(self, job_id: uuid.UUID) -> asyncio.Queue[JobStatusNotification]:
        queue: asyncio.Queue[JobStatusNotification] = asyncio.Queue(maxsize=16)
        async with self._lock:
            self._subscribers.setdefault(job_id, set()).add(queue)
        return queue

    async def unsubscribe(
        self,
        job_id: uuid.UUID,
        queue: asyncio.Queue[JobStatusNotification],
    ) -> None:
        async with self._lock:
            queues = self._subscribers.get(job_id)
            if queues is None:
                return
            queues.discard(queue)
            if not queues:
                self._subscribers.pop(job_id, None)

    async def publish(self, job_id: uuid.UUID, payload: JobStatusNotification) -> None:
        async with self._lock:
            queues = tuple(self._subscribers.get(job_id, ()))
        for queue in queues:
            if queue.full():
                _ = queue.get_nowait()
            queue.put_nowait(payload)


job_status_broadcaster = JobStatusBroadcaster()


def _parse_notification_payload(
    payload: str,
) -> tuple[uuid.UUID, JobStatusNotification] | None:
    try:
        data = json.loads(payload)
        job_id = uuid.UUID(str(data["job_id"]))
        status = str(data.get("status", ""))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        logger.warning("Ignoring invalid platform job notification payload: %r", payload)
        return None
    return job_id, JobStatusNotification(job_id=str(job_id), status=status)


# Strong references to in-flight publishes. The event loop only keeps a weak
# reference to a task, so one nobody holds can be collected before it runs.
_publish_tasks: set[asyncio.Task[None]] = set()


async def platform_job_notification_loop(
    stop_event: asyncio.Event,
    broadcaster: JobStatusBroadcaster = job_status_broadcaster,
) -> None:
    """Forward Postgres NOTIFY payloads into this process' SSE subscribers."""
    channel = get_job_settings().platform_job_notify_channel
    _quote_postgres_identifier(channel)

    while not stop_event.is_set():
        connection: asyncpg.Connection | None = None

        async def publish_notification(raw_payload: str) -> None:
            parsed = _parse_notification_payload(raw_payload)
            if parsed is None:
                return
            job_id, payload = parsed
            await broadcaster.publish(job_id, payload)

        def handle_notification(
            _connection: asyncpg.Connection,
            _pid: int,
            _channel: str,
            raw_payload: str,
        ) -> None:
            task = asyncio.create_task(publish_notification(raw_payload))
            _publish_tasks.add(task)
            task.add_done_callback(_publish_tasks.discard)

        try:
            connection = await asyncpg.connect(get_infrastructure_settings().sync_database_url)
            await connection.add_listener(channel, handle_notification)
            logger.info("Listening for platform job notifications on %s", channel)
            await stop_event.wait()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("platform job notification listener failed")
            # Back off for a second before reconnecting. Timing out is the
            # normal path; an early wake means shutdown was requested.
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop_event.wait(), timeout=1.0)
        finally:
            if connection is not None:
                try:
                    await connection.remove_listener(channel, handle_notification)
                finally:
                    await connection.close()
