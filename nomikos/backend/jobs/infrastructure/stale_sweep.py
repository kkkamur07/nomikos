"""Opportunistic stale-job recovery for hosts that run no background worker.

``clear_stale_callback_claims`` only ever ran
from ``worker.process_one_job``, i.e. inside ``worker_loop``, which
``backend.core.app`` starts only when ``job_worker_enabled`` is true. The
production API deployment sets ``JOB_WORKER_ENABLED=false`` because it is
request/response only, so stale claims were never released exactly where jobs
were reported stuck.

``release_expired_device_leases`` joins it here rather than anywhere else for
the same reason, and it is the reason the **lease** needs no machinery of its
own: ADR 0005 promises abandonment is "the existing stale sweep", and this is it.
A background lease reaper would be a process the serverless deployment cannot
run, which is exactly the trap this module already exists to avoid.

This module runs the same sweeps from ordinary job read paths instead:

* throttled to at most one sweep per process per
  ``job_stale_sweep_min_interval_seconds`` so a hot endpoint cannot turn into a
  sweep loop;
* serialized across API replicas with ``pg_try_advisory_xact_lock``: a replica
  that loses the race skips its turn rather than queueing, so a user request is
  never blocked waiting on another replica's sweep;
* run through ``asyncio.to_thread`` because the sweeps use sync sessions;
* never allowed to fail the request that triggered it.
"""

from __future__ import annotations

import asyncio
import logging
import time

from sqlalchemy import text

from backend.core.settings.device import get_device_settings
from backend.core.settings.job import get_job_settings
from backend.jobs.infrastructure.job_sweeps import (
    clear_stale_callback_claims,
    release_expired_device_leases,
)
from infrastructure.db import sync_system_session

logger = logging.getLogger(__name__)

# Advisory locks share one 64-bit namespace per database, so this must not
# collide with the inference queue-admission key or the test truncate key.
STALE_SWEEP_ADVISORY_LOCK_KEY = 8_402_762

_last_sweep_at: float | None = None


def _claim_throttle_window(now: float, min_interval_seconds: float) -> bool:
    """Return True when this caller owns the next sweep, and consume the window.

    Check-and-set is synchronous and the event loop is single threaded, so two
    requests arriving back to back cannot both pass: the first stamps the window
    before it awaits anything.
    """
    global _last_sweep_at
    if _last_sweep_at is not None and now - _last_sweep_at < min_interval_seconds:
        return False
    _last_sweep_at = now
    return True


def run_stale_job_sweep() -> int:
    """Re-pend expired leases and release stale callback claims.

    Blocking; call it in a thread. Returns the number of rows touched, or 0 when
    another replica already holds the sweep lock.
    """
    settings = get_job_settings()
    device_settings = get_device_settings()
    with sync_system_session() as session:
        # Transaction-scoped: the lock is released when this session's connection
        # is returned to the pool, even if the process dies mid-sweep. A
        # session-scoped lock could leak onto a pooled connection forever.
        acquired = session.execute(
            text("SELECT pg_try_advisory_xact_lock(:lock_key)"),
            {"lock_key": STALE_SWEEP_ADVISORY_LOCK_KEY},
        ).scalar_one()
        if not acquired:
            logger.debug("stale job sweep skipped: another replica holds the lock")
            return 0

        # Both sweeps open their own sessions, so they never contend with the
        # lock-holding transaction above. Every ``waiting`` page is one an
        # **inference agent** holds. These go back to the queue rather than
        # failing, because a closed laptop lid is not a failed job - until a page
        # has been abandoned
        # MAX_CLAIM_ATTEMPTS times, at which point the page is the problem and it
        # is failed. See release_expired_device_leases.
        re_pended = release_expired_device_leases(
            lease_seconds=device_settings.device_lease_seconds
        )
        if re_pended:
            logger.warning(
                "on-read sweep released %s page(s) whose device lease expired", re_pended
            )
        released = clear_stale_callback_claims(
            claim_timeout_seconds=settings.job_worker_callback_claim_timeout_seconds
        )
        if released:
            logger.warning("on-read sweep released %s stale inference callback claim(s)", released)
        return re_pended + released


async def sweep_stale_jobs_on_read() -> None:
    """Recover stale jobs before a client reads job state. Never raises.

    Safe to call on every job read: the throttle gate returns immediately in the
    common case, and the sweep itself is skipped when another replica is already
    running one.
    """
    settings = get_job_settings()
    if not settings.job_stale_sweep_on_read_enabled:
        return
    if not _claim_throttle_window(time.monotonic(), settings.job_stale_sweep_min_interval_seconds):
        return
    try:
        await asyncio.to_thread(run_stale_job_sweep)
    except Exception:
        # The user asked for a job, not for a sweep. Never fail their request.
        logger.exception("opportunistic stale job sweep failed")
