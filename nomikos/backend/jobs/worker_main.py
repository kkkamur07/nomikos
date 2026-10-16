"""Standalone platform job worker for production (Railway, Fly, etc.).

Runs the job types the platform executes itself. Segment and transcribe are not
among them: an inference agent claims those from the platform over HTTP
(ADR 0003). Disable the in-process worker on serverless API hosts with
JOB_WORKER_ENABLED=false.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal

from backend.core.settings import (
    get_infrastructure_settings,
    get_job_settings,
)
from backend.jobs.infrastructure.worker import worker_loop

logger = logging.getLogger(__name__)

_STOP_SIGNALS = (signal.SIGTERM, signal.SIGINT)


def validate_worker_settings() -> None:
    """Resolve worker dependencies before connecting to the database."""
    get_infrastructure_settings()
    get_job_settings()


async def _run() -> None:
    stop_event = asyncio.Event()

    def _request_stop(*_args: object) -> None:
        logger.info("platform job worker shutdown requested")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in _STOP_SIGNALS:
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, _request_stop)

    logger.info("platform job worker started")
    try:
        await worker_loop(stop_event)
    finally:
        # Handed back so a loop that outlives the worker (a test's, an embedding
        # process's) does not keep routing SIGTERM to an event nobody awaits.
        for sig in _STOP_SIGNALS:
            with contextlib.suppress(NotImplementedError):
                loop.remove_signal_handler(sig)
    logger.info("platform job worker stopped")


def main() -> None:
    validate_worker_settings()
    asyncio.run(_run())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
