"""Unit tests for the standalone worker process and its poll loop.

``worker_loop`` is stubbed for the process tests and ``process_one_job`` for the
loop tests, so start-up order, shutdown on a signal and the idle back-off are
checked without a database. What a tick does is tested in test_job_lifecycle.py.
"""

from __future__ import annotations

import asyncio
import os
import signal
from types import SimpleNamespace

import pytest

from backend.jobs import worker_main
from backend.jobs.infrastructure import worker as worker_module

# --- Process start-up and shutdown ---


def test_main_checks_settings_before_the_worker_starts(monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    monkeypatch.setattr(
        worker_main, "get_infrastructure_settings", lambda: events.append("infrastructure")
    )
    monkeypatch.setattr(worker_main, "get_job_settings", lambda: events.append("job"))

    async def _loop(stop_event: asyncio.Event) -> None:
        events.append("loop" if not stop_event.is_set() else "loop already stopped")

    monkeypatch.setattr(worker_main, "worker_loop", _loop)

    worker_main.main()

    assert events == ["infrastructure", "job", "loop"]


def test_invalid_settings_stop_the_process_before_the_worker(monkeypatch: pytest.MonkeyPatch):
    started: list[asyncio.Event] = []

    def _broken() -> None:
        raise ValueError("DATABASE_URL is required")

    async def _loop(stop_event: asyncio.Event) -> None:
        started.append(stop_event)

    monkeypatch.setattr(worker_main, "get_infrastructure_settings", _broken)
    monkeypatch.setattr(worker_main, "worker_loop", _loop)

    with pytest.raises(ValueError, match="DATABASE_URL"):
        worker_main.main()
    assert started == []


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
async def test_a_stop_signal_ends_the_worker_and_releases_the_handler(
    monkeypatch: pytest.MonkeyPatch, sig: signal.Signals
):
    async def _loop(stop_event: asyncio.Event) -> None:
        os.kill(os.getpid(), sig)
        await asyncio.wait_for(stop_event.wait(), timeout=1.0)

    monkeypatch.setattr(worker_main, "worker_loop", _loop)

    await worker_main._run()

    # Nothing left installed: removing it again reports there was no handler.
    assert asyncio.get_running_loop().remove_signal_handler(sig) is False


async def test_a_crashing_worker_still_releases_the_handlers(monkeypatch: pytest.MonkeyPatch):
    async def _loop(_stop_event: asyncio.Event) -> None:
        raise RuntimeError("worker died")

    monkeypatch.setattr(worker_main, "worker_loop", _loop)

    with pytest.raises(RuntimeError, match="worker died"):
        await worker_main._run()
    loop = asyncio.get_running_loop()
    assert loop.remove_signal_handler(signal.SIGTERM) is False
    assert loop.remove_signal_handler(signal.SIGINT) is False


# --- Poll loop ---
# Tests the idle back-off: it doubles to the ceiling, survives a failed tick and
# resets after work.


async def test_the_idle_wait_doubles_to_the_ceiling_and_resets_after_work(
    monkeypatch: pytest.MonkeyPatch,
):
    stop_event = asyncio.Event()
    outcomes: list[bool | Exception] = [False, False, False, RuntimeError("tick"), True, False]
    waits: list[float] = []

    def _tick() -> bool:
        outcome = outcomes.pop(0)
        if not outcomes:
            stop_event.set()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def _wait_for(awaitable, timeout: float) -> None:
        waits.append(timeout)
        awaitable.close()
        if not stop_event.is_set():
            raise TimeoutError

    monkeypatch.setattr(
        worker_module,
        "get_job_settings",
        lambda: SimpleNamespace(job_poll_interval_seconds=1.0, job_poll_max_interval_seconds=5.0),
    )
    monkeypatch.setattr(worker_module, "process_one_job", _tick)
    monkeypatch.setattr(
        worker_module,
        "asyncio",
        SimpleNamespace(to_thread=asyncio.to_thread, wait_for=_wait_for),
    )

    await asyncio.wait_for(worker_module.worker_loop(stop_event), timeout=2.0)

    # 1, 2, 4, then capped at 5; the failed tick waits like an idle one; work
    # resets the next idle wait to the floor.
    assert waits == [1.0, 2.0, 4.0, 5.0, 1.0]
