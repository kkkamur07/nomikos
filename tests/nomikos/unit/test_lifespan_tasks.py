"""Lifespan startup is decided by a pure plan and shut down cleanly."""

from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from backend.core import app as app_module
from backend.core.app import create_app, plan_lifespan_tasks
from backend.core.settings import reset_settings_caches
from backend.core.settings.job import JobSettings

STRONG_SECRET = "xQ7v2Kd9RmZ4pB6wLt1yHs3nCf8jUa5eG0oV"


def _jobs(**overrides) -> JobSettings:
    values = {
        "JOB_WORKER_ENABLED": "true",
        "JOB_SSE_NOTIFICATIONS_ENABLED": "true",
        "MEDIA_GC_ENABLED": "true",
    }
    values.update(overrides)
    return JobSettings(**{k: v == "true" for k, v in values.items()})


def test_plan_runs_everything_in_development_by_default() -> None:
    plan = plan_lifespan_tasks(environment="development", job_settings=_jobs())

    assert plan == app_module.LifespanPlan(
        ensure_dev_user=True, notifications=True, worker=True, media_gc=True
    )


def test_plan_is_quiet_in_production_with_everything_off() -> None:
    plan = plan_lifespan_tasks(
        environment="production",
        job_settings=_jobs(
            JOB_WORKER_ENABLED="false",
            JOB_SSE_NOTIFICATIONS_ENABLED="false",
            MEDIA_GC_ENABLED="false",
        ),
    )

    assert plan == app_module.LifespanPlan(
        ensure_dev_user=False, notifications=False, worker=False, media_gc=False
    )


def test_media_gc_can_be_gated_off_while_the_worker_runs() -> None:
    plan = plan_lifespan_tasks(
        environment="development", job_settings=_jobs(MEDIA_GC_ENABLED="false")
    )

    assert plan.media_gc is False
    assert plan.worker is True


async def test_dev_seed_success_and_failure_are_both_quiet(monkeypatch) -> None:
    import backend.dev.bootstrap as bootstrap

    async def _created(session) -> bool:
        return True

    async def _exists(session) -> bool:
        return False

    async def _broken(session) -> bool:
        raise ConnectionError("db is down")

    monkeypatch.setattr(bootstrap, "ensure_dev_user_exists", _created)
    await app_module._ensure_dev_user_at_startup()
    monkeypatch.setattr(bootstrap, "ensure_dev_user_exists", _exists)
    await app_module._ensure_dev_user_at_startup()
    monkeypatch.setattr(bootstrap, "ensure_dev_user_exists", _broken)
    await app_module._ensure_dev_user_at_startup()


def test_lifespan_starts_and_stops_every_planned_task(monkeypatch) -> None:
    monkeypatch.setenv("JWT_SECRET", STRONG_SECRET)
    reset_settings_caches()
    started: list[str] = []

    async def _loop(name: str, stop_event: asyncio.Event) -> None:
        started.append(name)
        await stop_event.wait()

    async def _notifications(stop_event) -> None:
        await _loop("notifications", stop_event)

    async def _worker(stop_event) -> None:
        await _loop("worker", stop_event)

    async def _gc(stop_event) -> None:
        await _loop("media_gc", stop_event)

    async def _no_dev_user() -> None:
        started.append("dev_user")

    monkeypatch.setattr(app_module, "platform_job_notification_loop", _notifications)
    monkeypatch.setattr(app_module, "worker_loop", _worker)
    monkeypatch.setattr(app_module, "media_gc_loop", _gc)
    monkeypatch.setattr(app_module, "_ensure_dev_user_at_startup", _no_dev_user)

    with TestClient(create_app()):
        assert sorted(started) == ["dev_user", "media_gc", "notifications", "worker"]
