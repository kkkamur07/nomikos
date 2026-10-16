"""Unit tests for job lifecycle sweeps, worker ownership, and history clearing.

Statement-level tests drive the repository with a recording session so the SQL
and the notification fan-out are checked without Postgres. Real round-trips live
in tests/nomikos/integration/test_jobs.py.
"""

from __future__ import annotations

import inspect
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import Select, TextClause
from sqlalchemy.dialects import postgresql

from backend.core.exceptions import ConflictError
from backend.core.settings.device import get_device_settings
from backend.core.settings.job import JobSettings, get_job_settings
from backend.jobs.infrastructure import job_claim_engine, job_repository, job_sweeps, job_terminal
from backend.jobs.infrastructure import stale_sweep as stale_sweep_module
from backend.jobs.infrastructure import worker as worker_module
from backend.jobs.infrastructure.job_claim_engine import claim_next_pending_job, worker_identity
from backend.jobs.infrastructure.job_sql import AGENT_CLAIM_PREFIX
from backend.jobs.infrastructure.job_sweeps import (
    clear_stale_callback_claims,
    reclaim_stale_running_jobs,
)
from backend.jobs.infrastructure.job_terminal import (
    apply_cancellation,
    mark_job_done,
    mark_job_failed,
)
from backend.jobs.infrastructure.orm_models import Job, JobStatus, JobType
from backend.ml.domain.execution import ExecutionTarget


class _FakeResult:
    def __init__(self, rows: list | None = None, rowcount: int = 0) -> None:
        self._rows = list(rows or [])
        self.rowcount = rowcount

    def scalars(self):
        return self

    def all(self) -> list:
        return list(self._rows)

    def __iter__(self):
        return iter(self._rows)

    def scalar_one_or_none(self):
        return self._rows[0] if self._rows else None

    def scalar_one(self):
        return self._rows[0]


class _FakeSession:
    """Records executed statements instead of talking to Postgres."""

    def __init__(self, results: list[_FakeResult] | None = None) -> None:
        self._results = list(results or [])
        self.statements: list = []
        self.commits = 0

    def execute(self, statement, *_args, **_kwargs) -> _FakeResult:
        self.statements.append(statement)
        return self._results.pop(0) if self._results else _FakeResult()

    def commit(self) -> None:
        self.commits += 1

    def refresh(self, _instance) -> None:
        return None


class _FakeAsyncSession:
    """Async counterpart of ``_FakeSession`` for the repository's read/write path."""

    def __init__(self, rowcount: int = 0) -> None:
        self.statements: list = []
        self.commits = 0
        self._rowcount = rowcount

    async def execute(self, statement, *_args, **_kwargs) -> _FakeResult:
        self.statements.append(statement)
        return _FakeResult(rowcount=self._rowcount)

    async def commit(self) -> None:
        self.commits += 1


# The sync claim/sweep/terminal half of job persistence, each opening its own
# sessions and announcing its own status changes.
_ENGINE_MODULES = (job_claim_engine, job_sweeps, job_terminal)


def _use_session(monkeypatch: pytest.MonkeyPatch, session: _FakeSession) -> None:
    @contextmanager
    def _factory():
        yield session

    for module in _ENGINE_MODULES:
        monkeypatch.setattr(module, "sync_system_session", _factory)


def _record_notifications(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    notified: list[tuple] = []
    for module in (job_claim_engine, job_terminal):
        monkeypatch.setattr(
            module,
            "notify_platform_job_status_changed",
            lambda job_id, status: notified.append((job_id, status)),
        )
    monkeypatch.setattr(job_sweeps, "notify_platform_job_statuses_changed", notified.extend)
    return notified


def _sql(statement) -> str:
    return str(statement.compile(dialect=postgresql.dialect()))


def _params(statement) -> dict:
    return statement.compile(dialect=postgresql.dialect()).params


# --- Stale callback claims ---
# Tests an abandoned callback claim stops blocking cancellation.
# Does not test the callback merge itself.


def test_clear_stale_callback_claims_releases_non_terminal_jobs(monkeypatch: pytest.MonkeyPatch):
    session = _FakeSession([_FakeResult(rowcount=3)])
    _use_session(monkeypatch, session)

    assert clear_stale_callback_claims(claim_timeout_seconds=300.0) == 3

    sql = _sql(session.statements[0])
    assert "callback_claimed_at IS NOT NULL" in sql
    # Bumping updated_at here would push the device-lease deadline out another window.
    assert "updated_at=jobs.updated_at" in sql
    values = _params(session.statements[0])
    assert values["callback_claimed_at"] is None
    assert set(values["status_1"]) == {JobStatus.pending, JobStatus.running, JobStatus.waiting}


def test_job_with_stale_callback_claim_becomes_cancellable_again():
    job = Job(
        id=uuid.uuid4(),
        type=JobType.segment,
        status=JobStatus.waiting,
        payload={},
        callback_claimed_at=datetime.now(UTC) - timedelta(minutes=30),
    )
    now = datetime.now(UTC)

    with pytest.raises(ConflictError):
        apply_cancellation(job, now)

    # Exactly what clear_stale_callback_claims writes to the row.
    job.callback_claimed_at = None

    apply_cancellation(job, now)
    assert job.status == JobStatus.cancelled
    assert job.completed_at == now
    assert job.result is None


# --- Opportunistic on-read sweep ---
# Tests stale-job recovery still happens with no background worker, which is the
# production API deployment (JOB_WORKER_ENABLED=false). Without this an expired
# device lease is never released there and a page sits in waiting forever. Does
# not test the worker loop, which sweeps on its own tick.


class _SweepStore:
    """One in-memory job plus counters, driven by the real sweep statements."""

    def __init__(self, job: Job, *, lock_granted: bool = True) -> None:
        self.job = job
        self.lock_granted = lock_granted
        self.lock_attempts = 0
        self.lease_selects = 0
        self.claim_clears = 0


def _wants_agent_claims(statement) -> bool:
    """Whether this statement is restricted to agent-held pages, read from its own SQL.

    The fake has to model ``claimed_by``: the lease sweep may only ever touch a
    page an agent holds.
    """
    return "jobs.claimed_by LIKE" in _sql(statement)


def _agent_held(job: Job) -> bool:
    return bool(job.claimed_by and job.claimed_by.startswith(AGENT_CLAIM_PREFIX))


class _SweepSession:
    """Interprets the sweep's own SQL against ``_SweepStore`` instead of Postgres."""

    def __init__(self, store: _SweepStore) -> None:
        self._store = store

    def execute(self, statement, *_args, **_kwargs) -> _FakeResult:
        store = self._store
        job = store.job
        if isinstance(statement, TextClause):
            store.lock_attempts += 1
            return _FakeResult(rows=[store.lock_granted])

        values = _params(statement)
        wants_agent = _wants_agent_claims(statement)
        matches_claim = not wants_agent or _agent_held(job)

        if isinstance(statement, Select):
            store.lease_selects += 1
            stale = (
                job.status == JobStatus.waiting
                and matches_claim
                and job.updated_at <= values["updated_at_1"]
            )
            if not stale:
                return _FakeResult(rows=[])
            # The lease sweep reads the claim budget with the id, because it has
            # to decide per row whether to re-pend or fail.
            return _FakeResult(rows=[(job.id, job.claim_attempts)])

        if "status" in values:  # release_expired_device_leases
            if job.status != JobStatus.waiting or not matches_claim:
                return _FakeResult(rowcount=0)
            job.status = values["status"]
            job.callback_claimed_at = None
            if "claim_attempts" in _sql(statement):
                job.claim_attempts = (job.claim_attempts or 0) + 1
            if values["status"] == JobStatus.pending:  # the lease went back to the queue
                job.claimed_by = None
                job.inference_job_id = None
                job.started_at = None
                job.heartbeat_at = None
            else:
                job.error = values["error"]
                job.completed_at = values["completed_at"]
            return _FakeResult(rowcount=1)

        store.claim_clears += 1  # clear_stale_callback_claims
        claimed_at = job.callback_claimed_at
        if claimed_at is None or claimed_at > values["callback_claimed_at_1"]:
            return _FakeResult(rowcount=0)
        job.callback_claimed_at = None
        return _FakeResult(rowcount=1)

    def commit(self) -> None:
        return None

    def refresh(self, _instance) -> None:
        return None


def _install_sweep_store(
    monkeypatch: pytest.MonkeyPatch, job: Job, *, lock_granted: bool = True
) -> _SweepStore:
    store = _SweepStore(job, lock_granted=lock_granted)

    @contextmanager
    def _factory():
        yield _SweepSession(store)

    # The advisory lock runs on the sweep module's session; the two sweeps open
    # their own sessions inside the repository module.
    monkeypatch.setattr(stale_sweep_module, "sync_system_session", _factory)
    for module in _ENGINE_MODULES:
        monkeypatch.setattr(module, "sync_system_session", _factory)
    return store


@pytest.fixture(autouse=True)
def _clean_sweep_throttle():
    """The throttle is process-global; never let it leak between tests."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(stale_sweep_module, "_last_sweep_at", None)
        yield


@pytest.fixture
def job_env(monkeypatch: pytest.MonkeyPatch):
    """Rebuild the cached JobSettings from env for the duration of one test."""

    def _apply(**env: str) -> JobSettings:
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        get_job_settings.cache_clear()
        return get_job_settings()

    yield _apply
    get_job_settings.cache_clear()


class _StubUser:
    def __init__(self, user_id: uuid.UUID) -> None:
        self.id = user_id


class _StubJobService:
    def __init__(self, job: Job) -> None:
        self._job = job
        self.reads = 0

    async def get_job(self, _job_id: uuid.UUID) -> Job:
        self.reads += 1
        return self._job


def _readable_job(**overrides) -> Job:
    """A job with every column the JobResponse DTO needs already populated."""
    now = datetime.now(UTC)
    fields = {
        "id": uuid.uuid4(),
        "type": JobType.segment,
        "status": JobStatus.waiting,
        "payload": {},
        "result": None,
        "error": None,
        "user_id": uuid.uuid4(),
        "document_id": None,
        "document_part_id": None,
        "inference_job_id": uuid.uuid4(),
        "created_at": now,
        "updated_at": now,
        "started_at": None,
        "completed_at": None,
        # Not defaulted by the ORM until the row is flushed, and the DTO
        # names the inference host on every job, so it has to be set here.
        "execution_target": ExecutionTarget.cloud,
        "preferred_execution_target": ExecutionTarget.cloud,
    }
    fields.update(overrides)
    return Job(**fields)


async def test_read_releases_an_expired_lease_when_no_worker_runs(
    monkeypatch: pytest.MonkeyPatch, job_env
):
    from backend.jobs.api import jobs as jobs_api

    settings = job_env(JOB_WORKER_ENABLED="false")
    # Premise: app.py never starts worker_loop, so process_one_job never sweeps.
    assert settings.job_worker_enabled is False

    lease = get_device_settings().device_lease_seconds
    job = _readable_job(
        claimed_by=f"{AGENT_CLAIM_PREFIX}{uuid.uuid4()}",
        updated_at=datetime.now(UTC) - timedelta(seconds=lease + 60),
    )
    store = _install_sweep_store(monkeypatch, job)
    notified = _record_notifications(monkeypatch)
    service = _StubJobService(job)

    response = await jobs_api.get_job(
        job_id=job.id, service=service, current_user=_StubUser(job.user_id)
    )

    assert store.lock_attempts == 1
    assert response.status == JobStatus.pending
    # The read must sweep first: reading a stale row and reporting it as alive is
    # exactly the bug.
    assert service.reads == 1
    assert notified == [(job.id, JobStatus.pending)]


async def test_two_quick_reads_run_only_one_sweep(monkeypatch: pytest.MonkeyPatch, job_env):
    job_env(JOB_STALE_SWEEP_MIN_INTERVAL_SECONDS="30")
    job = _readable_job(updated_at=datetime.now(UTC) - timedelta(seconds=600))
    store = _install_sweep_store(monkeypatch, job)
    _record_notifications(monkeypatch)

    await stale_sweep_module.sweep_stale_jobs_on_read()
    await stale_sweep_module.sweep_stale_jobs_on_read()

    # A sweep per request would put two extra round-trips on every poll.
    assert store.lock_attempts == 1
    assert store.lease_selects == 1

    monkeypatch.setattr(stale_sweep_module, "_last_sweep_at", None)
    await stale_sweep_module.sweep_stale_jobs_on_read()
    assert store.lock_attempts == 2


async def test_a_failing_sweep_does_not_fail_the_read(monkeypatch: pytest.MonkeyPatch):
    from backend.jobs.api import jobs as jobs_api

    def _boom() -> int:
        raise RuntimeError("sweep exploded")

    monkeypatch.setattr(stale_sweep_module, "run_stale_job_sweep", _boom)

    job = _readable_job()
    service = _StubJobService(job)

    response = await jobs_api.get_job(
        job_id=job.id, service=service, current_user=_StubUser(job.user_id)
    )

    assert response.id == job.id
    assert service.reads == 1


async def test_read_clears_a_stale_callback_claim_so_cancel_works(
    monkeypatch: pytest.MonkeyPatch, job_env
):
    from backend.jobs.api import jobs as jobs_api

    job_env(JOB_WORKER_ENABLED="false")
    # Fresh updated_at: the lease must not fire, only the claim release.
    job = _readable_job(
        updated_at=datetime.now(UTC),
        callback_claimed_at=datetime.now(UTC) - timedelta(minutes=30),
    )
    store = _install_sweep_store(monkeypatch, job)
    _record_notifications(monkeypatch)

    with pytest.raises(ConflictError):
        apply_cancellation(job, datetime.now(UTC))

    await jobs_api.get_job(
        job_id=job.id, service=_StubJobService(job), current_user=_StubUser(job.user_id)
    )

    assert store.claim_clears == 1
    assert job.callback_claimed_at is None
    assert job.status == JobStatus.waiting

    apply_cancellation(job, datetime.now(UTC))
    assert job.status == JobStatus.cancelled


async def test_sweep_skips_when_another_replica_holds_the_lock(
    monkeypatch: pytest.MonkeyPatch, job_env
):
    job = _readable_job(updated_at=datetime.now(UTC) - timedelta(seconds=600))
    store = _install_sweep_store(monkeypatch, job, lock_granted=False)
    notified = _record_notifications(monkeypatch)

    await stale_sweep_module.sweep_stale_jobs_on_read()

    assert store.lock_attempts == 1
    assert store.lease_selects == 0
    assert job.status == JobStatus.waiting
    assert notified == []


async def test_sweep_flag_disables_the_on_read_recovery(monkeypatch: pytest.MonkeyPatch, job_env):
    job_env(JOB_STALE_SWEEP_ON_READ_ENABLED="false")
    job = _readable_job(updated_at=datetime.now(UTC) - timedelta(seconds=600))
    store = _install_sweep_store(monkeypatch, job)

    await stale_sweep_module.sweep_stale_jobs_on_read()

    assert store.lock_attempts == 0
    assert job.status == JobStatus.waiting


# --- Worker ownership ---
# Tests claims are attributed and terminal writes stay with their owner.
# Does not test the inference callback path, which finalizes waiting jobs elsewhere.


def test_claim_records_the_worker_identity(monkeypatch: pytest.MonkeyPatch):
    job = Job(
        id=uuid.uuid4(),
        type=JobType.pipeline,
        status=JobStatus.pending,
        payload={"test": True},
    )
    _use_session(monkeypatch, _FakeSession([_FakeResult(rows=[job])]))
    _record_notifications(monkeypatch)

    claimed = claim_next_pending_job(test_only=True)

    assert claimed is job
    assert claimed.status == JobStatus.running
    assert claimed.claimed_by == worker_identity()
    assert claimed.heartbeat_at is not None


def test_reclaim_releases_worker_ownership(monkeypatch: pytest.MonkeyPatch):
    session = _FakeSession([_FakeResult(rows=[(uuid.uuid4(), 0)]), _FakeResult(rowcount=1)])
    _use_session(monkeypatch, session)
    _record_notifications(monkeypatch)

    assert reclaim_stale_running_jobs(running_timeout_seconds=1800.0) == 1

    # The claim budget is read under the same lock the release takes, so a second
    # sweeper cannot decide from a stale count.
    assert "FOR UPDATE SKIP LOCKED" in _sql(session.statements[0])
    values = _params(session.statements[1])
    assert values["status"] == JobStatus.pending
    assert values["claimed_by"] is None
    assert values["heartbeat_at"] is None
    # The lap is recorded, or a page that kills every worker cycles forever.
    assert "claim_attempts=(jobs.claim_attempts + " in _sql(session.statements[1])


def test_reclaim_fails_a_job_that_exhausted_its_claim_budget(monkeypatch: pytest.MonkeyPatch):
    job_id = uuid.uuid4()
    session = _FakeSession(
        [
            _FakeResult(rows=[(job_id, job_sweeps.MAX_CLAIM_ATTEMPTS - 1)]),
            _FakeResult(rowcount=1),
        ]
    )
    _use_session(monkeypatch, session)
    notified = _record_notifications(monkeypatch)

    assert reclaim_stale_running_jobs(running_timeout_seconds=1800.0) == 1

    values = _params(session.statements[1])
    assert values["status"] == JobStatus.failed
    assert values["error"] == job_sweeps.poison_page_error(job_sweeps.MAX_CLAIM_ATTEMPTS)
    assert values["completed_at"] is not None
    # Terminal, so a browser watching this job has to be told; a re-pend is not
    # announced because pending is not a state the UI renders differently.
    assert notified == [(job_id, JobStatus.failed)]


def test_mark_job_done_is_scoped_to_the_owning_claim(monkeypatch: pytest.MonkeyPatch):
    session = _FakeSession([_FakeResult(rowcount=0)])
    _use_session(monkeypatch, session)
    notified = _record_notifications(monkeypatch)

    mark_job_done(uuid.uuid4(), {"ok": True}, claimed_by="host:1")

    sql = _sql(session.statements[0])
    assert "claimed_by IS NOT DISTINCT FROM" in sql
    assert _params(session.statements[0])["claimed_by_1"] == "host:1"
    # The row was reclaimed by another worker: no write, so no status broadcast.
    assert notified == []


def test_mark_job_failed_is_scoped_to_the_owning_claim(monkeypatch: pytest.MonkeyPatch):
    job_id = uuid.uuid4()
    session = _FakeSession([_FakeResult(rowcount=1)])
    _use_session(monkeypatch, session)
    notified = _record_notifications(monkeypatch)

    mark_job_failed(job_id, "Job failed", claimed_by="host:1")

    assert "claimed_by IS NOT DISTINCT FROM" in _sql(session.statements[0])
    assert notified == [(job_id, JobStatus.failed)]


def test_unclaimed_job_still_accepts_its_terminal_write(monkeypatch: pytest.MonkeyPatch):
    session = _FakeSession([_FakeResult(rowcount=1)])
    _use_session(monkeypatch, session)
    _record_notifications(monkeypatch)

    mark_job_failed(uuid.uuid4(), "Job type is not supported", claimed_by=None)

    assert "claimed_by IS NOT DISTINCT FROM NULL" in _sql(session.statements[0])


def test_execute_claimed_job_passes_the_claim_owner(monkeypatch: pytest.MonkeyPatch):
    job = Job(
        id=uuid.uuid4(),
        type=JobType.pipeline,
        status=JobStatus.running,
        payload={"handler": "noop", "test": True},
        claimed_by="host:99",
    )
    recorded: dict = {}
    monkeypatch.setattr(worker_module, "run_test_handler", lambda _job: {"ok": True})
    monkeypatch.setattr(
        worker_module,
        "mark_job_done",
        lambda job_id, result, *, claimed_by: recorded.update(
            job_id=job_id, result=result, claimed_by=claimed_by
        ),
    )

    worker_module.execute_claimed_job(job)

    assert recorded == {"job_id": job.id, "result": {"ok": True}, "claimed_by": "host:99"}


# --- Worker sweep wiring ---
# Tests the order the stale-state sweeps run in, and the deadline each is handed.
# Does not test handler dispatch, and does not test what any sweep does to a row --
# that needs Postgres and lives in tests/nomikos/integration/test_job_worker_sweeps.py.


def test_process_one_job_runs_every_stale_sweep(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[str, dict]] = []

    def _sweep(name: str, count: int):
        def _run(**kwargs) -> int:
            calls.append((name, kwargs))
            return count

        return _run

    # All three sweeps, not two. An unpatched sweep does not make this test more
    # realistic -- it makes `process_one_job` open a live Postgres connection from the
    # lane CI runs with no database, so the test fails on connectivity rather than on
    # the ordering it exists to pin.
    monkeypatch.setattr(worker_module, "reclaim_stale_running_jobs", _sweep("reclaim", 1))
    monkeypatch.setattr(worker_module, "release_expired_device_leases", _sweep("lease", 3))
    monkeypatch.setattr(worker_module, "clear_stale_callback_claims", _sweep("claims", 4))
    monkeypatch.setattr(worker_module, "claim_next_pending_job", lambda **_kwargs: None)

    assert worker_module.process_one_job() is False

    settings = get_job_settings()
    # Leases are released before callback claims, as on the read path.
    assert [name for name, _ in calls] == ["reclaim", "lease", "claims"]
    assert calls[0][1] == {"running_timeout_seconds": settings.job_worker_running_timeout_seconds}
    assert calls[1][1] == {"lease_seconds": get_device_settings().device_lease_seconds}
    assert calls[2][1] == {
        "claim_timeout_seconds": settings.job_worker_callback_claim_timeout_seconds
    }


# --- Clear job history ---
# Tests the terminal-only filter on DELETE /jobs/history.
# Does not test project membership, which reuses ProjectService.get_project.


async def test_clear_history_deletes_only_terminal_project_jobs():
    session = _FakeAsyncSession(rowcount=4)
    project_id = uuid.uuid4()

    deleted = await job_repository.JobRepository(session).delete_terminal_jobs_for_project(
        project_id
    )

    assert deleted == 4
    assert session.commits == 1

    sql = _sql(session.statements[0])
    assert sql.startswith("DELETE FROM jobs")
    assert "JOIN documents ON jobs.document_id = documents.id" in sql
    values = _params(session.statements[0])
    assert values["project_id_1"] == project_id
    # pending/running/waiting are absent, so active jobs survive the clear.
    assert set(values["status_1"]) == {JobStatus.done, JobStatus.failed, JobStatus.cancelled}


# --- SSE stream dependencies ---
# Tests the event stream does not hold a request-scoped session.
# Does not test live streaming, which needs Postgres.


def test_stream_job_events_takes_no_request_session():
    from backend.jobs.api.jobs import stream_job_events
    from infrastructure.db import get_db

    parameters = inspect.signature(stream_job_events).parameters
    assert "service" not in parameters
    assert "db" not in parameters
    defaults = [parameter.default for parameter in parameters.values()]
    assert all(getattr(default, "dependency", None) is not get_db for default in defaults)


async def test_job_event_stream_ends_cleanly_when_access_is_lost(
    monkeypatch: pytest.MonkeyPatch,
):
    from backend.core.exceptions import NotFoundError
    from backend.jobs.api import jobs as jobs_api

    async def _denied(*_args, **_kwargs):
        raise NotFoundError("job gone")

    monkeypatch.setattr(jobs_api, "_load_authorized_job", _denied)

    class _Request:
        async def is_disconnected(self) -> bool:
            return True

    chunks = [chunk async for chunk in jobs_api._job_events(uuid.uuid4(), object(), _Request())]

    # A hung stream is the failure mode we are avoiding; it must terminate.
    assert len(chunks) == 1
    assert chunks[0].startswith("event: error")
    assert "Job stream closed" in chunks[0]


# --- Baseline schema ---
# ``test_baseline_migration_builds_the_job_queue_columns`` and
# ``test_job_model_exposes_worker_ownership_columns`` stood here. Column presence
# and nullability are settled by
# ``integration/test_migrations.py::test_migration_chain_matches_orm_metadata``,
# which diffs the whole migrated schema against ``Base.metadata`` on a real
# Postgres instead of a monkeypatched ``op``. The one claim that check cannot
# make - autogenerate does not compare triggers - is
# ``jobs_execution_target_is_fixed``, and that is covered far better than by the
# substring search that stood here: ``integration/test_execution_target.py``
# issues a raw ``UPDATE jobs SET execution_target`` against Postgres and asserts
# it is refused, with a sibling confirming every other column stays writable.


# --- Poison-page ceiling ---
# Tests a page that is abandoned over and over eventually reaches a terminal
# status. Does not test the lease duration, which decides *when* a claim is
# considered abandoned.


async def test_a_repeatedly_abandoned_page_is_finally_failed(
    monkeypatch: pytest.MonkeyPatch, job_env
):
    """Re-pending is right until the page itself is the reason.

    Before the counter this loop had no exit: an image the model crashes on took
    down every agent that claimed it, so the lease sweep re-pended it forever.
    The job never reached a terminal status, so nothing ever told the researcher,
    and it held one claim slot on every lap.
    """
    job_env(JOB_WORKER_ENABLED="false")
    job = _readable_job(
        claimed_by=f"{AGENT_CLAIM_PREFIX}{uuid.uuid4()}",
        claim_attempts=0,
    )
    _record_notifications(monkeypatch)
    store = _install_sweep_store(monkeypatch, job)

    for lap in range(job_sweeps.MAX_CLAIM_ATTEMPTS):
        # What the next agent's claim writes: the sweep cleared the claim, and
        # only an agent-held page is under the lease.
        job.status = JobStatus.waiting
        job.claimed_by = f"{AGENT_CLAIM_PREFIX}{uuid.uuid4()}"
        job.updated_at = datetime.now(UTC) - timedelta(days=1)
        monkeypatch.setattr(stale_sweep_module, "_last_sweep_at", None)
        await stale_sweep_module.sweep_stale_jobs_on_read()
        if lap < job_sweeps.MAX_CLAIM_ATTEMPTS - 1:
            assert job.status == JobStatus.pending, f"lap {lap} should still be retried"

    assert store.lease_selects == job_sweeps.MAX_CLAIM_ATTEMPTS
    assert job.status == JobStatus.failed
    assert job.error == job_sweeps.poison_page_error(job_sweeps.MAX_CLAIM_ATTEMPTS)


def test_a_finished_job_starts_over_with_a_full_claim_budget(monkeypatch: pytest.MonkeyPatch):
    """Success is what resets the counter, on both paths that can produce it."""
    session = _FakeSession([_FakeResult(rowcount=1)])
    _use_session(monkeypatch, session)
    _record_notifications(monkeypatch)

    mark_job_done(uuid.uuid4(), {"ok": True}, claimed_by="host:1")
    assert _params(session.statements[0])["claim_attempts"] == 0

    from backend.jobs.application.job_callback_service import _mark_done_from_callback_sync

    job = _readable_job(claim_attempts=3)
    _mark_done_from_callback_sync(job, {"ok": True})
    assert job.claim_attempts == 0
