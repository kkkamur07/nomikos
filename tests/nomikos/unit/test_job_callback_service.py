"""Unit tests for the inference callback transaction shape and failure reporting.

A recording session stands in for Postgres so the commit boundaries and the
notification ordering are checkable without a database. Real rollback semantics
- the thing the single transaction actually buys - are proven against Postgres in
tests/nomikos/integration/test_ml_job_callback.py.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from datetime import UTC, datetime

import pytest

from backend.jobs.application import job_callback_service as service
from backend.jobs.application.job_callback_service import (
    CALLBACK_PROCESSING_ERROR,
    INFERENCE_FAILURE_ERROR,
    _apply_callback_locked,
    _merge_and_finalize,
    _merge_context,
    _public_callback_error,
)
from backend.jobs.infrastructure.orm_models import Job, JobStatus, JobType
from nomikos_inference.contracts.jobs import JobCallbackRequest


class _FakeResult:
    def __init__(self, job: Job | None) -> None:
        self._job = job

    def scalar_one_or_none(self) -> Job | None:
        return self._job


class _FakeSession:
    """Records commits instead of talking to Postgres; every read returns ``job``."""

    def __init__(self, job: Job | None, events: list[str], index: int) -> None:
        self._job = job
        self._events = events
        self._index = index
        self.commits = 0

    def execute(self, _statement, *_args, **_kwargs) -> _FakeResult:
        return _FakeResult(self._job)

    def commit(self) -> None:
        self.commits += 1
        self._events.append(f"commit:{self._index}")


def _use_sessions(
    monkeypatch: pytest.MonkeyPatch, job: Job | None, events: list[str]
) -> list[_FakeSession]:
    sessions: list[_FakeSession] = []

    @contextmanager
    def _factory():
        session = _FakeSession(job, events, len(sessions))
        sessions.append(session)
        yield session

    monkeypatch.setattr(service, "sync_system_session", _factory)
    return sessions


def _record_notifications(monkeypatch: pytest.MonkeyPatch, events: list[str]) -> list[tuple]:
    notified: list[tuple] = []

    def _notify(job_id, status) -> None:
        notified.append((job_id, status))
        events.append(f"notify:{status.value}")

    monkeypatch.setattr(service, "notify_platform_job_status_changed", _notify)
    return notified


def _waiting_job(*, claimed: bool = False) -> Job:
    return Job(
        id=uuid.uuid4(),
        type=JobType.segment,
        status=JobStatus.waiting,
        payload={},
        inference_job_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        document_part_id=uuid.uuid4(),
        callback_claimed_at=datetime.now(UTC) if claimed else None,
    )


def _segment_done_callback(job: Job) -> JobCallbackRequest:
    return JobCallbackRequest.model_validate(
        {
            "inference_job_id": str(job.inference_job_id),
            "product_job_id": str(job.id),
            "task": "segment",
            "status": "done",
            "output": {
                "kind": "segment",
                "data": {
                    "lines": [
                        {
                            "external_id": "l1",
                            "order": 0,
                            "baseline": {"type": "LineString", "coordinates": [[1, 1], [2, 1]]},
                            "points": [[1, 1], [2, 1], [2, 2], [1, 2]],
                        }
                    ]
                },
            },
        }
    )


def _failed_callback(job: Job, error: str) -> JobCallbackRequest:
    return JobCallbackRequest.model_validate(
        {
            "inference_job_id": str(job.inference_job_id),
            "product_job_id": str(job.id),
            "task": "segment",
            "status": "failed",
            "error": error,
        }
    )


def _stub_merge(monkeypatch: pytest.MonkeyPatch, sessions_seen: list) -> None:
    def _merge(session, _context, _callback) -> dict:
        sessions_seen.append(session)
        return {"lines_count": 1}

    monkeypatch.setattr(service, "_merge_into_session", _merge)


# --- Merge/finalize atomicity ---
# Tests the document writes and the job's terminal row update share one commit.
# Does not exercise the merge services themselves; those are stubbed here.


def test_merge_and_finalize_commit_once_in_one_session(monkeypatch: pytest.MonkeyPatch):
    job = _waiting_job(claimed=True)
    events: list[str] = []
    sessions = _use_sessions(monkeypatch, job, events)
    merged: list = []
    _stub_merge(monkeypatch, merged)

    assert (
        _merge_and_finalize(
            _merge_context(job, _segment_done_callback(job)), _segment_done_callback(job)
        )
        is True
    )

    # One session, one commit: the merge cannot become durable without the
    # ``done`` row update riding along.
    assert len(sessions) == 1
    assert merged == [sessions[0]]
    assert sessions[0].commits == 1
    assert job.status == JobStatus.done
    assert job.callback_claimed_at is None


def test_finalize_failure_leaves_the_merge_uncommitted(monkeypatch: pytest.MonkeyPatch):
    job = _waiting_job(claimed=True)
    events: list[str] = []
    sessions = _use_sessions(monkeypatch, job, events)
    _stub_merge(monkeypatch, [])

    def _boom(*_args, **_kwargs) -> None:
        raise RuntimeError("finalize exploded")

    monkeypatch.setattr(service, "_mark_done_from_callback_sync", _boom)

    with pytest.raises(RuntimeError, match="finalize exploded"):
        _merge_and_finalize(
            _merge_context(job, _segment_done_callback(job)), _segment_done_callback(job)
        )

    # Nothing committed, so the staged document writes die with the session.
    assert sessions[0].commits == 0


def test_terminal_job_is_not_merged(monkeypatch: pytest.MonkeyPatch):
    job = _waiting_job(claimed=True)
    events: list[str] = []
    sessions = _use_sessions(monkeypatch, job, events)
    merged: list = []
    _stub_merge(monkeypatch, merged)

    callback = _segment_done_callback(job)
    context = _merge_context(job, callback)
    # A cancel that committed between the claim and the merge: the FOR UPDATE
    # read inside the merge transaction sees it before any document write.
    job.status = JobStatus.cancelled

    assert _merge_and_finalize(context, callback) is False
    assert merged == []
    assert sessions[0].commits == 0


# --- Notification ordering ---
# Tests status announcements only leave after the transaction that produced them
# is durable. Does not test the SSE fan-out itself.


def test_done_is_announced_only_after_the_final_commit(monkeypatch: pytest.MonkeyPatch):
    job = _waiting_job()
    events: list[str] = []
    _use_sessions(monkeypatch, job, events)
    notified = _record_notifications(monkeypatch, events)
    _stub_merge(monkeypatch, [])

    assert _apply_callback_locked(_segment_done_callback(job)) is True

    # claim commit, merge+finalize commit, then the announcement.
    assert events == ["commit:0", "commit:1", "notify:done"]
    assert notified == [(job.id, JobStatus.done)]


def test_merge_failure_releases_the_claim_and_announces_failed(monkeypatch: pytest.MonkeyPatch):
    job = _waiting_job()
    events: list[str] = []
    sessions = _use_sessions(monkeypatch, job, events)
    notified = _record_notifications(monkeypatch, events)

    def _merge(_session, _context, _callback) -> dict:
        raise RuntimeError("merge exploded")

    monkeypatch.setattr(service, "_merge_into_session", _merge)

    with pytest.raises(RuntimeError, match="merge exploded"):
        _apply_callback_locked(_segment_done_callback(job))

    # The merge session never commits; the compensation is a separate write that
    # has to happen anyway because the claim committed on its own.
    assert sessions[1].commits == 0
    assert events == ["commit:0", "commit:2", "notify:failed"]
    assert notified == [(job.id, JobStatus.failed)]
    assert job.status == JobStatus.failed
    assert job.error == CALLBACK_PROCESSING_ERROR
    assert job.callback_claimed_at is None


# --- Inference failure reporting ---
# Tests the callback's error reaches job.error and the logs. Does not test the
# webhook transport, which lives in tests/nomikos/integration.


def test_callback_error_reaches_job_error(monkeypatch: pytest.MonkeyPatch):
    job = _waiting_job()
    events: list[str] = []
    _use_sessions(monkeypatch, job, events)
    _record_notifications(monkeypatch, events)

    assert _apply_callback_locked(_failed_callback(job, "weights not found in cache")) is True

    assert job.status == JobStatus.failed
    assert job.error == f"{INFERENCE_FAILURE_ERROR}: weights not found in cache"
    assert job.callback_claimed_at is None
    assert job.completed_at is not None
    # Terminal write first, announcement after.
    assert events == ["commit:0", "notify:failed"]


@pytest.mark.parametrize(
    "raw",
    [
        "download failed for https://hf.example/models/x?token=abcdefghijklmnop",
        "weights missing at /srv/inference/weights/model.mlmodel",
        "auth rejected: Authorization=Bearer sk-live-0123456789abcdefghij",
        "connect failed postgresql://user:hunter2@db.internal:5432/kalamos",
    ],
)
def test_callback_error_redacts_secret_shaped_text(raw: str):
    stored = _public_callback_error(_failed_callback(_waiting_job(), raw))

    assert stored.startswith(INFERENCE_FAILURE_ERROR)
    for leaked in ("hf.example", "abcdefghijklmnop", "hunter2", "sk-live", "/srv/inference"):
        assert leaked not in stored


def test_callback_error_is_bounded_but_keeps_a_stable_prefix():
    stored = _public_callback_error(_failed_callback(_waiting_job(), "boom " * 400))

    assert stored.startswith(f"{INFERENCE_FAILURE_ERROR}: boom boom")
    assert len(stored) <= len(INFERENCE_FAILURE_ERROR) + 2 + service._MAX_PUBLIC_ERROR_CHARS
    assert stored.endswith("…")


def test_callback_error_falls_back_when_nothing_survives_redaction():
    stored = _public_callback_error(_failed_callback(_waiting_job(), "/etc/inference/secrets.env"))

    assert stored == INFERENCE_FAILURE_ERROR


# --- Partial transcribe batches -------------------------------------------------
#
# The inference service isolates per-line failures instead of discarding the page,
# so a callback can now carry lines with ``error`` set and ``output`` absent. The
# merge loop must skip those rather than dereference ``None``.


class _FakeLine:
    def __init__(self, part_id: uuid.UUID, line_id: uuid.UUID) -> None:
        self.part_id = part_id
        self.id = line_id


class _FakePart:
    def __init__(self, part_id: uuid.UUID, document_id: uuid.UUID) -> None:
        self.id = part_id
        self.document_id = document_id


class _ScalarResult:
    def __init__(self, rows: list) -> None:
        self._rows = rows

    def scalars(self) -> _ScalarResult:
        return self

    def all(self) -> list:
        return self._rows

    def scalar_one_or_none(self):
        return self._rows[0] if self._rows else None


class _LineLookupSession:
    """Returns a line owned by ``part_id`` for every id the batched lookup asks for.

    Also answers the part-row lock, and records each statement as ``"lock"`` or
    ``"read"`` in the order it arrives, so a test can assert which came first.
    """

    def __init__(self, part_id: uuid.UUID, document_id: uuid.UUID | None = None) -> None:
        self._part_id = part_id
        self._document_id = document_id
        self.statements: list[str] = []

    def execute(self, statement, *_args, **_kwargs) -> _ScalarResult:
        locking = "FOR UPDATE" in str(statement)
        self.statements.append("lock" if locking else "read")
        if locking:
            return _ScalarResult([_FakePart(self._part_id, self._document_id)])
        ids: list[uuid.UUID] = []
        for value in statement.compile().params.values():
            for item in value if isinstance(value, (list, tuple)) else [value]:
                if isinstance(item, uuid.UUID):
                    ids.append(item)
        return _ScalarResult([_FakeLine(self._part_id, line_id) for line_id in ids])


def _batch(entries: list[tuple[int, str | None]]):
    from nomikos_inference.contracts.transcribe import (
        TRANSCRIBE_LINE_ERROR,
        TranscribeBatchLineResult,
        TranscribeBatchRunResponse,
        TranscribeRunResponse,
    )

    lines = []
    for index, text in entries:
        if text is None:
            lines.append(
                TranscribeBatchLineResult(
                    line_id=str(uuid.uuid4()), line_index=index, error=TRANSCRIBE_LINE_ERROR
                )
            )
            continue
        lines.append(
            TranscribeBatchLineResult(
                line_id=str(uuid.uuid4()),
                line_index=index,
                output=TranscribeRunResponse(
                    text=text,
                    confidence=1.0,
                    character_confidences=[
                        {"char": character, "confidence": 1.0} for character in text
                    ],
                ),
            )
        )
    return TranscribeBatchRunResponse(lines=lines)


def _transcribe_context(part_id: uuid.UUID):
    job = _waiting_job(claimed=True)
    job.type = JobType.transcribe
    job.document_id = uuid.uuid4()
    job.document_part_id = part_id
    callback = _segment_done_callback(job)
    return service._merge_context(job, callback)


def test_failed_lines_are_skipped_and_reported(monkeypatch: pytest.MonkeyPatch):
    part_id = uuid.uuid4()
    merged: list = []
    monkeypatch.setattr(
        service.TranscribeMergeService,
        "apply_sync",
        lambda _self, _session, **kwargs: (
            merged.append(kwargs["lines_with_output"]) or {"transcription_id": "t", "lines": []}
        ),
    )

    context = _transcribe_context(part_id)
    summary = service._apply_transcribe_merge_sync(
        _LineLookupSession(part_id, context.document_id),
        context=context,
        output=_batch([(0, "alpha"), (1, None), (2, "gamma")]),
    )

    # Only the two successful lines reach the merge, and the caller can still see
    # that the page was not fully transcribed.
    assert [text for _line, output in merged[0] for text in [output.text]] == ["alpha", "gamma"]
    assert summary["failed_line_indexes"] == [1]


def test_a_fully_failed_batch_is_not_merged_as_a_success(monkeypatch: pytest.MonkeyPatch):
    """The guard holds even if a batch of nothing but failures reaches the merge.

    ``TranscribeBatchRunResponse`` already refuses to validate an all-error body,
    so this is defence in depth rather than a reachable path today - built with
    ``model_construct`` to skip that validator. It exists because the alternative
    to raising is merging zero lines and reporting the page as transcribed.
    """
    from nomikos_inference.contracts.transcribe import (
        TRANSCRIBE_LINE_ERROR,
        TranscribeBatchLineResult,
        TranscribeBatchRunResponse,
    )

    part_id = uuid.uuid4()
    monkeypatch.setattr(
        service.TranscribeMergeService,
        "apply_sync",
        lambda *_args, **_kwargs: pytest.fail("a batch with no usable lines must not merge"),
    )
    all_failed = TranscribeBatchRunResponse.model_construct(
        lines=[
            TranscribeBatchLineResult(
                line_id=str(uuid.uuid4()), line_index=index, error=TRANSCRIBE_LINE_ERROR
            )
            for index in range(2)
        ]
    )

    context = _transcribe_context(part_id)
    with pytest.raises(service.TranscribeJobHandlerError):
        service._apply_transcribe_merge_sync(
            _LineLookupSession(part_id, context.document_id),
            context=context,
            output=all_failed,
        )


def test_the_part_is_locked_before_its_lines_are_read(monkeypatch: pytest.MonkeyPatch):
    """The lock has to precede the read, not sit inside the merge after it.

    A segment-health merge or deletion committing between an unlocked read and a
    later lock leaves this callback holding a ``Line`` whose row is gone, and the
    ``LineTranscription`` insert dies on the foreign key, failing a job whose model
    output was perfectly good. Locking first makes the two queue instead.
    """
    part_id = uuid.uuid4()
    context = _transcribe_context(part_id)
    session = _LineLookupSession(part_id, context.document_id)
    monkeypatch.setattr(
        service.TranscribeMergeService,
        "apply_sync",
        lambda _self, _session, **_kwargs: {"transcription_id": "t", "lines": []},
    )

    service._apply_transcribe_merge_sync(
        session,
        context=context,
        output=_batch([(0, "alpha")]),
    )

    assert session.statements[0] == "lock"
    assert "read" in session.statements


def test_a_transcribe_callback_for_a_missing_part_is_refused(monkeypatch: pytest.MonkeyPatch):
    """The lock doubles as the ownership check, so it must reject a foreign part."""
    part_id = uuid.uuid4()
    context = _transcribe_context(part_id)
    session = _LineLookupSession(part_id, uuid.uuid4())  # a different document
    monkeypatch.setattr(
        service.TranscribeMergeService,
        "apply_sync",
        lambda *_args, **_kwargs: pytest.fail("a part on another document must not merge"),
    )

    with pytest.raises(service.TranscribeJobHandlerError):
        service._apply_transcribe_merge_sync(
            session, context=context, output=_batch([(0, "alpha")])
        )


# --- Line scope and identity --------------------------------------------------
#
# Tests every reported line names a line of this job's part before anything is
# merged. Does not test the merge itself.


def _with_line_id(output, line_id: str | None):
    first = output.lines[0].model_copy(update={"line_id": line_id})
    return output.model_copy(update={"lines": [first, *output.lines[1:]]})


@pytest.mark.parametrize(
    ("line_id", "message"),
    [
        (None, "missing line_id"),
        ("not-a-uuid", "line_id is invalid"),
    ],
)
def test_a_line_without_a_usable_id_is_refused(line_id: str | None, message: str):
    part_id = uuid.uuid4()
    output = _with_line_id(_batch([(0, "a")]), line_id)
    context = _transcribe_context(part_id)

    with pytest.raises(service.TranscribeJobHandlerError, match=message):
        service._apply_transcribe_merge_sync(
            _LineLookupSession(part_id, context.document_id), context=context, output=output
        )


def test_a_line_outside_a_line_scoped_job_is_refused():
    part_id = uuid.uuid4()
    output = _batch([(0, "a")])
    context = _transcribe_context(part_id)
    scoped = service._MergeContext(
        **{**context.__dict__, "allowed_line_ids": frozenset({uuid.uuid4()})}
    )
    session = _LineLookupSession(part_id, context.document_id)

    with pytest.raises(service.TranscribeJobHandlerError, match="outside the job's scope"):
        service._apply_transcribe_merge_sync(session, context=scoped, output=output)
    # Refused before any line is read.
    assert session.statements == ["lock"]


class _ForeignLineSession(_LineLookupSession):
    """Locks the job's part but finds every requested line on another part."""

    def execute(self, statement, *args, **kwargs) -> _ScalarResult:
        result = super().execute(statement, *args, **kwargs)
        if self.statements[-1] == "read":
            for line in result.all():
                line.part_id = uuid.uuid4()
        return result


def test_a_line_of_another_part_is_refused():
    part_id = uuid.uuid4()
    output = _batch([(0, "a")])
    context = _transcribe_context(part_id)

    with pytest.raises(service.TranscribeJobHandlerError, match="Document line not found"):
        service._apply_transcribe_merge_sync(
            _ForeignLineSession(part_id, context.document_id), context=context, output=output
        )


def test_an_empty_id_list_reads_no_lines():
    assert service._load_lines(_LineLookupSession(uuid.uuid4()), []) == {}
