"""Apply inference job completion callbacks to Product jobs."""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select

from backend.core.exceptions import ConflictError, NotFoundError
from backend.document.application.segment_merge_service import SegmentMergeService
from backend.document.application.transcribe_merge_service import (
    TranscribeJobHandlerError,
    TranscribeMergeService,
)
from backend.document.infrastructure.orm_models import Line
from backend.jobs.infrastructure.notifications import notify_platform_job_status_changed
from backend.jobs.infrastructure.orm_models import Job, JobStatus, JobType
from backend.ml.application.segment_mapping import to_canonical_segment
from infrastructure.db import sync_system_session
from nomikos_inference.contracts.common import InferenceJobStatus
from nomikos_inference.contracts.common import InferenceTask as WireInferenceTask
from nomikos_inference.contracts.jobs import JobCallbackRequest
from nomikos_inference.contracts.segment import SegmentRunResponse
from nomikos_inference.contracts.transcribe import TranscribeBatchRunResponse

logger = logging.getLogger(__name__)

_TERMINAL_STATUSES = frozenset({JobStatus.done, JobStatus.failed, JobStatus.cancelled})

INFERENCE_FAILURE_ERROR = "Inference job failed"
CALLBACK_PROCESSING_ERROR = "Callback processing failed"

# ``job.error`` is Text (unbounded), so this cap is not a column constraint. It
# exists because the string is rendered whole in a toast and a traceback-length
# message there is unreadable. The ``INFERENCE_FAILURE_ERROR`` prefix stays
# stable so callers can still recognise the class of failure.
_MAX_PUBLIC_ERROR_CHARS = 200

# Ordered: URLs are swallowed whole before the path rule can nibble at their
# path component, and assignments before the opaque-token rule, so the redacted
# text names what was removed.
_ERROR_REDACTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b[a-z][a-z0-9+.\-]*://\S+", re.IGNORECASE), "<url>"),
    (
        re.compile(
            r"\b(?:token|secret|password|passwd|api[_-]?key|access[_-]?key|authorization)\b"
            r"\s*[:=]\s*\S+",
            re.IGNORECASE,
        ),
        "<redacted>",
    ),
    (re.compile(r"\bbearer\s+\S+", re.IGNORECASE), "<redacted>"),
    (re.compile(r"(?<![\w/])/(?:[\w.\-]+/)+[\w.\-]*"), "<path>"),
    (re.compile(r"\b[A-Za-z0-9_\-]{24,}\b"), "<redacted>"),
)


@dataclass(frozen=True)
class _MergeContext:
    job_id: uuid.UUID
    job_type: JobType
    document_id: uuid.UUID | None
    document_part_id: uuid.UUID | None
    inference_job_id: uuid.UUID
    #: When a "transcribe selected lines" job restricts itself to a subset via
    #: ``payload["line_ids"]``, the callback must not merge results outside that
    #: subset. ``None`` means the whole page was in scope.
    allowed_line_ids: frozenset[uuid.UUID] | None = None


def _job_type_for_task(task: WireInferenceTask) -> JobType:
    return JobType(task.value)


def _allowed_line_ids(job: Job) -> frozenset[uuid.UUID] | None:
    raw = (job.payload or {}).get("line_ids")
    if not raw:
        return None
    try:
        return frozenset(uuid.UUID(str(line_id)) for line_id in raw)
    except (ValueError, TypeError):
        # A malformed restriction is our own bad data, not the agent's; fall back
        # to the part-scope check rather than 500 on every callback for this job.
        return None


def _segment_output(callback: JobCallbackRequest) -> SegmentRunResponse:
    if callback.output is None or callback.output.kind != "segment":
        raise ValueError("Segment callback missing structured output")
    return callback.output.data


def _transcribe_output(callback: JobCallbackRequest) -> TranscribeBatchRunResponse:
    if callback.output is None or callback.output.kind != "transcribe":
        raise TranscribeJobHandlerError("Transcribe callback missing structured output")
    data = callback.output.data
    if not isinstance(data, TranscribeBatchRunResponse):
        raise TranscribeJobHandlerError("Transcribe callback missing batched line results")
    return data


def _merge_context(job: Job, callback: JobCallbackRequest) -> _MergeContext:
    return _MergeContext(
        job_id=job.id,
        job_type=job.type,
        document_id=job.document_id,
        document_part_id=job.document_part_id,
        inference_job_id=callback.inference_job_id,
        allowed_line_ids=_allowed_line_ids(job),
    )


def _apply_segment_merge(session, context: _MergeContext, callback: JobCallbackRequest) -> dict:
    if context.document_part_id is None:
        raise ValueError("Segment job is missing its target document part")
    canonical = to_canonical_segment(_segment_output(callback))
    summary = SegmentMergeService().apply_sync(
        session,
        part_id=context.document_part_id,
        canonical_segment=canonical,
        job_id=context.job_id,
        commit=False,
    )
    return {
        "blocks_count": summary.blocks_count,
        "lines_count": summary.lines_count,
        "added_lines": summary.added_lines,
        "pruned_lines": summary.pruned_lines,
        "preserved_manual_lines": summary.preserved_manual_lines,
        "preserved_transcribed_lines": summary.preserved_transcribed_lines,
        "skipped_covered_lines": summary.skipped_covered_lines,
    }


def _apply_transcribe_merge_sync(
    session,
    *,
    context: _MergeContext,
    output: TranscribeBatchRunResponse,
) -> dict:
    if context.document_id is None or context.document_part_id is None:
        raise TranscribeJobHandlerError("Transcribe job is missing its target document part")

    # Lock the part before reading its lines, not inside ``apply_sync`` after. A
    # segment-health merge or deletion committing in that gap leaves the loop
    # below holding a ``Line`` whose row is gone, and the insert dies on the
    # foreign key. Locking first makes the two operations queue: whichever wins
    # commits, and the loser reads the state the winner left, so a line that
    # really did disappear surfaces as "Document line not found" instead.
    TranscribeMergeService.lock_part_sync(
        session,
        document_id=context.document_id,
        part_id=context.document_part_id,
    )

    # First pass: validate ids and enforce the job's own line scope, collecting
    # the ids so every line is fetched in one query instead of one SELECT per
    # line (a 50-line page was 50 sequential round trips under the locked job row).
    parsed: list[tuple[object, uuid.UUID]] = []
    for result in sorted(output.lines, key=lambda item: item.line_index):
        if result.line_id is None:
            raise TranscribeJobHandlerError("Transcribe callback line is missing line_id")
        try:
            line_id = uuid.UUID(result.line_id)
        except ValueError as exc:
            raise TranscribeJobHandlerError("Transcribe callback line_id is invalid") from exc
        # A compromised or buggy agent holding this job could report lines the
        # job never selected. The part check below stops cross-part writes; this
        # stops cross-line writes within the part when the job was line-scoped.
        if context.allowed_line_ids is not None and line_id not in context.allowed_line_ids:
            raise TranscribeJobHandlerError("Transcribe callback line is outside the job's scope")
        parsed.append((result, line_id))

    lines_by_id = (
        {
            line.id: line
            for line in session.execute(
                select(Line).where(Line.id.in_([line_id for _, line_id in parsed]))
            )
            .scalars()
            .all()
        }
        if parsed
        else {}
    )

    lines_with_output = []
    failed_line_indexes: list[int] = []
    for result, line_id in parsed:
        line = lines_by_id.get(line_id)
        if line is None or line.part_id != context.document_part_id:
            raise TranscribeJobHandlerError("Document line not found")
        # A batch can be a partial success: the inference service isolates
        # per-line failures instead of discarding the whole page, and sends
        # those lines with ``error`` set and ``output`` absent. Merging one
        # would pass ``None`` where the merge service dereferences ``.text``.
        # Skip them, but count them, since a page that silently transcribed
        # 12 of 40 lines and reported plain success would be worse than a
        # total failure.
        if result.output is None:
            failed_line_indexes.append(result.line_index)
            continue
        lines_with_output.append((line, result.output))

    if not lines_with_output:
        raise TranscribeJobHandlerError("Transcribe callback contained no successful lines")

    summary = TranscribeMergeService().apply_sync(
        session,
        document_id=context.document_id,
        part_id=context.document_part_id,
        job_id=context.job_id,
        lines_with_output=lines_with_output,
        commit=False,
    )
    if failed_line_indexes:
        logger.warning(
            "transcribe_callback_partial job_id=%s inference_job_id=%s failed_lines=%s",
            context.job_id,
            context.inference_job_id,
            failed_line_indexes,
        )
        summary = {**summary, "failed_line_indexes": failed_line_indexes}
    return summary


def _merge_into_session(session, context: _MergeContext, callback: JobCallbackRequest) -> dict:
    """Stage the document writes for this callback. Never commits; see ``_merge_and_finalize``."""
    if context.job_type == JobType.segment:
        return _apply_segment_merge(session, context, callback)
    if context.job_type == JobType.transcribe:
        return _apply_transcribe_merge_sync(
            session,
            context=context,
            output=_transcribe_output(callback),
        )
    raise ConflictError(
        f"job {context.job_id} type {context.job_type.value} cannot receive inference callbacks"
    )


def _public_callback_error(callback: JobCallbackRequest) -> str:
    """Client-visible failure text derived from the inference service's message.

    ``job.error`` is served verbatim by ``JobResponse`` and rendered in a toast,
    while ``callback.error`` is ``str(exc)`` raised anywhere inside ``run_job``:
    weights paths, registry download URLs, driver chatter. So the same rule as
    ``worker._public_job_error`` applies - nothing that could carry a credential
    or a server-side path reaches the client - but blanket-dropping the message
    is what made production failures unattributable. Redact the shapes that can
    hold a secret and keep the human sentence, which is the only part a user can
    act on. The unredacted text is logged, never stored.
    """
    detail = " ".join((callback.error or "").split())
    for pattern, placeholder in _ERROR_REDACTIONS:
        detail = pattern.sub(placeholder, detail)
    detail = detail.strip()
    # A message that redacted down to nothing but placeholders carries no signal.
    if not detail or not re.search(r"[^\W\d_]", re.sub(r"<[a-z]+>", "", detail)):
        return INFERENCE_FAILURE_ERROR
    if len(detail) > _MAX_PUBLIC_ERROR_CHARS:
        detail = detail[: _MAX_PUBLIC_ERROR_CHARS - 1].rstrip() + "…"
    return f"{INFERENCE_FAILURE_ERROR}: {detail}"


def _mark_failed_from_callback_sync(job: Job, callback: JobCallbackRequest) -> None:
    # The callback message is the only diagnostic the inference service sends;
    # log it whole here because the stored copy is redacted and truncated.
    logger.warning(
        "inference callback reported failure for job %s (inference_job_id=%s): %s",
        job.id,
        callback.inference_job_id,
        callback.error or "<no error reported>",
    )
    now = datetime.now(UTC)
    job.status = JobStatus.failed
    job.error = _public_callback_error(callback)
    job.callback_claimed_at = None
    job.completed_at = now
    job.updated_at = now


def _assert_callback_matches_job(job: Job, callback: JobCallbackRequest) -> None:
    expected_type = _job_type_for_task(callback.task)
    if job.type != expected_type:
        raise ConflictError(
            f"job {job.id} type {job.type.value} does not match callback task {callback.task.value}"
        )

    if job.inference_job_id is None or callback.inference_job_id != job.inference_job_id:
        raise ConflictError(f"job {job.id} does not recognize callback inference_job_id")


def _mark_done_from_callback_sync(job: Job, result: dict) -> None:
    now = datetime.now(UTC)
    job.status = JobStatus.done
    job.result = result
    job.error = None
    # The agent's claim was not abandoned, it was honoured. Clearing the counter
    # keeps the two success paths (this one and ``mark_job_done``) writing the
    # same row, so ``jobs.claim_attempts`` means "abandoned since the last
    # success" whichever of them finished the job.
    job.claim_attempts = 0
    job.callback_claimed_at = None
    job.completed_at = now
    job.updated_at = now


def _release_claim_as_failed(job_id: uuid.UUID, error: str) -> bool:
    """Fail a job whose merge transaction rolled back. Returns whether it moved.

    Merge and finalize now share a transaction, so a failure there leaves no
    document rows behind, but the claim from ``_validate_callback`` committed
    in its own transaction and is still on the row. Without this compensating
    write the job sits ``waiting`` and uncancellable until the stale-claim
    sweep gets to it, minutes of a user staring at a job that's already dead.
    The guard keeps it a no-op if anything else already moved the row.
    """
    now = datetime.now(UTC)
    with sync_system_session() as session:
        job = session.execute(
            select(Job).where(Job.id == job_id).with_for_update()
        ).scalar_one_or_none()
        if job is None or job.status != JobStatus.waiting or job.callback_claimed_at is None:
            return False
        job.status = JobStatus.failed
        job.error = error
        # Clearing the claim keeps failed rows consistent with every other
        # terminal write; a lingering claim only confuses the sweeps.
        job.callback_claimed_at = None
        job.completed_at = now
        job.updated_at = now
        session.commit()
    return True


def _validate_callback(callback: JobCallbackRequest) -> tuple[bool, _MergeContext | None]:
    """Atomically claim one waiting callback before its merge can begin.

    Deliberately its own transaction, separate from merge + finalize. The claim
    has to be durable before the merge starts: that is what makes a duplicate
    callback in another worker return immediately instead of queueing behind the
    row lock for the length of a thousand-line merge, and what lets
    ``clear_stale_callback_claims`` see a merge that is in flight. Folding it
    into the merge transaction would hide the claim until the merge commits and
    turn concurrent deliveries into lock waits bounded only by statement_timeout.
    """
    notify_failed = False
    context: _MergeContext | None = None
    with sync_system_session() as session:
        job = session.execute(
            select(Job).where(Job.id == callback.product_job_id).with_for_update()
        ).scalar_one_or_none()
        if job is None:
            raise NotFoundError(f"job {callback.product_job_id} not found")
        _assert_callback_matches_job(job, callback)
        if job.status in _TERMINAL_STATUSES:
            return False, None
        if job.status != JobStatus.waiting:
            raise ConflictError(f"job {job.id} is not waiting for an inference callback")
        if job.callback_claimed_at is not None:
            return False, None

        if callback.status == InferenceJobStatus.failed:
            _mark_failed_from_callback_sync(job, callback)
            notify_failed = True
        else:
            context = _merge_context(job, callback)
            job.callback_claimed_at = datetime.now(UTC)
            job.updated_at = job.callback_claimed_at
        session.commit()

    # Announce outside the transaction, only once the commit is durable: an SSE
    # subscriber that reacts to a status a rollback then erased has been lied to.
    if notify_failed:
        notify_platform_job_status_changed(callback.product_job_id, JobStatus.failed)
    return True, context


def _merge_and_finalize(context: _MergeContext, callback: JobCallbackRequest) -> bool:
    """Merge the document writes and complete the job in a single transaction.

    The merge services are called with ``commit=False`` so the document rows,
    the ``done`` status, and the cleared claim land in one commit. Committing
    them separately meant a crash (or any raise) between the two commits left
    merged lines under a job still marked ``waiting``, which the compensation
    then failed: a failed job sitting on top of successfully merged content,
    and a retry that merged it a second time.
    """
    with sync_system_session() as session:
        # FOR UPDATE taken before the first document write and held to commit.
        # This subsumes the old pre-merge terminal check, which read its own
        # snapshot that could go stale before the merge even started: a cancel
        # that beat us to the row is visible here and we return without merging,
        # and one arriving later blocks until this transaction resolves.
        job = session.execute(
            select(Job).where(Job.id == context.job_id).with_for_update()
        ).scalar_one_or_none()
        if job is None:
            raise NotFoundError(f"job {context.job_id} not found")
        _assert_callback_matches_job(job, callback)
        if job.status in _TERMINAL_STATUSES:
            return False
        if job.status != JobStatus.waiting or job.callback_claimed_at is None:
            raise ConflictError(f"job {job.id} is not processing this inference callback")

        result = _merge_into_session(session, context, callback)
        _mark_done_from_callback_sync(job, result)
        session.commit()
    return True


def _apply_callback_locked(callback: JobCallbackRequest) -> bool:
    applied, context = _validate_callback(callback)
    if not applied or context is None:
        return applied

    try:
        finalized = _merge_and_finalize(context, callback)
    except Exception:
        logger.exception(
            "inference callback merge failed for job %s (inference_job_id=%s)",
            context.job_id,
            context.inference_job_id,
        )
        if _release_claim_as_failed(context.job_id, CALLBACK_PROCESSING_ERROR):
            notify_platform_job_status_changed(context.job_id, JobStatus.failed)
        raise

    if finalized:
        notify_platform_job_status_changed(context.job_id, JobStatus.done)
    return finalized


class JobCallbackService:
    async def apply_callback(self, callback: JobCallbackRequest) -> bool:
        """Apply callback. Returns False when the Product job was already terminal."""
        # Everything below the seam is sync: ``sync_system_session`` round-trips
        # plus a document merge that can touch thousands of lines. Running it on
        # the event loop stalls every other request on this worker. Offload the
        # whole claim -> merge -> finalize sequence as one unit, matching
        # ``cancel_job_async``; splitting it would let a cancel interleave between
        # the claim and the merge and change the transaction semantics.
        return await asyncio.to_thread(_apply_callback_locked, callback)
