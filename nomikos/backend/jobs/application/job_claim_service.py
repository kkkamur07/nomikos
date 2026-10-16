"""Hand exactly one page of work to one inference agent.

This is the single new endpoint's worth of behaviour that ADR 0003 costs. It is
deliberately small, because everything around it already exists: the queue is the
platform's ``jobs`` table, completion and failure are ``JobCallbackRequest``,
abandonment is the stale sweep, and status delivery is the existing SSE path.

One page per claim
------------------
A batch is N claims, not one claim of N pages. Work is seconds-to-minutes, so a
page in flight is a small thing to lose: a slept laptop drops one page rather
than a document, and progress is free because each page completes as it goes.
That is also why there is no heartbeat endpoint - the lease covers the work with
margin, and a heartbeat would be a second liveness mechanism for a window nothing
runs past.

Why a claimed page becomes ``waiting`` and not ``running``
----------------------------------------------------------
``waiting`` already means exactly this: an inference host holds the job and the
platform is waiting for its callback. Claiming *is* the dispatch now - there is
no second queue to POST into - so the claim writes the state the old dispatch
wrote, mints the ``inference_job_id`` the callback contract matches on, and the
whole of ``JobCallbackService`` keeps working with no change at all. ``running``
is the platform worker's own status, for jobs it executes in-process.

Two agents never receive the same page
--------------------------------------
``FOR UPDATE SKIP LOCKED`` on one row, the same primitive
``claim_next_pending_job`` uses. The second claimer skips the locked row and
takes the next one, or finds nothing; it never waits on the first and never sees
the same id.

The page image arrives by signed link, and by nothing else
----------------------------------------------------------
The claim carries a link to the one page image, good for about a minute. An
authenticated ``GET /device/v1/jobs/{id}/image`` was rejected (ADR 0002): the
production API is serverless, so streaming manuscript scans through it costs
money for nothing, and it would put a route on the device credential that has to
independently re-derive job ownership. The signature *is* the authorization, and
it covers exactly one object key.

The response used to violate that anyway. ``request`` was an
``nomikos_inference.contracts.jobs.JobSubmitRequest`` - the body the platform once POSTed
into the second queue - so it carried ``image_bytes``, and the claim shipped the
whole scan base64-encoded at about 1.33x its stored size *next to* the link. No
agent has ever read that field. It is gone: the claim payload is now
``PageRunRequest``, which holds the instruction and not the page, and nothing on
this path reads an image out of storage at all.

The link's lifetime is not the lease's. The agent fetches once, immediately after
claiming, so the link only has to outlive one download on a bad connection;
tying it to the 600 second lease would keep a bearer token in a URL alive ten
times longer to buy nothing.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from urllib.parse import urljoin

from sqlalchemy import select

from backend.document.infrastructure.media_store import get_media_store
from backend.jobs.application.inference_dispatcher import (
    PageRunRequest,
    build_page_run,
)
from backend.jobs.infrastructure.job_claim_engine import AGENT_CLAIMED_JOB_TYPES
from backend.jobs.infrastructure.job_sql import AGENT_CLAIM_PREFIX
from backend.jobs.infrastructure.notifications import notify_platform_job_status_changed
from backend.jobs.infrastructure.orm_models import Job, JobStatus, JobType
from backend.ml.application.agent_credentials import InferenceAgent
from backend.ml.domain.execution import ExecutionTarget
from infrastructure.db import sync_system_session

logger = logging.getLogger(__name__)

UNBUILDABLE_PAYLOAD_ERROR = "This page could not be prepared for inference"


def agent_claim_owner(device_id: uuid.UUID) -> str:
    """The value written to ``jobs.claimed_by`` for an agent-held page.

    Prefixed so "which agent holds this page" is answerable from the row alone -
    which is what lets the callback authorise a device without a second table,
    and what lets the stale sweep tell a leased page from a dispatched one
    without joining anything.

    The collision with ``worker_identity()`` is prevented on the *other* side:
    that value is namespaced ``worker:host/pid``, so it cannot begin with this
    prefix whatever the machine is called. It used to be a bare ``host:pid``,
    which meant a machine whose hostname happened to be the prefix's own word
    produced a claim id every sweep read as agent-held.
    """
    return f"{AGENT_CLAIM_PREFIX}{device_id}"


@dataclass(frozen=True)
class ClaimedPage:
    """One page of work, and everything the agent needs to run and report it."""

    product_job_id: uuid.UUID
    inference_job_id: uuid.UUID
    job_type: JobType
    execution_target: ExecutionTarget
    lease_expires_at: datetime
    request: PageRunRequest
    #: Signed link to the one page image, and the moment it stops working. Two
    #: separate fields rather than one, because an agent has to be able to tell
    #: whether its link is still worth using without parsing a URL.
    page_image_url: str
    page_image_expires_at: datetime


def _claimable_job_query(agent: InferenceAgent):
    query = (
        select(Job)
        .where(Job.status == JobStatus.pending)
        # The job types the platform worker deliberately leaves alone: with the
        # second queue gone it has no way to run them, so they sit pending for an
        # agent. The two predicates are complements, by construction.
        .where(Job.type.in_(AGENT_CLAIMED_JOB_TYPES))
        # **Execution target** is fixed at submission and is not a preference: a
        # ``cloud`` job is never handed to a laptop, and a ``local`` job is never
        # handed to a hosted worker, whatever either one asks for.
        .where(Job.execution_target == agent.execution_target)
        .order_by(Job.created_at, Job.id)
    )
    if agent.claims_own_account_only:
        # The entire authorization scope of a device credential, applied to the
        # queue: one researcher's laptop sees one researcher's pages.
        query = query.where(Job.user_id == agent.user_id)
    return query.with_for_update(skip_locked=True).limit(1)


def claim_one_page(
    agent: InferenceAgent,
    *,
    lease_seconds: int,
    page_image_ttl_seconds: int,
    base_url: str,
) -> ClaimedPage | None:
    """Take at most one pending page for *agent*, or return ``None``.

    Synchronous on purpose. It is called through ``asyncio.to_thread`` from a
    route that holds no request-scoped session, so the connection is checked out
    for the length of one short transaction and returned before the caller waits
    again. A long poll is a sequence of these, not one held connection.

    *base_url* only resolves a relative link the local storage backend produces;
    a Supabase link is already absolute and passes through untouched.
    """
    now = datetime.now(UTC)
    inference_job_id = uuid.uuid4()
    with sync_system_session() as session:
        job = session.execute(_claimable_job_query(agent)).scalar_one_or_none()
        if job is None:
            return None
        job.status = JobStatus.waiting
        job.inference_job_id = inference_job_id
        job.claimed_by = agent_claim_owner(agent.device_id)
        job.started_at = now
        job.heartbeat_at = now
        job.callback_claimed_at = None
        job.updated_at = now
        session.commit()
        job_id = job.id
        job_type = job.type
        execution_target = job.execution_target
        # Detached but usable: ``expire_on_commit=False``. The build below still
        # runs outside this transaction - it reads the part and its lines - so a
        # queue row is never locked across it.
        detached_job = job

    notify_platform_job_status_changed(job_id, JobStatus.waiting)

    page_image_expires_at = now + timedelta(seconds=page_image_ttl_seconds)
    try:
        run = build_page_run(detached_job)
        page_image_url = urljoin(
            base_url,
            get_media_store().signed_object_url(run.image_key, expires_at=page_image_expires_at),
        )
    except Exception:
        # The page is already claimed, so leaving it would park a job on an agent
        # that was never given anything to run, until the sweep failed it minutes
        # later. Fail it now, with the same allowlisted text rule as everywhere
        # else - the exception detail is logged, never stored.
        logger.exception("could not build the claim payload for job %s", job_id)
        _fail_unbuildable_page(job_id, agent)
        raise

    logger.info(
        "job_claimed job_id=%s device_id=%s target=%s service_worker=%s inference_job_id=%s",
        job_id,
        agent.device_id,
        execution_target.value,
        agent.is_service_worker,
        inference_job_id,
    )
    return ClaimedPage(
        product_job_id=job_id,
        inference_job_id=inference_job_id,
        job_type=job_type,
        execution_target=execution_target,
        lease_expires_at=now + timedelta(seconds=lease_seconds),
        request=run.request,
        page_image_url=page_image_url,
        page_image_expires_at=page_image_expires_at,
    )


def _fail_unbuildable_page(job_id: uuid.UUID, agent: InferenceAgent) -> None:
    from backend.jobs.infrastructure.job_terminal import mark_job_failed

    mark_job_failed(
        job_id, UNBUILDABLE_PAYLOAD_ERROR, claimed_by=agent_claim_owner(agent.device_id)
    )


def job_is_held_by(job: Job, agent: InferenceAgent) -> bool:
    """Whether *agent* is the one holding this page right now.

    Used by the callback path. It is a comparison against ``claimed_by`` rather
    than a re-derivation of ownership, because the claim is what the agent was
    given and the claim is what it may report on - a device that merely *could*
    have claimed a page has no business completing it.
    """
    return job.claimed_by == agent_claim_owner(agent.device_id)
