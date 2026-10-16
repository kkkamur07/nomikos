"""Internal inference routes: job completion callbacks from an inference agent.

Completion and failure are **not** new endpoints. Since ADR 0003 there is one
inference agent, local or hosted, and it reports the way the inference service
always did: one ``JobCallbackRequest`` to this route, validated by the same
contract, applied by the same ``JobCallbackService``, announced on the same SSE
channel. The claim endpoint is the only thing that layer adds.

What did have to change is *who may present a callback*. The webhook secret is a
platform-side credential; a researcher's laptop does not have one and must not be
given one, since a shared secret handed to every agent would let any of them
complete any job on the platform. So this route also accepts the two agent
credentials, and narrows them to the page that agent is actually holding:
``jobs.claimed_by`` names the claiming device, and nothing else is accepted. A
device that merely *could* have claimed a page cannot report on it.

The webhook branch is narrowed by the same column, from the other side
--------------------------------------------------------------------
Presenting ``INFERENCE_WEBHOOK_SECRET`` used to be the end of the check: the
branch authenticated the caller and then applied the callback to whatever job id
the body named. But the only jobs that can *receive* a callback are ``waiting``
ones with a matching ``inference_job_id``, and since ADR 0003 the only writer of
that pair is ``claim_one_page`` - so every job the webhook branch could reach was
a page leased to somebody's laptop. A holder of the platform secret could
complete or fail work an agent was in the middle of.

So the branch now refuses an agent-held page. It is the exact complement of the
narrowing above, read off the same column: an agent credential may report on the
page it holds and no other, and the platform secret may report on pages no agent
holds. Today that second set is empty, which is why this reads as a guard rather
than a feature - the platform's own HTTP hop into an inference service is gone,
and nothing writes a ``waiting`` row without an agent claim any more. It is
kept, and guarded, rather than deleted, because the secret it authenticates is
still required in production by ``backend.core.settings.ml`` and still documented
as the platform-side callback credential; deleting only the branch would leave a
mandatory secret that authenticates nothing.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.exceptions import InvalidCredentialsError
from backend.jobs.api.dependencies import require_inference_webhook_secret
from backend.jobs.application.job_callback_service import JobCallbackService
from backend.jobs.application.job_claim_service import job_is_held_by
from backend.jobs.infrastructure.job_claim_engine import AGENT_CLAIM_PREFIX
from backend.jobs.infrastructure.orm_models import Job
from backend.ml.application.agent_credentials import (
    SERVICE_TOKEN_HEADER,
    WORKER_NAME_HEADER,
    resolve_inference_agent,
)
from backend.ml.application.device_auth import DEVICE_TOKEN_HEADER
from infrastructure.db import get_db
from nomikos_inference.contracts.jobs import JobCallbackRequest
from nomikos_inference.contracts.webhooks import INFERENCE_WEBHOOK_SECRET_HEADER

router = APIRouter(prefix="/internal/inference", tags=["internal-inference"])


def _callback_service(db: AsyncSession = Depends(get_db)) -> JobCallbackService:
    return JobCallbackService(db)


async def _authorize_agent_callback(
    db: AsyncSession,
    body: JobCallbackRequest,
    *,
    device_token: str | None,
    service_token: str | None,
    worker_name: str | None,
) -> None:
    try:
        agent = await resolve_inference_agent(
            db,
            device_token=device_token,
            service_token=service_token,
            worker_name=worker_name,
        )
    except InvalidCredentialsError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid agent credential",
            headers={"WWW-Authenticate": DEVICE_TOKEN_HEADER},
        ) from None

    job = (await db.execute(select(Job).where(Job.id == body.product_job_id))).scalar_one_or_none()
    # An unknown job is 403 here rather than 404: an agent that does not hold a
    # page must not be able to probe which job ids exist.
    if job is None or not job_is_held_by(job, agent):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This agent is not holding that job",
        )


async def _authorize_webhook_callback(db: AsyncSession, body: JobCallbackRequest) -> None:
    """Keep the platform-side secret away from a page an agent is holding.

    The webhook secret says "the platform is calling itself back". It says
    nothing about *which* job, and a page under lease belongs to the agent
    holding it - completing or failing it from here would discard the run that
    laptop is in the middle of, and the agent's own callback would then be
    rejected as a duplicate.

    Same 403 and same silence as the agent path: an unknown job is not
    distinguished from a leased one, so neither credential can enumerate job ids.
    """
    claimed_by = (
        await db.execute(select(Job.claimed_by).where(Job.id == body.product_job_id))
    ).scalar_one_or_none()
    if claimed_by is not None and claimed_by.startswith(AGENT_CLAIM_PREFIX):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This job is held by an inference agent",
        )


@router.post(
    "/job-complete",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
async def complete_inference_job(
    body: JobCallbackRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    service: Annotated[JobCallbackService, Depends(_callback_service)],
    x_inference_webhook_secret: Annotated[
        str | None, Header(alias=INFERENCE_WEBHOOK_SECRET_HEADER)
    ] = None,
    x_nomikos_device_token: Annotated[str | None, Header(alias=DEVICE_TOKEN_HEADER)] = None,
    x_nomikos_service_token: Annotated[str | None, Header(alias=SERVICE_TOKEN_HEADER)] = None,
    x_nomikos_worker_name: Annotated[str | None, Header(alias=WORKER_NAME_HEADER)] = None,
) -> Response:
    if x_nomikos_device_token is not None or x_nomikos_service_token is not None:
        await _authorize_agent_callback(
            db,
            body,
            device_token=x_nomikos_device_token,
            service_token=x_nomikos_service_token,
            worker_name=x_nomikos_worker_name,
        )
    else:
        # No agent credential presented means this is the platform-side webhook
        # path, with the same 503 / 401 / 403 outcomes it always had - and then
        # the job-level check it never had. See the module docstring.
        require_inference_webhook_secret(x_inference_webhook_secret)
        await _authorize_webhook_callback(db, body)
    await service.apply_callback(body)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
