"""Wire shapes for the claim endpoint.

The claimed page carries an instruction, not a payload. It used to carry a whole
``nomikos_inference.contracts.jobs.JobSubmitRequest`` - the body the platform once POSTed
into a second inference queue - so that local and cloud agents took literally the
same object. Since ADR 0003 there is no second queue and no POST: the agent
claims from the platform's own table and fetches the scan from the signed link in
this same response. What reusing the submit contract bought was one field,
``image_bytes``, which no agent has ever read and which base64-encoded a whole
manuscript page into every claim at about 1.33x its stored size.

So the claim has its own shape, holding exactly the fields the agent reads. The
submit contract stays where it belongs, describing a submission.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from backend.jobs.infrastructure.orm_models import JobType
from backend.ml.api.agent_version import AgentVersionNotice
from backend.ml.domain.execution import ExecutionTarget
from nomikos_inference.contracts.common import InferenceTask


class JobClaimRequest(BaseModel):
    """How long the agent is willing to wait for work.

    A laptop long-polls, because an idle researcher's page should start within a
    second of being submitted. A hosted worker sends ``0`` and short-polls: it is
    never idle for long, it does not need the latency, and every second it waits
    is a request-handler slot it is holding on a serverless host.

    There is deliberately no "how many pages" field. One claim is one page.
    """

    wait_seconds: int = Field(
        default=0,
        ge=0,
        description=(
            "Seconds to wait for a page before returning empty. Clamped server-side to "
            "DEVICE_CLAIM_MAX_WAIT_SECONDS. 0 is a short poll."
        ),
    )


class ClaimedPageRequest(BaseModel):
    """What to run on the claimed page.

    There is no image field, deliberately: the page arrives by signed link
    (``ClaimedPageResponse.page_image_url``), which is the only mechanism, not
    one of two. An agent that reads these four fields and fetches that link has
    everything it needs.
    """

    task: InferenceTask
    registry_model_id: str = Field(min_length=1)
    registry_tag: str = Field(default="stable", min_length=1)
    product_job_id: UUID
    params: dict[str, object] = Field(default_factory=dict)


class ClaimedPageResponse(BaseModel):
    """One page of work, and the short-lived link to its image.

    The link is not authenticated by anything the agent presents: the signature
    in it *is* the authorization, and it reaches exactly one object (ADR 0002).
    An authenticated ``GET /device/v1/jobs/{id}/image`` was rejected because the
    production API is serverless - streaming manuscript scans through it costs
    money for nothing - and because it would put a route on the device credential
    that must independently re-derive job ownership. The same reasoning is why
    ``request`` below carries no image: this response used to stream the scan
    through the API *as well as* hand out the link.
    """

    product_job_id: UUID
    inference_job_id: UUID
    job_type: JobType
    execution_target: ExecutionTarget
    lease_expires_at: datetime
    request: ClaimedPageRequest
    page_image_url: str = Field(
        description=(
            "Signed link to this page's image, and to nothing else. Carries its own "
            "authorization, so it is fetched with no device credential attached."
        )
    )
    page_image_expires_at: datetime = Field(
        description=(
            "When the link above stops working - about a minute out, and deliberately "
            "not the lease. The agent fetches once, right after claiming."
        )
    )


class JobClaimResponse(BaseModel):
    """Always 200, with or without a page.

    An empty queue is the normal state of a healthy platform, not an error. A 404
    or a 204 here would make an agent's logs unreadable and would tempt a client
    into treating "nothing to do" as a failure to back off from.
    """

    page: ClaimedPageResponse | None = None
    poll_after_seconds: float = Field(
        description="What the agent should wait before asking again. 0 when a page was handed over."
    )
    lease_seconds: int = Field(
        description="How long a claimed page stays with the agent. Read from the platform, not compiled in."
    )
    server_time: datetime = Field(
        description="So an agent with a wrong clock can still reason about its lease."
    )
    agent: AgentVersionNotice = Field(
        description=(
            "What the platform makes of this agent's version. Present on every claim "
            "response, page or no page, so an idle agent still learns it is behind. An "
            "agent below the floor never sees this - it gets a 426 instead."
        )
    )
