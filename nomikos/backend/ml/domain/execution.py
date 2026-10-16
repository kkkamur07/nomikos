"""Execution target and capacity - the two words submission is decided in.

**Execution target** is the **inference host** one job runs on, ``local`` or
``cloud``. It is fixed when the job is submitted and never changed afterwards,
which is the property every other decision in ADR 0002 rests on: a job never
silently moves host, and a job never waits on a machine that is not there.

**Capacity** is whether an inference host currently has a machine able to take
work. A researcher's laptop and a hosted worker are the same kind of thing here,
so there is one question with one answer rather than two - the answer comes from
:mod:`backend.ml.application.capacity_service`, which reads recent device
activity.

Three inputs decide one target, and they are deliberately different in kind:

* **preference** - the account-level setting, "use my computer when it is
  available". There is no per-job toggle: a researcher cannot know which host is
  faster for a given page, so asking at every action is a decision without a
  basis.
* **eligibility** - what the chosen model permits. **Host eligibility**
  *constrains* which targets a job may choose; it does not choose one.
* **capacity** - which hosts can actually take work right now.

There is no cloud-fallback timer, no hold window, and no sweeper for unclaimed
local jobs. The decision is made once, before the job exists, and when neither
host can take the work submission refuses rather than creating a job nobody will
claim.

``local_only`` is deliberately absent. Its justification - manuscripts never
leave the machine - was never true, since page images already live in the
platform's media store and the browser downloads them from there; and it was the
one mode that could leave a job with no terminal outcome. See ADR 0002.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from backend.core.exceptions import ConflictError


class ExecutionTarget(StrEnum):
    """The **inference host** a single job runs on.

    Also the discriminator on a paired device, because a hosted worker is a
    device like any other (ADR 0003) - a device *is* one of these two hosts.
    """

    local = "local"
    cloud = "cloud"


ALL_EXECUTION_TARGETS = frozenset(ExecutionTarget)

NO_CAPACITY_MESSAGE = (
    "No inference host is available right now: your computer is not running the "
    "nomikos agent, and no cloud worker is online. Start the agent and try again."
)

_NO_ELIGIBLE_HOST_MESSAGE = (
    "The selected model cannot run on any inference host that is available right now"
)


@dataclass(frozen=True)
class ExecutionRequest:
    """What submission knows before the model is resolved.

    Both fields are plain data on purpose: capacity is read once, from one
    session, at the top of submission, and then carried down. That keeps the
    decision a pure function of three values rather than something that can
    re-query - and re-decide - halfway through.
    """

    preferred: ExecutionTarget
    available: frozenset[ExecutionTarget]

    @classmethod
    def for_preference(
        cls, *, prefer_local: bool, available: frozenset[ExecutionTarget]
    ) -> ExecutionRequest:
        return cls(
            preferred=ExecutionTarget.local if prefer_local else ExecutionTarget.cloud,
            available=frozenset(available),
        )


@dataclass(frozen=True)
class ExecutionDecision:
    """Where the job will run, and where the researcher asked for it to run."""

    target: ExecutionTarget
    preferred: ExecutionTarget


def choose_execution_target(
    request: ExecutionRequest, *, eligible: frozenset[ExecutionTarget]
) -> ExecutionDecision:
    """Fix one target, or refuse.

    Order is preference first, then the other host. Nothing else is consulted:
    there is no scoring, no queue depth, no latency estimate - a second host is
    tried only because the first cannot take the work at all.

    Raises :class:`ConflictError` (409) when no eligible host has capacity, with
    a message naming the situation. Refusing is the point: a job created for a
    host that nobody is claiming from has no terminal outcome.
    """
    other = (
        ExecutionTarget.cloud
        if request.preferred is ExecutionTarget.local
        else ExecutionTarget.local
    )
    for candidate in (request.preferred, other):
        if candidate in eligible and candidate in request.available:
            return ExecutionDecision(target=candidate, preferred=request.preferred)

    if not request.available & eligible and request.available:
        # Some host has capacity, but not one this model may run on. Naming the
        # model rather than the hosts is what makes this actionable.
        raise ConflictError(_NO_ELIGIBLE_HOST_MESSAGE)
    raise ConflictError(NO_CAPACITY_MESSAGE)
