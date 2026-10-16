"""The platform's device-protocol records, and how to read them back.

The dataclasses are the shapes the CLI actually uses: what pairing hands over,
what the version floor answers, what one claim carries. The readers below turn
a decoded JSON body into one of those, tolerating a refusal that arrived
without the contract (a 426 from something that is not the platform must not
raise a second error on top of the refusal).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime

from nomikos_inference.cli.errors import AgentVersionRefused, PlatformError

# Wire bodies are typed as `object`: JSON decodes to anything, and every reader
# below narrows with `isinstance` before touching a value.

AGENT_VERSION_REFUSED_STATUS = 426
AGENT_VERSION_UNSUPPORTED = "AGENT_VERSION_UNSUPPORTED"
"""The one error code the CLI matches on, in the run loop and the launch check
alike. Stable by contract (the platform's own module says changing it breaks
every agent), and the only refusal that arrives machine-readable, since the
error envelope otherwise replaces `HTTPException.detail` with a fixed public
string."""

# Pairing states, named as the platform names them (`backend/ml/domain/devices.py`).
# All arrive inside a 200 body: the platform's error envelope replaces
# `HTTPException.detail` with a fixed public string, so a machine-readable
# state can't survive a non-2xx response.
STATUS_AUTHORIZATION_PENDING = "authorization_pending"
STATUS_SLOW_DOWN = "slow_down"
STATUS_ACCESS_DENIED = "access_denied"
STATUS_EXPIRED = "expired"
STATUS_APPROVED = "approved"


@dataclass(frozen=True)
class StartedPairing:
    """What `POST /device/v1/pairings` hands an unpaired machine."""

    pairing_id: str
    device_code: str
    verification_url: str
    confirmation_code: str
    """Not a secret and not a `user_code`: no endpoint accepts it. It is a keyed
    derivation of the pairing id, shown here so the researcher can compare it
    with the consent screen (ADR 0001, decision 13)."""
    expires_in: int
    interval_seconds: int


@dataclass(frozen=True)
class PairingPoll:
    """One result from the token collection loop."""

    status: str
    interval_seconds: int
    device_id: str | None = None
    device_token: str | None = None
    token_expires_at: datetime | None = None
    account_email: str | None = None


@dataclass(frozen=True)
class AgentFloor:
    """What the platform makes of this agent's version, asked at launch.

    One shape for both answers: a refusal and a notice carry the same four
    facts (what was presented, the floor, the latest, the package to install)
    and differ only in whether work would have been handed over. Collapsing
    them into one record with two booleans keeps the caller from having to
    catch an exception to learn something the platform already stated.
    """

    agent_version: str
    minimum_version: str
    latest_version: str
    package: str
    refused: bool
    """Below the floor (or unstatable): the platform will not hand this agent a
    **claim** until it upgrades."""
    outdated: bool
    """At or above the floor, behind the latest. Served, and told - a notice, not
    a refusal."""
    message: str = ""
    """The platform's own sentence about it. Printed rather than reworded, so a
    researcher and the server logs say the same thing."""
    upgrade_command: str = ""
    """A hint for a human to read. Never executed: the platform names a package,
    and handing a remote process a server-supplied string to run would be a worse
    bargain than the one ADR 0002 already accepts."""


@dataclass(frozen=True)
class DeviceIdentity:
    """What `GET /device/v1/self` tells a paired machine about itself."""

    device_id: str
    name: str
    account_email: str
    token_expires_at: datetime | None


@dataclass(frozen=True)
class AgentNotice:
    """What the platform makes of this build, delivered *with* the work.

    Arrives on every claim response, page or no page, so an idle agent still
    learns it is behind. **Outdated** is not the same state as being below the
    **version floor**: this one is served normally and merely noted.
    """

    agent_version: str
    latest_version: str
    outdated: bool
    upgrade_command: str


@dataclass(frozen=True)
class ClaimedPage:
    """One page of work, and everything needed to run and report it.

    The run fields are a flattened `JobSubmitRequest`, the same contract the
    inference runtime already takes, which is what keeps a laptop and a hosted
    worker literally the same program (ADR 0003).
    """

    product_job_id: str
    inference_job_id: str
    lease_expires_at: datetime | None
    """When the platform stops considering this agent the owner of the page.
    Read before reporting, to name the cause when the callback is refused."""
    task: str
    registry_model_id: str
    registry_tag: str
    params: dict
    page_image_url: str
    page_image_expires_at: datetime | None
    """Two fields rather than one, so the agent can tell whether its link is
    still worth using without parsing a URL."""

    def image_link_expired(self, *, now: datetime | None = None) -> bool:
        """Whether the **signed page image link** is already dead.

        Checked before the fetch rather than after the 403, so a page whose
        link died (a lid closed mid-claim, a machine that swapped) fails with
        the reason instead of a status the route deliberately makes ambiguous.
        """
        if self.page_image_expires_at is None:
            return False
        return self.page_image_expires_at <= (now or datetime.now(UTC))

    def lease_expired(self, *, now: datetime | None = None) -> bool:
        """Whether this agent's **lease** has run out on the local clock.

        Only used to *explain*, never to skip the callback: the platform is the
        authority on who holds a page, and a laptop with a skewed clock must not
        talk itself out of reporting work it actually finished.
        """
        if self.lease_expires_at is None:
            return False
        return self.lease_expires_at <= (now or datetime.now(UTC))


@dataclass(frozen=True)
class Claim:
    """One answer from the claim endpoint. An empty queue is one of these, not
    an error: it arrives as a 200 with no page, because a healthy platform is
    idle most of the time."""

    page: ClaimedPage | None
    poll_after_seconds: float
    agent: AgentNotice | None


def _parse_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _decode(raw: bytes) -> object:
    if not raw:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def _agent_notice(raw: object) -> AgentNotice | None:
    if not isinstance(raw, dict):
        return None
    return AgentNotice(
        agent_version=str(raw.get("agent_version") or ""),
        latest_version=str(raw.get("latest_version") or ""),
        outdated=bool(raw.get("outdated")),
        upgrade_command=str(raw.get("upgrade_command") or ""),
    )


def _claimed_page(raw: object) -> ClaimedPage | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise TypeError("page must be an object")
    request = raw["request"]
    if not isinstance(request, dict):
        raise TypeError("page.request must be an object")
    params = request.get("params")
    return ClaimedPage(
        product_job_id=str(raw["product_job_id"]),
        inference_job_id=str(raw["inference_job_id"]),
        lease_expires_at=_parse_datetime(raw.get("lease_expires_at")),
        task=str(request["task"]),
        registry_model_id=str(request["registry_model_id"]),
        registry_tag=str(request.get("registry_tag") or "stable"),
        params=params if isinstance(params, dict) else {},
        page_image_url=str(raw["page_image_url"]),
        page_image_expires_at=_parse_datetime(raw.get("page_image_expires_at")),
    )


def _unexpected(status: int, body: object, doing: str) -> str:
    message: object = None
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        message = body["error"].get("message")
    detail = f": {message}" if message else ""
    return f"The platform answered {status} while {doing}{detail}"


def _version_refusal(body: object, base_url: str) -> AgentVersionRefused:
    """Read a 426 back into an exception the run loop can print.

    The refusal is the one failure on this platform that keeps its detail, so
    every field below really is there, but a 426 from something that isn't the
    platform is still possible, so reading it must not raise a `KeyError` on
    top of the refusal.
    """
    error = body.get("error") if isinstance(body, dict) else None
    if not isinstance(error, dict):
        error = {}
    raw_version = error.get("agent_version")
    return AgentVersionRefused(
        message=str(error.get("message") or f"{base_url} refused this version of the agent"),
        agent_version=raw_version if isinstance(raw_version, str) else None,
        minimum_version=str(error.get("minimum_version") or "unknown"),
        latest_version=str(error.get("latest_version") or "unknown"),
        upgrade_command=str(error.get("upgrade_command") or "uv tool upgrade nomikos-inference"),
    )


def _refusal_floor(base_url: str, agent_version: str, body: object) -> AgentFloor:
    """Read a 426 body into the same record a 200 produces.

    A 426 that doesn't carry the contract is a bug on the platform, not a
    stale agent, and it must not be reported as one: an agent told to upgrade
    with no version to upgrade to would fail loudly for the wrong reason.
    """
    error = body.get("error") if isinstance(body, dict) else None
    if not isinstance(error, dict) or error.get("code") != AGENT_VERSION_UNSUPPORTED:
        raise PlatformError(
            f"{base_url} refused this agent's version with {AGENT_VERSION_REFUSED_STATUS} "
            "but did not say what it requires"
        )
    try:
        return AgentFloor(
            agent_version=str(error.get("agent_version") or agent_version),
            minimum_version=str(error["minimum_version"]),
            latest_version=str(error["latest_version"]),
            package=str(error["package"]),
            refused=True,
            outdated=False,
            message=str(error.get("message") or ""),
            upgrade_command=str(error.get("upgrade_command") or ""),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PlatformError(
            f"{base_url} refused this agent's version but did not say what it requires"
        ) from exc
