"""Who is asking for a page of work, and which **execution target** they may take.

There is one inference agent implementation (ADR 0003) and therefore one claim
endpoint. It accepts two credentials, and the credential - not a request field -
decides the **execution target** the caller may claim:

* an ``X-Nomikos-Device-Token`` is a researcher's own computer. It claims
  ``local`` work belonging to the single account on ``helper_devices.user_id``,
  and nothing else. It can never claim ``cloud``: cloud work is platform work and
  is not scoped by that foreign key, so honouring a device token for it would
  hand one researcher's laptop every account's pages.
* an ``X-Nomikos-Service-Token`` is a hosted worker. It claims ``cloud`` work
  regardless of which account submitted it, which is the same asymmetry
  ``hosts_with_recent_devices`` already encodes: a laptop answers for its owner,
  a hosted worker answers for everyone.

The target is therefore never negotiable by the caller, and the two rules are one
line each rather than a policy object.

The service account, decided here
---------------------------------
``helper_devices.user_id`` is ``NOT NULL`` by design (ADR 0001, decision 6): that
foreign key *is* the authorization scope of a device credential, and making it
nullable to accommodate a hosted worker would delete the invariant for every
device. So a hosted worker's row needs an owner, and the owner must not be a
researcher - a researcher's account deletion cascades to their devices, their
``GET /devices`` would list platform infrastructure, and revoking it from a phone
would stop cloud inference for everyone.

The owner is therefore a **service account**: one ``users`` row that no one can
log into, addressed by a *fixed UUID5 primary key* rather than by email. Keying
on the primary key is what makes it un-hijackable - anyone can try to register an
address, nobody can choose their own uuid4. Its password is a bcrypt hash of a
discarded 256-bit secret, so no password grants it, and it holds no browser
session, no project, and no document. The only thing it owns is the
``inference_host = cloud`` device rows that report cloud **capacity**.

Those rows are provisioned on first claim rather than by hand: a hosted worker
registers itself by working, exactly as a laptop registers itself by pairing.
Provisioning takes an advisory lock so two workers starting together create one
row each rather than two rows for the same name.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID, uuid5

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.exceptions import InvalidCredentialsError
from backend.core.settings.device import DeviceSettings, get_device_settings
from backend.ml.application.device_auth import authenticate_device
from backend.ml.application.opaque_tokens import new_secret
from backend.ml.domain.execution import ExecutionTarget
from backend.ml.infrastructure.device_orm_models import (
    MAX_DEVICE_NAME_LENGTH,
    MAX_IP_LENGTH,
    HelperDevice,
)
from backend.users.application.password import hash_password_async
from backend.users.infrastructure.orm_models import User

logger = logging.getLogger(__name__)

SERVICE_TOKEN_HEADER = "X-Nomikos-Service-Token"  # noqa: S105 - a header name, not a token
"""Hosted-worker credential. A separate header from the device token for the same
reason the device token is separate from ``Authorization``: two credentials that
resolve to different scopes must not be interchangeable by accident."""

WORKER_NAME_HEADER = "X-Nomikos-Worker-Name"
"""Which hosted worker is calling. Names its ``helper_devices`` row; not a secret.

**Required**, with no default. It used to fall back to ``"cloud-worker"``, which
meant every hosted worker that did not send it resolved to the *same*
``helper_devices`` row - and therefore the same ``device.id``, and therefore the
same ``agent_claim_owner(device_id)`` written to ``jobs.claimed_by``. Two cloud
workers were then indistinguishable to ``job_is_held_by``, so either one could
complete or fail the page the other was running, and the claim service's promise
that "which agent holds this page is answerable from the row alone" was false for
the whole hosted fleet.

It is not a secret and it is not authentication - the service token is both. It
is an identity, and the reason it is now mandatory is that a missing identity
silently became a shared one. A worker that cannot name itself is refused with
the same 401 as a bad token, because the platform cannot tell it apart from the
one already running."""

MISSING_WORKER_NAME_ERROR = (
    f"{WORKER_NAME_HEADER} is required: a hosted worker must identify itself"
)

# Fixed, so the row is addressed by primary key rather than by an address someone
# else could claim first. Derived rather than random so it is the same value in
# every environment and readable in a log line.
SERVICE_ACCOUNT_ID: UUID = uuid5(
    UUID("6ba7b811-9dad-11d1-80b4-00c04fd430c8"),  # NAMESPACE_URL
    "https://nomikos.app/service-accounts/inference-worker",
)
SERVICE_ACCOUNT_USERNAME = "nomikos-inference-worker"
# RFC 2606 reserved TLD: this address is unroutable by construction, so nobody is
# ever going to be emailed at it and no mail path can be confused by it.
SERVICE_ACCOUNT_EMAIL = "inference-worker@nomikos.invalid"

# One constant, because the thing being serialised is "provisioning a hosted
# worker row", not one particular name.
_SERVICE_WORKER_LOCK_ID = 0x52C1A1D


@dataclass(frozen=True)
class InferenceAgent:
    """A caller authorised to claim one page, and the host it claims for.

    Plain values, deliberately: the claim loop runs without a request-scoped
    session, so nothing here may be a live ORM instance bound to a connection
    that has already been returned to the pool.
    """

    device_id: UUID
    user_id: UUID
    execution_target: ExecutionTarget
    is_service_worker: bool

    @property
    def claims_own_account_only(self) -> bool:
        return self.execution_target is ExecutionTarget.local


async def resolve_inference_agent(
    session: AsyncSession,
    *,
    device_token: str | None,
    service_token: str | None,
    worker_name: str | None = None,
    request_ip: str | None = None,
    now: datetime | None = None,
    settings: DeviceSettings | None = None,
) -> InferenceAgent:
    """Authenticate a claimant and record that it is alive.

    Liveness is recorded here rather than on a separate endpoint because polling
    for work *is* the liveness signal: an agent asking for a page is an agent
    that would run one. ``last_seen_at`` is what submission reads as **capacity**,
    so a worker that polls keeps its host submittable without a second call.

    Raises :class:`InvalidCredentialsError` for every failure mode.
    """
    settings = settings or get_device_settings()
    now = now or datetime.now(UTC)

    if service_token is not None:
        device = await _resolve_service_worker(
            session,
            presented_token=service_token,
            worker_name=worker_name,
            request_ip=request_ip,
            now=now,
            settings=settings,
        )
        return InferenceAgent(
            device_id=device.id,
            user_id=device.user_id,
            execution_target=ExecutionTarget.cloud,
            is_service_worker=True,
        )

    authenticated = await authenticate_device(session, device_token, settings=settings, now=now)
    device = authenticated.device
    device.last_seen_at = now
    if request_ip:
        device.last_seen_ip = request_ip[:MAX_IP_LENGTH]
    await session.commit()
    return InferenceAgent(
        device_id=device.id,
        user_id=authenticated.user.id,
        # Never ``device.inference_host``. A device credential is scoped by one
        # ``user_id``; ``cloud`` work is not scoped by it at all, so the two can
        # only be joined by handing a laptop work it does not own.
        execution_target=ExecutionTarget.local,
        is_service_worker=False,
    )


def _service_token_matches(presented: str, settings: DeviceSettings) -> bool:
    import secrets

    configured = settings.inference_worker_service_token
    if not configured:
        return False
    return secrets.compare_digest(presented, configured)


async def _resolve_service_worker(
    session: AsyncSession,
    *,
    presented_token: str,
    worker_name: str | None,
    request_ip: str | None,
    now: datetime,
    settings: DeviceSettings,
) -> HelperDevice:
    if not _service_token_matches(presented_token, settings):
        # Same 401 as every device rejection, and the same silence about which
        # of "not configured" and "wrong value" it was.
        logger.warning("inference_worker_service_auth_rejected reason=token_mismatch")
        raise InvalidCredentialsError("Invalid service token")

    name = _clean_worker_name(worker_name)
    # Serialise provisioning across processes. Two hosted workers booting at once
    # would otherwise both miss the row and both insert it.
    await session.execute(select(func.pg_advisory_xact_lock(_SERVICE_WORKER_LOCK_ID)))
    user = await _ensure_service_account(session, now=now)
    device = (
        await session.execute(
            select(HelperDevice)
            .where(
                HelperDevice.user_id == user.id,
                HelperDevice.inference_host == ExecutionTarget.cloud,
                HelperDevice.name == name,
                HelperDevice.revoked_at.is_(None),
            )
            .order_by(HelperDevice.created_at, HelperDevice.id)
            .limit(1)
        )
    ).scalar_one_or_none()
    if device is None:
        device = HelperDevice(
            user_id=user.id,
            name=name,
            inference_host=ExecutionTarget.cloud,
            platform="hosted",
            helper_version="service",
            capabilities={},
            # Never authenticates as a device: the empty digest cannot equal any
            # 64-character hex string, so the service credential is the only way
            # in and revoking it is one environment variable.
            token_hash="",
            token_prefix="",
            created_at=now,
            updated_at=now,
        )
        session.add(device)
        logger.info("inference_worker_device_provisioned name=%r user_id=%s", name, user.id)
    device.last_seen_at = now
    if request_ip:
        device.last_seen_ip = request_ip[:MAX_IP_LENGTH]
    await session.commit()
    await session.refresh(device)
    return device


def _clean_worker_name(worker_name: str | None) -> str:
    """The worker's own name, or a refusal. Never a shared default.

    Stripping unprintables can empty an otherwise non-blank header, so that case
    is refused too - falling back would put the caller back on a shared row,
    which is the whole failure this exists to prevent.
    """
    candidate = (worker_name or "").strip()
    printable = "".join(char for char in candidate if char.isprintable())
    if not printable:
        logger.warning("inference_worker_service_auth_rejected reason=missing_worker_name")
        raise InvalidCredentialsError(MISSING_WORKER_NAME_ERROR)
    return printable[:MAX_DEVICE_NAME_LENGTH]


async def _ensure_service_account(session: AsyncSession, *, now: datetime) -> User:
    """The one ``users`` row that owns hosted-worker devices.

    Looked up by primary key. That is the whole security property: an address can
    be registered by whoever gets there first, a fixed UUID5 cannot be, so this
    can never resolve to an account a person controls.
    """
    user = await session.get(User, SERVICE_ACCOUNT_ID)
    if user is not None:
        return user
    user = User(
        id=SERVICE_ACCOUNT_ID,
        email=SERVICE_ACCOUNT_EMAIL,
        username=SERVICE_ACCOUNT_USERNAME,
        # A bcrypt hash of a secret that is discarded on the next line. No
        # password grants this account, and there is nothing to leak.
        hashed_password=await hash_password_async(new_secret()),
        prefer_local_inference=False,
        created_at=now,
    )
    session.add(user)
    await session.flush()
    logger.info("inference_service_account_created user_id=%s", user.id)
    return user
