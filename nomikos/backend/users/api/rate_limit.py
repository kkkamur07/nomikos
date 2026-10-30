"""Postgres-backed rate limiter for sensitive and unauthenticated endpoints.

Each attempt is recorded as a row in ``auth_rate_limit_attempts``. Because the
store lives in the shared database, the limit is enforced uniformly across all
uvicorn worker processes and across serverless invocations - an in-process dict
would silently divide the effective limit by the number of workers, or reset on
every cold start.

Choosing the key is the harder half of the problem. ``request.client.host`` is
the direct TCP peer, which on a managed platform is the provider's proxy rather
than the browser. Keying a "per-client" limit on that value produces one global
bucket: it throttles the whole product instead of the attacker. So the peer is
only used when the deployment declares it meaningful (see
``AppSettings.trust_peer_ip``), and sensitive auth routes carry a second,
independent bucket keyed on the account being targeted - which is correct no
matter what the network path looks like.

That second bucket only holds if the account is read out of every request that
could reach a password check, so nothing about how the client *describes* its
body is allowed to decide whether the body gets read. See
``_request_account_identity``.

The two dimensions are also charged at different moments, which is the other
half of getting the account bucket right - see ``throttle_auth_attempts``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from ipaddress import ip_address, ip_network
from typing import NamedTuple

from fastapi import HTTPException, Request
from sqlalchemy import delete, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.settings.app import get_app_settings
from backend.core.settings.auth import get_auth_settings
from backend.users.infrastructure.orm_models import AuthRateLimitAttempt
from infrastructure.db import AsyncSessionLocal

logger = logging.getLogger(__name__)

#: Auth payloads under this dependency are a handful of short fields; nothing
#: legitimate comes near this. A larger body is *refused*, not skipped: pydantic
#: ignores unknown keys, so `{"email": ..., "password": ..., "pad": "x" * 9000}`
#: still authenticates. If "too big to read" meant "too big to attribute", the
#: account bucket would be one padding field away from being switched off.
MAX_IDENTITY_BODY_BYTES = 8 * 1024

#: Ceiling for requests that name no account and come from no attributable
#: address. Deliberately far above the per-account limit - see
#: ``throttle_auth_attempts`` for why a shared bucket is safe *here* and would be
#: an outage on the main path.
UNATTRIBUTABLE_AUTH_RATE_LIMIT = 300


def client_ip_for_request(request: Request) -> str:
    """Public helper for IP-based throttles outside auth routes."""
    return _real_ip(request)


def _is_trusted_proxy_peer(host: str | None, forwarded_allow_ips: str | None) -> bool:
    if not host or not forwarded_allow_ips:
        return False
    try:
        peer = ip_address(host)
    except ValueError:
        return False

    for entry in forwarded_allow_ips.split(","):
        try:
            if peer in ip_network(entry.strip(), strict=False):
                return True
        except ValueError:
            continue
    return False


def _forwarded_client_ip(forwarded_for: str) -> str | None:
    """Return the canonical leftmost X-Forwarded-For address, if valid."""
    first_hop = forwarded_for.split(",", maxsplit=1)[0].strip()
    try:
        return str(ip_address(first_hop))
    except ValueError:
        return None


def _trusted_forwarded_ip(request: Request) -> str | None:
    app_settings = get_app_settings()
    if not app_settings.behind_proxy:
        return None
    forwarded_for = request.headers.get("X-Forwarded-For")
    if not forwarded_for:
        return None
    client_host = request.client.host if request.client else None
    if not _is_trusted_proxy_peer(client_host, app_settings.forwarded_allow_ips):
        return None
    return _forwarded_client_ip(forwarded_for)


def _real_ip(request: Request) -> str:
    client_ip = _trusted_forwarded_ip(request)
    if client_ip:
        return client_ip
    client_host = request.client.host if request.client else None
    if client_host:
        return client_host[:128]
    raise HTTPException(status_code=400, detail="Unable to identify request client")


def attributable_client_ip(request: Request) -> str | None:
    """Return an address that identifies one client, or ``None`` if none does.

    ``None`` is the honest answer when the process sits behind a proxy tier it
    cannot allowlist: the peer address is then shared by every visitor, and a
    bucket keyed on it says nothing about who sent the request.
    """
    client_ip = _trusted_forwarded_ip(request)
    if client_ip:
        return client_ip
    if not get_app_settings().trust_peer_ip:
        return None
    client_host = request.client.host if request.client else None
    return client_host[:128] if client_host else None


def _account_identity(payload: object) -> str | None:
    """Hash the account an auth request targets, for identity-keyed throttling."""
    if not isinstance(payload, dict):
        return None
    email = payload.get("email")
    if not isinstance(email, str):
        return None
    normalized = email.strip().casefold()
    if not normalized:
        return None
    # Hashed so the limiter table never becomes a list of registered addresses.
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


async def _request_account_identity(request: Request) -> str | None:
    """Hash the account a request targets, whatever it claims to be sending.

    Deliberately *not* gated on ``Content-Type``. Media types are case-insensitive
    (RFC 9110 s8.3.1) and FastAPI decodes a body as JSON for any
    ``application/*+json`` subtype and for a request that declares no content type
    at all - so a gate that string-compares the header against
    ``"application/json"`` is one an attacker steps over by capitalising a letter,
    while the route behind it parses the body and checks the password regardless.
    Erasing the account key is the whole attack, because on a deployment with
    ``TRUST_PEER_IP=false`` there is no IP key to fall back on. The bytes decide,
    not the header.

    Raises 413 for a body too large to attribute; see ``MAX_IDENTITY_BODY_BYTES``.
    """
    try:
        raw = await request.body()
    except Exception:  # pragma: no cover - client disconnect during read
        return None
    if not raw:
        return None
    if len(raw) > MAX_IDENTITY_BODY_BYTES:
        raise HTTPException(status_code=413, detail="Request body too large")
    # Starlette caches the body on the request, so the route still parses it.
    try:
        payload = json.loads(raw)
    except (ValueError, RecursionError):
        # Not JSON, so it cannot reach a password check either: every route under
        # this dependency declares a pydantic body model, and FastAPI only fills
        # those from JSON - none of them accept form-encoded or multipart input.
        # There is nothing further to probe. RecursionError joins ValueError
        # because the stdlib scanner raises it, not a JSONDecodeError, on deeply
        # nested input, and a 500 here would be its own denial of service.
        return None
    return _account_identity(payload)


def _normalized(keys: Sequence[str]) -> list[str]:
    # Sorted so two requests sharing two keys take the advisory locks in the same
    # order and cannot deadlock.
    return sorted({key[:255] for key in keys})


@asynccontextmanager
async def _locked_rate_limit_session(
    keys: Sequence[str],
) -> AsyncIterator[tuple[AsyncSession, list[str]]]:
    """Yield one session holding every key's advisory lock, in sorted order.

    The check and the charge happen inside the same lock: without it N parallel
    attempts all read the bucket under the cap and all proceed. Locks are taken
    in sorted order so two requests sharing two keys cannot deadlock.
    """
    unique_keys = _normalized(keys)
    async with AsyncSessionLocal() as db:
        for key in unique_keys:
            await db.execute(
                text("SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))"),
                {"key": key},
            )
        yield db, unique_keys


async def _prune_expired(db: AsyncSession, keys: Sequence[str], window_start: datetime) -> None:
    for key in keys:
        await db.execute(
            delete(AuthRateLimitAttempt).where(
                AuthRateLimitAttempt.key == key,
                AuthRateLimitAttempt.attempted_at < window_start,
            )
        )


async def _attempt_count(db: AsyncSession, key: str, window_start: datetime) -> int:
    return (
        await db.scalar(
            select(func.count())
            .select_from(AuthRateLimitAttempt)
            .where(
                AuthRateLimitAttempt.key == key,
                AuthRateLimitAttempt.attempted_at >= window_start,
            )
        )
        or 0
    )


async def consume_rate_limit(
    keys: Sequence[str],
    *,
    limit: int,
    window_seconds: int,
    detail: str,
) -> None:
    """Charge one attempt against every key, or raise 429 if any is exhausted.

    All keys are checked before any is charged, so a request rejected by one
    bucket does not consume budget in the others.
    """
    unique_keys = _normalized(keys)
    if not unique_keys:
        return
    now = datetime.now(UTC)
    window_start = now - timedelta(seconds=window_seconds)
    async with _locked_rate_limit_session(keys) as (db, locked_keys):
        await _prune_expired(db, locked_keys, window_start)
        for key in locked_keys:
            if await _attempt_count(db, key, window_start) >= limit:
                await db.rollback()
                raise HTTPException(
                    status_code=429,
                    detail=detail,
                    headers={"Retry-After": str(window_seconds)},
                )
        for key in locked_keys:
            db.add(AuthRateLimitAttempt(key=key, attempted_at=now))
        await db.commit()


async def refund_rate_limit(keys: Sequence[str], *, window_seconds: int) -> None:
    """Return one attempt per key, for a charge the caller did not spend.

    The account bucket is charged before the password is verified and refunded
    when the attempt succeeds, so only failures spend budget while a parallel
    burst cannot slip the cap.

    The select and the delete run under the key's advisory lock: unlocked, two
    concurrent successes picked the same latest row, both deleted that one id,
    and only one charge came back.
    """
    unique_keys = _normalized(keys)
    if not unique_keys:
        return
    window_start = datetime.now(UTC) - timedelta(seconds=window_seconds)
    async with _locked_rate_limit_session(unique_keys) as (db, locked_keys):
        for key in locked_keys:
            row_id = await db.scalar(
                select(AuthRateLimitAttempt.id)
                .where(
                    AuthRateLimitAttempt.key == key,
                    AuthRateLimitAttempt.attempted_at >= window_start,
                )
                .order_by(
                    AuthRateLimitAttempt.attempted_at.desc(),
                    AuthRateLimitAttempt.id.desc(),
                )
                .limit(1)
            )
            if row_id is not None:
                await db.execute(
                    delete(AuthRateLimitAttempt).where(AuthRateLimitAttempt.id == row_id)
                )
        await db.commit()


class AuthRateLimitKeys(NamedTuple):
    """The two independent dimensions an auth attempt can be charged against.

    * ``ip`` - only when the address identifies one client. ``None`` rather than
      a global bucket, because a global login limit is a self-inflicted outage,
      not a defence.
    * ``account`` - the targeted account, hashed. Survives any proxy topology
      and is what caps online password guessing against a user.

    They are kept apart rather than concatenated because they are charged at
    different moments; see ``throttle_auth_attempts``.
    """

    ip: str | None
    account: str | None


async def auth_rate_limit_keys(request: Request) -> AuthRateLimitKeys:
    """Resolve both dimensions for one request; either may be ``None``."""
    path = request.url.path
    client_ip = attributable_client_ip(request)
    identity = await _request_account_identity(request)
    return AuthRateLimitKeys(
        ip=f"ip:{client_ip}:{path}" if client_ip else None,
        account=f"account:{identity}:{path}" if identity else None,
    )


async def throttle_auth_attempts(request: Request) -> AsyncIterator[None]:
    """Throttle a sign-in attempt on both dimensions, charging each when it is due.

    The IP bucket is charged before the handler runs: an attacker's budget has to
    shrink whether or not a guess lands.

    The account bucket is charged before the handler runs too, under the same
    advisory lock the IP bucket uses. Checking it unlocked and charging only
    after a failure let N parallel attempts all read the bucket under the cap,
    so a burst slipped the per-account limit entirely. A success refunds its
    charge below, so only failures spend budget: anyone who knows an address
    still cannot hold its owner at HTTP 429 with successful sign-ins, and on a
    deployment where ``TRUST_PEER_IP=false`` leaves no IP dimension the bucket
    still caps guessing without capping the owner.
    """
    settings = get_auth_settings()
    keys = await auth_rate_limit_keys(request)
    window = settings.auth_rate_limit_window_seconds
    limit = settings.auth_rate_limit_requests
    detail = "Too many authentication attempts; try again later"

    if keys.ip is None and keys.account is None:
        # Fail closed, but coarsely. Now that identity extraction ignores the
        # declared content type, the only requests reaching this branch are ones
        # that name no account at all - and every route under this dependency
        # requires an `email`, so such a body is rejected before any password is
        # checked. That is what makes one shared bucket safe here and an outage
        # on the main path: no real sign-in enters this bucket, so exhausting it
        # locks nobody out of signing in.
        #
        # That claim is only true while every route under this dependency is an
        # `/auth/*` route whose body carries an email. `POST /device/v1/pairings`
        # used to sit here and did not: its body has no email, so *every* honest
        # pairing landed in this shared bucket and one attacker at ~5 req/s
        # locked the whole platform out of `nomikos pair`. It now has its own
        # limiter - ``throttle_device_pairing_starts``. Do not mount a route here
        # unless its body names the account it is acting on.
        #
        # Returning without charging anything, as this branch used to, meant a
        # single capitalised header bought unmetered password guessing.
        logger.warning("auth_rate_limit_unattributable path=%s", request.url.path)
        await consume_rate_limit(
            [f"unattributable:{request.url.path}"],
            limit=UNATTRIBUTABLE_AUTH_RATE_LIMIT,
            window_seconds=window,
            detail="Too many unattributable requests; try again later",
        )
        yield
        return

    if keys.ip is not None:
        await consume_rate_limit([keys.ip], limit=limit, window_seconds=window, detail=detail)
    if keys.account is not None:
        await consume_rate_limit([keys.account], limit=limit, window_seconds=window, detail=detail)
    succeeded = False
    try:
        yield
        succeeded = True
    finally:
        if succeeded and keys.account is not None:
            await refund_rate_limit([keys.account], window_seconds=window)


#: A helper installation runs `nomikos pair` once, so a handful of starts per
#: window per client is generous. The bucket exists to stop one machine looping
#: the route, not to bound the platform - that is the live-pairing ceiling in
#: ``DevicePairingService.start_pairing``.
#:
#: The budget lives in ``AuthSettings.device_pairing_rate_limit_requests`` so a
#: deployment pairing many devices from one NAT can raise it without also
#: raising the password-guessing budget.


async def throttle_device_pairing_starts(request: Request) -> None:
    """Cap unauthenticated pairing starts per attributable client - and only per client.

    ``POST /device/v1/pairings`` carries no account identity, so it has no second
    dimension to fall back on. Under the shared auth throttle it therefore landed
    in the coarse ``unattributable:<path>`` bucket, which is one bucket for every
    researcher on the platform: filling it locked all of them out of pairing a
    helper. ``DevicePairingService.start_pairing`` had already removed a per-IP
    cap for exactly this reason, and this dependency reinstated it one layer up.

    So when no address identifies one client, nothing is charged. That is the
    same posture the public thumbnail throttle takes, for the same reason: the
    only alternative is a global bucket, and a global bucket on this route is the
    outage. What bounds the work an unattributable flood can do is the
    platform-wide live-pairing ceiling, which answers cheaply and never rejects a
    caller who is not part of the flood.
    """
    client_ip = attributable_client_ip(request)
    if client_ip is None:
        return
    settings = get_auth_settings()
    await consume_rate_limit(
        [f"device-pairing:{client_ip}"],
        limit=settings.device_pairing_rate_limit_requests,
        window_seconds=settings.auth_rate_limit_window_seconds,
        detail="Too many pairing requests; try again later",
    )
