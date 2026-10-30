"""Device pairing state machine (RFC 8628 without the typable user code).

The helper starts a pairing, opens the researcher's default browser at
``<app>/pair#<verification_token>``, and polls for the token the browser
authorises. Nothing is ever typed.

Entropy, and why it is enough
-----------------------------
Both pairing secrets are ``secrets.token_urlsafe(32)`` - 256 bits. There is no
short code to guess, which is the point: a 6-to-8 character human-typable code
carries 30-40 bits and is the only brute-forceable surface such a protocol has.
Deleting it beats defending it.

The residual guessing surface is therefore:

* ``verification_token`` - 256 bits, looked up *by* digest because the browser
  holds no row id. Guessing it inside the 15 minute window at the
  10-requests/60s/IP auth throttle needs ~2^255 requests.
* ``device_code`` - 256 bits, and the caller must also present the matching
  ``pairing_id`` (a 122-bit UUID4) to even reach the compare. Five wrong
  presentations burn the row permanently, so the practical budget is five
  guesses against 2^256, once, per 15 minute window.

Pair codes are additionally single-use (``consumed_at``), short-lived
(``expires_at``, default 300s, extended only by *successful* polls up to a 15
minute hard cap), and cadence-throttled on the row itself rather than by the
shared auth limiter - a compliant 5 second poll is 12 requests/minute and would
trip a 10/60s limiter at poll 11.

What this state machine does *not* solve
----------------------------------------
Nothing here binds an approval to the device that requested it. A consent link
is transferable by construction: whoever opens it and clicks approve grants a
device token on *their* account to whichever process holds the ``device_code``.
The confirmation code (:func:`opaque_tokens.confirmation_code`) exists so a
researcher can notice that, and the shortened lifetime bounds how long a stolen
link stays useful, but neither is a fix. See ADR 0001, "Pairing phishing".

Logging
-------
Every state transition is logged with ``pairing_id``, ``device_id``,
``user_id``, and ``token_prefix`` - identifiers that are useless to an attacker
and sufficient for an incident. No raw ``device_code``, ``verification_token``,
or device token is ever passed to the logger, and
``test_raw_secrets_never_reach_log_output`` enforces that.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from uuid import UUID, uuid4

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.exceptions import ConflictError, NotFoundError
from backend.core.settings.device import DeviceSettings, get_device_settings
from backend.ml.application.device_management import _mint_token
from backend.ml.application.device_status import _clean_text, _device_status, _now
from backend.ml.application.opaque_tokens import (
    confirmation_code,
    hash_secret,
    new_secret,
    secret_matches,
)
from backend.ml.application.pairing_types import PairingPollResult, StartedPairing
from backend.ml.domain.devices import DeviceStatus, PairingStatus
from backend.ml.infrastructure.device_orm_models import (
    MAX_DEVICE_NAME_LENGTH,
    MAX_HELPER_VERSION_LENGTH,
    MAX_IP_LENGTH,
    MAX_PLATFORM_LENGTH,
    MAX_USER_AGENT_LENGTH,
    HelperDevice,
    HelperPairing,
)
from backend.ml.infrastructure.device_repository import HelperDeviceRepository
from backend.users.infrastructure.orm_models import User

logger = logging.getLogger(__name__)


class DevicePairingService:
    def __init__(
        self,
        repository: HelperDeviceRepository | None = None,
        settings: DeviceSettings | None = None,
    ) -> None:
        self._repo = repository or HelperDeviceRepository()
        self._settings = settings or get_device_settings()

    @property
    def settings(self) -> DeviceSettings:
        return self._settings

    def device_status(self, device: HelperDevice, *, now: datetime | None = None) -> DeviceStatus:
        return _device_status(device, settings=self._settings, now=now or _now())

    # ------------------------------------------------------------------
    # Helper bootstrap (unauthenticated)
    # ------------------------------------------------------------------

    async def start_pairing(
        self,
        session: AsyncSession,
        *,
        device_name: str,
        platform: str,
        helper_version: str,
        capabilities: dict[str, str],
        request_ip: str,
        user_agent: str | None = None,
        now: datetime | None = None,
    ) -> StartedPairing:
        """Create a pairing request and return its two one-time secrets.

        The only cap here is a platform-wide ceiling on live rows. A per-IP cap
        was removed: ``client_ip_for_request`` returns the direct peer unless a
        trusted-proxy range is configured, and the production deployment sits
        behind a proxy it does not allowlist, so every caller shares one address.
        A cap keyed on that value is not a per-client cap - it is a global one,
        and three unauthenticated requests would have blocked pairing for the
        whole platform.

        The replacement ceiling is global and deliberately large. It bounds the
        table; it does not stop an adversary, and it is not pretended to. Someone
        who can sustain a flood against this route can hold it full - but that is
        a request flood, answered at the edge, not by a counter here. The old cap
        was different in kind: three requests were enough.
        """
        now = now or _now()
        settings = self._settings
        await self._sweep_finished_pairings(session, now=now)
        live = await self._repo.count_live_pairings(session, now)
        if live >= settings.device_pairing_max_live_total:
            # The route turns this into 429 + Retry-After; it is the only
            # conflict condition this method has.
            logger.error(
                "device_pairing_ceiling_reached live=%s ceiling=%s",
                live,
                settings.device_pairing_max_live_total,
            )
            raise ConflictError("Pairing is temporarily unavailable; please try again shortly")

        device_code = new_secret()
        verification_token = new_secret()
        key = settings.hmac_key()
        pairing = HelperPairing(
            id=uuid4(),
            device_code_hash=hash_secret(device_code, key),
            verification_token_hash=hash_secret(verification_token, key),
            requested_name=_clean_text(
                device_name, limit=MAX_DEVICE_NAME_LENGTH, fallback="Unnamed computer"
            ),
            requested_platform=_clean_text(platform, limit=MAX_PLATFORM_LENGTH),
            requested_helper_version=_clean_text(helper_version, limit=MAX_HELPER_VERSION_LENGTH),
            requested_capabilities=capabilities or {},
            request_ip=request_ip[:MAX_IP_LENGTH],
            request_user_agent=(
                _clean_text(user_agent, limit=MAX_USER_AGENT_LENGTH, fallback="")
                if user_agent
                else None
            ),
            attempts=0,
            delivery_count=0,
            poll_interval_seconds=settings.device_pairing_poll_interval_seconds,
            created_at=now,
            expires_at=now + timedelta(seconds=settings.device_pairing_ttl_seconds),
        )
        self._repo.add_pairing(session, pairing)
        await session.commit()
        logger.info(
            "device_pairing_started pairing_id=%s name=%r platform=%s helper_version=%s "
            "peer_ip=%s expires_at=%s",
            pairing.id,
            pairing.requested_name,
            pairing.requested_platform,
            pairing.requested_helper_version,
            pairing.request_ip,
            pairing.expires_at.isoformat(),
        )
        return StartedPairing(
            pairing_id=pairing.id,
            device_code=device_code,
            # Fragment, not query: ``location.hash`` never reaches a server log,
            # a Referer header, or (after history.replaceState) the address bar.
            verification_url=f"{settings.pair_url_origin()}/pair#{verification_token}",
            confirmation_code=self.confirmation_code(pairing.id),
            expires_in=settings.device_pairing_ttl_seconds,
            interval_seconds=pairing.poll_interval_seconds,
        )

    def confirmation_code(self, pairing_id: UUID) -> str:
        """The code shown by the helper *and* on the consent screen, for comparison."""
        return confirmation_code(pairing_id, self._settings.hmac_key())

    async def _sweep_finished_pairings(self, session: AsyncSession, *, now: datetime) -> int:
        """Delete pairing rows that can never be acted on again.

        Run from the one endpoint that inserts into this table, so cleanup is
        proportional to insertion and needs no background loop - which matters
        because the production API is serverless and has no background loop to
        put it in.
        """
        cutoff = now - timedelta(seconds=self._settings.device_pairing_retention_seconds)
        deleted = await self._repo.delete_finished_pairings(session, cutoff)
        if deleted:
            logger.info("device_pairings_swept deleted=%s cutoff=%s", deleted, cutoff.isoformat())
        return deleted

    async def collect_token(
        self,
        session: AsyncSession,
        *,
        pairing_id: UUID,
        device_code: str,
        now: datetime | None = None,
    ) -> PairingPollResult:
        """Advance the poll loop and, once approved, mint the device token.

        The raw token is generated here, inside the transaction that verifies
        ``device_code``, and is never persisted - not even for the seconds
        between approval and collection.
        """
        now = now or _now()
        settings = self._settings
        pairing = await self._repo.get_pairing_for_update(session, pairing_id)
        if pairing is None:
            # Indistinguishable from a genuinely expired row: an unknown
            # pairing id must not confirm or deny that it ever existed.
            return PairingPollResult(
                status=PairingStatus.expired,
                interval_seconds=settings.device_pairing_poll_interval_seconds,
            )

        if self._polled_too_soon(pairing, now):
            pairing.poll_interval_seconds = min(
                pairing.poll_interval_seconds * 2,
                settings.device_pairing_max_poll_interval_seconds,
            )
            pairing.last_polled_at = now
            await session.commit()
            return PairingPollResult(
                status=PairingStatus.slow_down,
                interval_seconds=pairing.poll_interval_seconds,
            )
        pairing.last_polled_at = now

        rejected = await self._reject_bad_code(session, pairing, device_code, now=now)
        if rejected is not None:
            return rejected
        return await self._collect_approved(session, pairing, now=now)

    async def _reject_bad_code(
        self, session: AsyncSession, pairing: HelperPairing, device_code: str, *, now: datetime
    ) -> PairingPollResult | None:
        """Burn or warn on a wrong code. ``None`` means the code matched."""
        settings = self._settings
        if secret_matches(pairing.device_code_hash, device_code, settings.hmac_key()):
            return None
        # Saturating, not wrapping: attempts is a SmallInteger and a burned
        # row still accepts requests, so an unbounded counter would overflow.
        if pairing.attempts < settings.device_pairing_max_attempts:
            pairing.attempts += 1
        if pairing.attempts >= settings.device_pairing_max_attempts and pairing.denied_at is None:
            pairing.denied_at = now
            logger.warning(
                "device_pairing_burned pairing_id=%s attempts=%s",
                pairing.id,
                pairing.attempts,
            )
        else:
            logger.warning(
                "device_pairing_bad_device_code pairing_id=%s attempts=%s",
                pairing.id,
                pairing.attempts,
            )
        await session.commit()
        return PairingPollResult(
            status=PairingStatus.access_denied, interval_seconds=pairing.poll_interval_seconds
        )

    async def _collect_approved(
        self, session: AsyncSession, pairing: HelperPairing, *, now: datetime
    ) -> PairingPollResult:
        """Resolve a code-verified poll: terminal states, pending wait, or mint."""
        settings = self._settings
        # Only a caller holding the device code learns the real state.
        if pairing.denied_at is not None or pairing.consumed_at is not None:
            await session.commit()
            return PairingPollResult(
                status=PairingStatus.access_denied, interval_seconds=pairing.poll_interval_seconds
            )
        if pairing.expires_at <= now:
            await session.commit()
            return PairingPollResult(
                status=PairingStatus.expired, interval_seconds=pairing.poll_interval_seconds
            )
        if pairing.approved_at is None or pairing.device_id is None:
            # "Went to find my password manager" must not expire the pairing,
            # so a successful poll extends it - up to a hard lifetime cap.
            pairing.expires_at = min(
                now + timedelta(seconds=settings.device_pairing_ttl_seconds),
                pairing.created_at
                + timedelta(seconds=settings.device_pairing_max_lifetime_seconds),
            )
            await session.commit()
            return PairingPollResult(
                status=PairingStatus.authorization_pending,
                interval_seconds=pairing.poll_interval_seconds,
            )

        device = await self._repo.get_device_for_update(session, pairing.device_id)
        if device is None or device.revoked_at is not None:
            pairing.denied_at = now
            await session.commit()
            return PairingPollResult(
                status=PairingStatus.access_denied, interval_seconds=pairing.poll_interval_seconds
            )

        issued = _mint_token(device, settings, now=now)
        pairing.consumed_at = now
        pairing.delivery_count += 1
        await session.commit()
        logger.info(
            "device_token_issued device_id=%s user_id=%s pairing_id=%s token_prefix=%s "
            "expires_at=%s",
            device.id,
            device.user_id,
            pairing.id,
            device.token_prefix,
            issued.token_expires_at.isoformat(),
        )

        # Re-read to pick up the owning user for the helper's confirmation line.
        stored = await self._repo.get_device(session, device.id)
        return PairingPollResult(
            status=PairingStatus.approved,
            interval_seconds=pairing.poll_interval_seconds,
            device_id=device.id,
            device_token=issued.device_token,
            token_expires_at=issued.token_expires_at,
            account_email=stored.user.email if stored is not None else None,
        )

    def _polled_too_soon(self, pairing: HelperPairing, now: datetime) -> bool:
        if pairing.last_polled_at is None:
            return False
        elapsed = (now - pairing.last_polled_at).total_seconds()
        return elapsed < max(pairing.poll_interval_seconds - 1, 0)

    # ------------------------------------------------------------------
    # Browser consent (Bearer JWT)
    # ------------------------------------------------------------------

    async def lookup_pairing(
        self,
        session: AsyncSession,
        *,
        verification_token: str,
        now: datetime | None = None,
    ) -> HelperPairing:
        """Resolve the fragment token the helper handed to the browser."""
        now = now or _now()
        pairing = await self._repo.get_pairing_by_verification_hash(
            session, hash_secret(verification_token, self._settings.hmac_key())
        )
        if (
            pairing is None
            or pairing.denied_at is not None
            or pairing.consumed_at is not None
            or pairing.expires_at <= now
        ):
            # One indistinguishable 404 for unknown / expired / used / denied.
            raise NotFoundError("Pairing request not found")
        return pairing

    async def approve_pairing(
        self,
        session: AsyncSession,
        *,
        user: User,
        pairing_id: UUID,
        verification_token: str,
        request_ip: str | None = None,
        now: datetime | None = None,
    ) -> HelperDevice:
        """Consent step. Creates the device row with ``token_hash = ''``.

        ``verification_token`` is re-verified here rather than trusted from the
        earlier lookup, so possession of the fragment - not merely knowledge of
        a pairing id - is what authorises the grant.
        """
        now = now or _now()
        settings = self._settings
        pairing = await self._repo.get_pairing_for_update(session, pairing_id)
        if pairing is None or not secret_matches(
            pairing.verification_token_hash, verification_token, settings.hmac_key()
        ):
            raise NotFoundError("Pairing request not found")
        if pairing.denied_at is not None or pairing.consumed_at is not None:
            raise ConflictError("This pairing request has already been used")
        if pairing.expires_at <= now:
            raise ConflictError("This pairing request has expired")

        if pairing.approved_at is not None and pairing.device_id is not None:
            if pairing.approved_user_id != user.id:
                raise ConflictError("This pairing request belongs to another account")
            existing = await self._repo.get_device(session, pairing.device_id)
            if existing is None:
                raise ConflictError("This pairing request is no longer valid")
            return existing

        # The per-user device cap is counted under the user's advisory lock so
        # two pairings approved in parallel cannot both pass it.
        await session.execute(
            text("SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))"),
            {"key": f"device-cap:{user.id}"},
        )
        live = await self._repo.count_live_devices(session, user.id)
        if live >= settings.device_max_per_user:
            raise ConflictError(
                f"You have reached the limit of {settings.device_max_per_user} paired computers"
            )

        device = HelperDevice(
            id=uuid4(),
            user_id=user.id,
            name=pairing.requested_name,
            platform=pairing.requested_platform,
            helper_version=pairing.requested_helper_version,
            capabilities=dict(pairing.requested_capabilities or {}),
            token_hash="",
            token_prefix="",
            paired_from_ip=(request_ip or pairing.request_ip)[:MAX_IP_LENGTH],
            created_at=now,
            updated_at=now,
        )
        self._repo.add_device(session, device)
        await session.flush()
        pairing.approved_user_id = user.id
        pairing.approved_at = now
        pairing.device_id = device.id
        await session.commit()
        # Audit line for a grant that can't otherwise be recovered: who approved
        # what, under which presented name, and the confirmation code shown.
        logger.info(
            "device_pairing_approved pairing_id=%s device_id=%s user_id=%s name=%r "
            "platform=%s helper_version=%s confirmation_code=%s peer_ip=%s",
            pairing.id,
            device.id,
            user.id,
            device.name,
            device.platform,
            device.helper_version,
            self.confirmation_code(pairing.id),
            device.paired_from_ip,
        )
        return device

    async def deny_pairing(
        self,
        session: AsyncSession,
        *,
        user: User,
        pairing_id: UUID,
        verification_token: str,
        now: datetime | None = None,
    ) -> None:
        now = now or _now()
        pairing = await self._repo.get_pairing_for_update(session, pairing_id)
        if pairing is None or not secret_matches(
            pairing.verification_token_hash, verification_token, self._settings.hmac_key()
        ):
            raise NotFoundError("Pairing request not found")
        if pairing.denied_at is None:
            pairing.denied_at = now
            logger.info("device_pairing_denied pairing_id=%s user_id=%s", pairing.id, user.id)
        await session.commit()
