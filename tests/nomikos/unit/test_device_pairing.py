"""Unit tests for the helper device pairing state machine and device auth.

The pairing service takes its repository by constructor injection, so the whole
state machine - the part that carries every security-sensitive decision - is
driven here against an in-memory double with no Postgres. Real HTTP round-trips
live in tests/nomikos/integration/test_device_pairing.py.
"""

from __future__ import annotations

import hmac
import inspect
import logging
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from backend.core.exceptions import (
    AccessDeniedError,
    ConflictError,
    InvalidCredentialsError,
    NotFoundError,
)
from backend.core.settings.device import DeviceSettings
from backend.ml.api.device_responses import device_response_from_orm, pairing_response_from_orm
from backend.ml.application import opaque_tokens
from backend.ml.application.device_auth import authenticate_device
from backend.ml.application.device_management import DeviceManagementService
from backend.ml.application.pairing_service import DevicePairingService
from backend.ml.domain.devices import DeviceStatus, PairingStatus
from backend.ml.infrastructure.device_orm_models import HelperDevice, HelperPairing
from backend.users.infrastructure.orm_models import User

HMAC_KEY = "device-hmac-key-for-tests-at-least-32-bytes"


# ---------------------------------------------------------------------------
# In-memory doubles
# ---------------------------------------------------------------------------


class _FakeSession:
    """Stands in for AsyncSession; the repository double holds the state."""

    def __init__(self) -> None:
        self.commits = 0
        self.flushes = 0
        self.executed: list[object] = []

    async def commit(self) -> None:
        self.commits += 1

    async def flush(self) -> None:
        self.flushes += 1

    async def execute(self, statement: object, params: object = None) -> None:
        self.executed.append(params)
        return None


class _FakeDeviceRepository:
    def __init__(self) -> None:
        self.devices: dict[uuid.UUID, HelperDevice] = {}
        self.pairings: dict[uuid.UUID, HelperPairing] = {}
        self.users: dict[uuid.UUID, User] = {}

    def _attach_user(self, device: HelperDevice) -> HelperDevice:
        user = self.users.get(device.user_id)
        if user is not None:
            device.user = user
        return device

    async def get_device(self, session, device_id):
        device = self.devices.get(device_id)
        return None if device is None else self._attach_user(device)

    async def get_device_for_update(self, session, device_id):
        return self.devices.get(device_id)

    async def list_devices(self, session, user_id, *, include_revoked=False):
        rows = [device for device in self.devices.values() if device.user_id == user_id]
        if not include_revoked:
            rows = [device for device in rows if device.revoked_at is None]
        return sorted(rows, key=lambda device: (device.created_at, str(device.id)))

    async def count_live_devices(self, session, user_id):
        return len(await self.list_devices(session, user_id))

    def add_device(self, session, device):
        self.devices[device.id] = device
        return device

    async def get_pairing(self, session, pairing_id):
        return self.pairings.get(pairing_id)

    async def get_pairing_for_update(self, session, pairing_id):
        return self.pairings.get(pairing_id)

    async def get_pairing_by_verification_hash(self, session, verification_token_hash):
        for pairing in self.pairings.values():
            if hmac.compare_digest(pairing.verification_token_hash, verification_token_hash):
                return pairing
        return None

    async def count_live_pairings(self, session, now):
        return len(
            [
                pairing
                for pairing in self.pairings.values()
                if pairing.expires_at > now
                and pairing.denied_at is None
                and pairing.consumed_at is None
            ]
        )

    async def delete_finished_pairings(self, session, cutoff):
        doomed = [
            pairing_id
            for pairing_id, pairing in self.pairings.items()
            if pairing.expires_at < cutoff
            or (
                (pairing.consumed_at is not None or pairing.denied_at is not None)
                and pairing.created_at < cutoff
            )
        ]
        for pairing_id in doomed:
            del self.pairings[pairing_id]
        return len(doomed)

    def add_pairing(self, session, pairing):
        self.pairings[pairing.id] = pairing
        return pairing


def _settings(**overrides) -> DeviceSettings:
    """Hermetic settings — every value these tests assert on is stated here.

    ``DeviceSettings`` is a pydantic-settings model, so anything not passed
    explicitly is read from the ambient environment. The integration conftest
    exports ``DEVICE_PAIRING_POLL_INTERVAL_SECONDS=1`` (the routers capture the
    cadence at import time, so it has to be set before the app is imported), and
    when both suites run in one session these unit tests inherited it. A 200ms
    second poll is inside a 5s interval but outside a 1s one, so
    ``test_polling_faster_than_the_interval_returns_slow_down`` failed for a
    reason that had nothing to do with what it tests.

    Pinning the cadence to the production defaults makes these tests assert on
    values they declare rather than on whatever ran before them.
    """
    values = {
        "DEVICE_TOKEN_HMAC_SECRET": HMAC_KEY,
        "DEVICE_PAIRING_APP_ORIGIN": "https://app.nomikos.test",
        "DEVICE_PAIRING_POLL_INTERVAL_SECONDS": 5,
        "DEVICE_PAIRING_MAX_POLL_INTERVAL_SECONDS": 30,
    }
    values.update(overrides)
    return DeviceSettings(**values)


def _user(email: str = "researcher@example.test") -> User:
    return User(
        id=uuid.uuid4(),
        email=email,
        username=email.split("@")[0],
        hashed_password="not-a-real-hash",
    )


@pytest.fixture
def repo() -> _FakeDeviceRepository:
    return _FakeDeviceRepository()


@pytest.fixture
def session() -> _FakeSession:
    return _FakeSession()


@pytest.fixture
def service(repo: _FakeDeviceRepository) -> DevicePairingService:
    return DevicePairingService(repository=repo, settings=_settings())

@pytest.fixture
def management(repo: _FakeDeviceRepository, service: DevicePairingService) -> DeviceManagementService:
    return DeviceManagementService(repository=repo, settings=service.settings)


@pytest.fixture
def owner(repo: _FakeDeviceRepository) -> User:
    user = _user("owner@example.test")
    repo.users[user.id] = user
    return user


@pytest.fixture
def outsider(repo: _FakeDeviceRepository) -> User:
    user = _user("outsider@example.test")
    repo.users[user.id] = user
    return user


async def _pair(
    service: DevicePairingService,
    session: _FakeSession,
    user: User,
    *,
    now: datetime | None = None,
    device_name: str = "Krrish's MacBook Pro",
    request_ip: str = "203.0.113.10",
) -> tuple[HelperPairing, str]:
    """Run start -> approve and return the pairing plus its raw device_code."""
    now = now or datetime.now(UTC)
    started = await service.start_pairing(
        session,
        device_name=device_name,
        platform="darwin-arm64",
        helper_version="0.2.0",
        capabilities={"runtime": "torch"},
        request_ip=request_ip,
        now=now,
    )
    verification_token = started.verification_url.split("#", 1)[1]
    await service.approve_pairing(
        session,
        user=user,
        pairing_id=started.pairing_id,
        verification_token=verification_token,
        now=now,
    )
    pairing = await service._repo.get_pairing(session, started.pairing_id)
    return pairing, started.device_code


# ---------------------------------------------------------------------------
# Credential primitives
# ---------------------------------------------------------------------------


def test_secret_carries_256_bits_of_entropy() -> None:
    """The whole protocol rests on there being no short, guessable code."""
    assert opaque_tokens.SECRET_BYTES == 32
    secret = opaque_tokens.new_secret()
    # token_urlsafe returns base64url of SECRET_BYTES random bytes.
    padded = secret + "=" * (-len(secret) % 4)
    import base64

    assert len(base64.urlsafe_b64decode(padded)) == 32


def test_secret_comparison_is_constant_time() -> None:
    source = inspect.getsource(opaque_tokens.secret_matches)
    assert "hmac.compare_digest" in source
    secret = opaque_tokens.new_secret()
    digest = opaque_tokens.hash_secret(secret, HMAC_KEY)
    assert opaque_tokens.secret_matches(digest, secret, HMAC_KEY)
    assert not opaque_tokens.secret_matches(digest, opaque_tokens.new_secret(), HMAC_KEY)
    # '' is the approved-but-not-collected marker and must never authenticate.
    assert not opaque_tokens.secret_matches("", secret, HMAC_KEY)
    assert not opaque_tokens.secret_matches(None, secret, HMAC_KEY)


@pytest.mark.parametrize(
    "token",
    [
        None,
        "",
        "not-a-token",
        "nmd1.not-a-uuid.secret",
        f"nmd2.{uuid.uuid4()}.secret",
        f"nmd1.{uuid.uuid4()}.",
        f"{uuid.uuid4()}.secret",
    ],
)
def test_malformed_device_tokens_are_rejected(token) -> None:
    assert opaque_tokens.parse_device_token(token) is None


# ---------------------------------------------------------------------------
# Mint -> redeem -> authenticate
# ---------------------------------------------------------------------------


async def test_uncollected_device_cannot_authenticate(service, session, repo, owner) -> None:
    """token_hash = '' is the approved-but-not-collected marker."""
    now = datetime.now(UTC)
    pairing, _ = await _pair(service, session, owner, now=now)
    device = repo.devices[pairing.device_id]
    assert device.token_hash == ""
    assert service.device_status(device, now=now) is DeviceStatus.pairing

    forged = opaque_tokens.format_device_token(device.id, "")
    with pytest.raises(InvalidCredentialsError):
        await authenticate_device(
            session, forged, repository=repo, settings=service.settings, now=now
        )


async def test_device_token_resolves_only_to_its_own_user(
    service, session, repo, owner, outsider
) -> None:
    """A device token is scoped by a NOT NULL FK; it cannot reach another account."""
    now = datetime.now(UTC)
    owner_pairing, owner_code = await _pair(service, session, owner, now=now)
    owner_token = (
        await service.collect_token(
            session, pairing_id=owner_pairing.id, device_code=owner_code, now=now
        )
    ).device_token

    outsider_pairing, outsider_code = await _pair(
        service, session, outsider, now=now, device_name="Outsider laptop"
    )
    await service.collect_token(
        session, pairing_id=outsider_pairing.id, device_code=outsider_code, now=now
    )

    resolved = await authenticate_device(
        session, owner_token, repository=repo, settings=service.settings, now=now
    )
    assert resolved.user.id == owner.id
    assert resolved.user.id != outsider.id
    assert resolved.device.user_id == owner.id


async def test_revoking_another_users_device_is_denied(
    service, session, repo, owner, outsider, management
) -> None:
    """403, and the outsider's own listing never shows the device."""
    now = datetime.now(UTC)
    pairing, _ = await _pair(service, session, owner, now=now)

    with pytest.raises(AccessDeniedError):
        await management.revoke_device(session, user=outsider, device_id=pairing.device_id)

    assert await management.list_devices(session, user=outsider) == []
    assert [device.id for device in await management.list_devices(session, user=owner)] == [
        pairing.device_id
    ]


async def test_revoked_token_fails_on_the_next_request(
    service, session, repo, owner, management
) -> None:
    """No cache, no TTL: revocation lands on the very next device call."""
    now = datetime.now(UTC)
    pairing, device_code = await _pair(service, session, owner, now=now)
    token = (
        await service.collect_token(
            session, pairing_id=pairing.id, device_code=device_code, now=now
        )
    ).device_token

    await authenticate_device(session, token, repository=repo, settings=service.settings, now=now)
    await management.revoke_device(session, user=owner, device_id=pairing.device_id, now=now)

    with pytest.raises(InvalidCredentialsError):
        await authenticate_device(
            session,
            token,
            repository=repo,
            settings=service.settings,
            now=now + timedelta(seconds=1),
        )


async def test_expired_device_token_fails(service, session, repo, owner) -> None:
    now = datetime.now(UTC)
    pairing, device_code = await _pair(service, session, owner, now=now)
    issued = await service.collect_token(
        session, pairing_id=pairing.id, device_code=device_code, now=now
    )

    later = now + timedelta(days=service.settings.device_token_ttl_days + 1)
    with pytest.raises(InvalidCredentialsError):
        await authenticate_device(
            session, issued.device_token, repository=repo, settings=service.settings, now=later
        )


async def test_unknown_device_id_fails_authentication(service, session, repo) -> None:
    forged = opaque_tokens.format_device_token(uuid.uuid4(), opaque_tokens.new_secret())
    with pytest.raises(InvalidCredentialsError):
        await authenticate_device(session, forged, repository=repo, settings=service.settings)


async def test_token_from_a_different_hmac_key_fails(service, session, repo, owner) -> None:
    """Rotating DEVICE_TOKEN_HMAC_SECRET invalidates, it does not silently accept."""
    now = datetime.now(UTC)
    pairing, device_code = await _pair(service, session, owner, now=now)
    token = (
        await service.collect_token(
            session, pairing_id=pairing.id, device_code=device_code, now=now
        )
    ).device_token

    rotated = _settings(DEVICE_TOKEN_HMAC_SECRET="a-completely-different-key-32-bytes-plus")
    with pytest.raises(InvalidCredentialsError):
        await authenticate_device(session, token, repository=repo, settings=rotated, now=now)


# ---------------------------------------------------------------------------
# Pair code lifecycle
# ---------------------------------------------------------------------------


async def test_pair_code_cannot_be_redeemed_twice(service, session, repo, owner) -> None:
    now = datetime.now(UTC)
    pairing, device_code = await _pair(service, session, owner, now=now)

    first = await service.collect_token(
        session, pairing_id=pairing.id, device_code=device_code, now=now
    )
    assert first.status is PairingStatus.approved
    assert first.device_token is not None

    second = await service.collect_token(
        session,
        pairing_id=pairing.id,
        device_code=device_code,
        now=now + timedelta(seconds=30),
    )
    assert second.status is PairingStatus.access_denied
    assert second.device_token is None


async def test_expired_pair_code_fails(service, session, repo, owner) -> None:
    now = datetime.now(UTC)
    pairing, device_code = await _pair(service, session, owner, now=now)

    expired_at = pairing.expires_at + timedelta(seconds=1)
    result = await service.collect_token(
        session, pairing_id=pairing.id, device_code=device_code, now=expired_at
    )
    assert result.status is PairingStatus.expired
    assert result.device_token is None
    assert repo.devices[pairing.device_id].token_hash == ""


async def test_unknown_pairing_id_is_indistinguishable_from_expired(service, session) -> None:
    result = await service.collect_token(
        session, pairing_id=uuid.uuid4(), device_code=opaque_tokens.new_secret()
    )
    assert result.status is PairingStatus.expired


async def test_wrong_device_code_burns_the_pairing_after_max_attempts(
    service, session, repo, owner
) -> None:
    now = datetime.now(UTC)
    pairing, device_code = await _pair(service, session, owner, now=now)
    step = timedelta(seconds=service.settings.device_pairing_poll_interval_seconds)

    for attempt in range(service.settings.device_pairing_max_attempts):
        result = await service.collect_token(
            session,
            pairing_id=pairing.id,
            device_code=opaque_tokens.new_secret(),
            now=now + step * (attempt + 1),
        )
        assert result.status is PairingStatus.access_denied

    max_attempts = service.settings.device_pairing_max_attempts
    assert repo.pairings[pairing.id].attempts == max_attempts
    assert repo.pairings[pairing.id].denied_at is not None

    # Further wrong guesses saturate rather than overflow the SmallInteger.
    for extra in range(3):
        await service.collect_token(
            session,
            pairing_id=pairing.id,
            device_code=opaque_tokens.new_secret(),
            now=now + step * (max_attempts + extra + 1),
        )
    assert repo.pairings[pairing.id].attempts == max_attempts

    # The row is burned: even the correct code no longer redeems it.
    final = await service.collect_token(
        session,
        pairing_id=pairing.id,
        device_code=device_code,
        now=now + step * 20,
    )
    assert final.status is PairingStatus.access_denied
    assert repo.devices[pairing.device_id].token_hash == ""


async def test_polling_faster_than_the_interval_returns_slow_down(
    service, session, repo, owner
) -> None:
    """The poll is not under throttle_auth_attempts, so cadence lives on the row."""
    now = datetime.now(UTC)
    started = await service.start_pairing(
        session,
        device_name="laptop",
        platform="linux-x86_64",
        helper_version="0.2.0",
        capabilities={},
        request_ip="203.0.113.11",
        now=now,
    )
    first = await service.collect_token(
        session, pairing_id=started.pairing_id, device_code=started.device_code, now=now
    )
    assert first.status is PairingStatus.authorization_pending

    hasty = await service.collect_token(
        session,
        pairing_id=started.pairing_id,
        device_code=started.device_code,
        now=now + timedelta(milliseconds=200),
    )
    assert hasty.status is PairingStatus.slow_down
    assert hasty.interval_seconds > first.interval_seconds
    # A slow_down must not consume an attempt.
    assert repo.pairings[started.pairing_id].attempts == 0


async def test_successful_poll_extends_the_pairing_up_to_the_lifetime_cap(
    service, session, repo
) -> None:
    now = datetime.now(UTC)
    started = await service.start_pairing(
        session,
        device_name="laptop",
        platform="linux-x86_64",
        helper_version="0.2.0",
        capabilities={},
        request_ip="203.0.113.12",
        now=now,
    )
    pairing = repo.pairings[started.pairing_id]
    original_expiry = pairing.expires_at

    later = now + timedelta(seconds=service.settings.device_pairing_ttl_seconds - 60)
    await service.collect_token(
        session, pairing_id=pairing.id, device_code=started.device_code, now=later
    )
    assert pairing.expires_at > original_expiry

    # Keep polling inside the TTL - the only way a pairing survives - and the
    # hard lifetime cap still lands.
    step = timedelta(seconds=service.settings.device_pairing_ttl_seconds - 60)
    for tick in range(2, 12):
        await service.collect_token(
            session, pairing_id=pairing.id, device_code=started.device_code, now=now + step * tick
        )
    assert pairing.expires_at <= pairing.created_at + timedelta(
        seconds=service.settings.device_pairing_max_lifetime_seconds
    )


async def test_the_live_pairing_ceiling_is_global_not_per_ip(repo, session) -> None:
    """The cap must not be keyed on an address the platform cannot trust.

    Behind a proxy that is not allowlisted, every caller presents the edge's
    address, so a per-IP cap of three would have let one unauthenticated client
    block pairing for the entire platform. The replacement is a ceiling high
    enough that the route's own throttle keeps it out of reach.
    """
    service = DevicePairingService(
        repository=repo, settings=_settings(DEVICE_PAIRING_MAX_LIVE_TOTAL="2")
    )
    now = datetime.now(UTC)

    async def start(request_ip: str) -> None:
        await service.start_pairing(
            session,
            device_name="laptop",
            platform="linux-x86_64",
            helper_version="0.2.0",
            capabilities={},
            request_ip=request_ip,
            now=now,
        )

    # Distinct addresses share one budget: the bound is on the table, not the caller.
    await start("203.0.113.13")
    await start("198.51.100.13")
    with pytest.raises(ConflictError):
        await start("192.0.2.13")

    # And no setting keyed on the client address survives.
    assert not hasattr(service.settings, "device_pairing_max_live_per_ip")
    # The shipped default is a backstop, not a budget anyone can exhaust.
    assert _settings().device_pairing_max_live_total >= 1000


async def test_finished_pairings_are_swept(service, session, repo, owner) -> None:
    """helper_pairings is written by an unauthenticated route and must not grow."""
    now = datetime.now(UTC)
    retention = timedelta(seconds=service.settings.device_pairing_retention_seconds)

    expired, _ = await _pair(service, session, owner, now=now, request_ip="203.0.113.40")
    consumed, consumed_code = await _pair(
        service, session, owner, now=now, device_name="two", request_ip="203.0.113.41"
    )
    await service.collect_token(session, pairing_id=consumed.id, device_code=consumed_code, now=now)
    denied, _ = await _pair(
        service, session, owner, now=now, device_name="three", request_ip="203.0.113.42"
    )
    denied.denied_at = now
    assert len(repo.pairings) == 3

    # A later pairing request pays for the cleanup; there is no background loop
    # on the serverless API to put it in. Retention runs from the moment a row
    # stopped being actionable, so wait out the lifetime cap as well.
    later = (
        now
        + retention
        + timedelta(seconds=service.settings.device_pairing_max_lifetime_seconds + 1)
    )
    await service.start_pairing(
        session,
        device_name="fresh",
        platform="linux-x86_64",
        helper_version="0.2.0",
        capabilities={},
        request_ip="203.0.113.43",
        now=later,
    )

    assert expired.id not in repo.pairings
    assert consumed.id not in repo.pairings
    assert denied.id not in repo.pairings
    assert len(repo.pairings) == 1
    # Sweeping a consumed pairing must never take the device it created with it.
    assert repo.devices[consumed.device_id].token_hash != ""


# ---------------------------------------------------------------------------
# Consent
# ---------------------------------------------------------------------------


async def test_verification_token_travels_in_the_url_fragment(service, session) -> None:
    started = await service.start_pairing(
        session,
        device_name="laptop",
        platform="darwin-arm64",
        helper_version="0.2.0",
        capabilities={},
        request_ip="203.0.113.14",
    )
    url, _, fragment = started.verification_url.partition("#")
    assert url == "https://app.nomikos.test/pair"
    assert "?" not in started.verification_url
    assert fragment and fragment not in url


async def test_approve_requires_the_verification_token(service, session, owner) -> None:
    started = await service.start_pairing(
        session,
        device_name="laptop",
        platform="darwin-arm64",
        helper_version="0.2.0",
        capabilities={},
        request_ip="203.0.113.15",
    )
    with pytest.raises(NotFoundError):
        await service.approve_pairing(
            session,
            user=owner,
            pairing_id=started.pairing_id,
            verification_token=opaque_tokens.new_secret(),
        )


async def test_lookup_hides_consumed_expired_and_denied_pairings(
    service, session, repo, owner
) -> None:
    now = datetime.now(UTC)
    started = await service.start_pairing(
        session,
        device_name="laptop",
        platform="darwin-arm64",
        helper_version="0.2.0",
        capabilities={},
        request_ip="203.0.113.16",
        now=now,
    )
    verification_token = started.verification_url.split("#", 1)[1]
    found = await service.lookup_pairing(session, verification_token=verification_token, now=now)
    assert found.id == started.pairing_id

    await service.deny_pairing(
        session,
        user=owner,
        pairing_id=started.pairing_id,
        verification_token=verification_token,
        now=now,
    )
    with pytest.raises(NotFoundError):
        await service.lookup_pairing(session, verification_token=verification_token, now=now)


async def test_helper_supplied_strings_are_normalised(service, session, repo, owner) -> None:
    """Consent-screen text is attacker-controlled; strip control characters."""
    now = datetime.now(UTC)
    started = await service.start_pairing(
        session,
        device_name="Evil‮laptop\x00\n",
        platform="darwin-arm64",
        helper_version="0.2.0",
        capabilities={},
        request_ip="203.0.113.17",
        now=now,
    )
    pairing = repo.pairings[started.pairing_id]
    assert "‮" not in pairing.requested_name
    assert "\x00" not in pairing.requested_name
    assert "\n" not in pairing.requested_name
    assert len(pairing.requested_name) <= 120


async def test_device_limit_per_user_is_enforced(service, session, owner) -> None:
    now = datetime.now(UTC)
    for index in range(service.settings.device_max_per_user):
        await _pair(
            service,
            session,
            owner,
            now=now,
            device_name=f"laptop-{index}",
            request_ip=f"203.0.113.{100 + index}",
        )
    with pytest.raises(ConflictError, match="paired computers"):
        await _pair(
            service,
            session,
            owner,
            now=now,
            device_name="one-too-many",
            request_ip="203.0.113.200",
        )


# ---------------------------------------------------------------------------
# Renewal
# ---------------------------------------------------------------------------


async def test_renewal_keeps_the_previous_token_valid_during_the_overlap(
    service, session, repo, owner, management
) -> None:
    """A UI-less helper that loses a renewal response must not be bricked."""
    now = datetime.now(UTC)
    pairing, device_code = await _pair(service, session, owner, now=now)
    original = (
        await service.collect_token(
            session, pairing_id=pairing.id, device_code=device_code, now=now
        )
    ).device_token
    device = repo.devices[pairing.device_id]

    renewed = await management.renew_token(session, device=device, now=now)
    assert renewed.device_token != original

    for token in (original, renewed.device_token):
        resolved = await authenticate_device(
            session, token, repository=repo, settings=service.settings, now=now
        )
        assert resolved.user.id == owner.id

    after_overlap = now + timedelta(
        hours=service.settings.device_token_renew_overlap_hours, minutes=1
    )
    with pytest.raises(InvalidCredentialsError):
        await authenticate_device(
            session, original, repository=repo, settings=service.settings, now=after_overlap
        )
    still_valid = await authenticate_device(
        session, renewed.device_token, repository=repo, settings=service.settings, now=after_overlap
    )
    assert still_valid.device.id == device.id


# ---------------------------------------------------------------------------
# Secret containment
# ---------------------------------------------------------------------------


async def test_raw_token_is_never_stored_on_the_device_row(service, session, repo, owner) -> None:
    now = datetime.now(UTC)
    pairing, device_code = await _pair(service, session, owner, now=now)
    issued = await service.collect_token(
        session, pairing_id=pairing.id, device_code=device_code, now=now
    )
    raw = issued.device_token
    secret = raw.split(".", 2)[2]

    device = repo.devices[pairing.device_id]
    stored = {
        column.name: getattr(device, column.name) for column in HelperDevice.__table__.columns
    }
    for name, value in stored.items():
        assert secret not in str(value), f"raw secret leaked into helper_devices.{name}"
    assert device.token_hash == opaque_tokens.hash_secret(secret, service.settings.hmac_key())
    assert device.token_prefix == f"nmd1.{device.id.hex[:8]}"
    assert secret not in device.token_prefix

    pairing_columns = {
        column.name: getattr(pairing, column.name) for column in HelperPairing.__table__.columns
    }
    for name, value in pairing_columns.items():
        assert device_code not in str(value), f"raw device_code leaked into helper_pairings.{name}"


async def test_read_dto_exposes_no_secret_material(service, session, repo, owner) -> None:
    now = datetime.now(UTC)
    pairing, device_code = await _pair(service, session, owner, now=now)
    issued = await service.collect_token(
        session, pairing_id=pairing.id, device_code=device_code, now=now
    )
    device = repo.devices[pairing.device_id]

    payload = device_response_from_orm(device, service=service, now=now).model_dump()
    serialized = str(payload)
    assert issued.device_token not in serialized
    assert device.token_hash not in serialized
    assert not {"token_hash", "previous_token_hash", "device_code", "device_token"} & set(payload)
    assert payload["token_prefix"] == device.token_prefix


# ---------------------------------------------------------------------------
# Device liveness
# ---------------------------------------------------------------------------


def test_device_dependency_uses_a_dedicated_header_and_type() -> None:
    """A device token must be structurally unable to satisfy get_current_user.

    Restored after a test-suite reduction pass cut it. The integration suite
    covers the *reverse* direction - ``test_device_claim.py`` presents a browser
    bearer as an agent credential and gets 401 - but nothing anywhere presents a
    device token as a user credential. This is the only record of that direction,
    and it is the dangerous one: a device token is long-lived, sits on a
    researcher's laptop and is scoped to claiming pages, so a fallback added to
    ``get_current_user`` would quietly promote every agent to a full session.
    """
    from backend.ml.application.device_auth import DEVICE_TOKEN_HEADER, AuthenticatedDevice
    from backend.users.api.dependencies import get_current_device, get_current_user

    assert DEVICE_TOKEN_HEADER == "X-Nomikos-Device-Token"
    signature = inspect.signature(get_current_device)
    assert "x_nomikos_device_token" in signature.parameters
    assert signature.return_annotation is AuthenticatedDevice

    # get_current_user is untouched: still HTTPBearer -> User.
    assert inspect.signature(get_current_user).return_annotation is User
    assert "x_nomikos_device_token" not in inspect.signature(get_current_user).parameters


# ``test_device_migration_matches_the_orm_models`` and
# ``test_orm_models_are_registered_for_alembic_metadata`` stood here. Both are
# subsumed by ``integration/test_migrations.py::test_migration_chain_matches_orm_metadata``,
# which migrates a scratch database from empty to head and asserts alembic's
# autogenerate finds no difference against ``Base.metadata`` - every table, not
# just these two, against a real Postgres rather than a monkeypatched ``op``, and
# without naming a revision, so it survives the chain being renumbered.


def test_device_migration_grants_the_runtime_role_access() -> None:
    """A table the API role cannot write is an outage on the first request.

    Restored after a test-suite reduction pass cut it: nothing else in the suite
    asserts the helper-device tables are reachable by ``nomikos_api``. The
    chain-vs-ORM check named above covers *shape*, not *privilege* - it migrates
    as a superuser, so a chain that creates both tables and grants nothing passes
    it and then 500s on the first pairing request in production.
    """
    from pathlib import Path

    migration = (
        Path(__file__).resolve().parents[3]
        / "nomikos"
        / "infrastructure"
        / "alembic"
        / "versions"
        / "003_helper_devices.py"
    ).read_text()

    assert "nomikos_api" in migration
    assert "GRANT SELECT, INSERT, UPDATE, DELETE" in migration
    for table in ("helper_devices", "helper_pairings"):
        assert table in migration.split("_grant_runtime_privileges")[1]


async def test_touch_device_records_liveness(service, session, repo, owner, management) -> None:
    now = datetime.now(UTC)
    pairing, device_code = await _pair(service, session, owner, now=now)
    await service.collect_token(session, pairing_id=pairing.id, device_code=device_code, now=now)
    device = repo.devices[pairing.device_id]
    assert service.device_status(device, now=now) is DeviceStatus.offline

    await management.touch_device(session, device=device, request_ip="198.51.100.7", now=now)
    assert service.device_status(device, now=now) is DeviceStatus.online
    assert device.last_seen_ip == "198.51.100.7"

    stale = now + timedelta(seconds=service.settings.device_idle_window_seconds + 1)
    assert service.device_status(device, now=stale) is DeviceStatus.offline


# ---------------------------------------------------------------------------
# Router mounting
# ---------------------------------------------------------------------------


DEVICE_ROUTES = {
    ("POST", "/device/v1/pairings"),
    ("POST", "/device/v1/pairings/token"),
    ("POST", "/devices/pairings/lookup"),
    ("POST", "/devices/pairings/{pairing_id}/approve"),
    ("POST", "/devices/pairings/{pairing_id}/deny"),
    ("GET", "/devices"),
    ("DELETE", "/devices/{device_id}"),
    ("GET", "/device/v1/self"),
    ("POST", "/device/v1/token/renew"),
}


def _app_routes() -> set[tuple[str, str]]:
    from backend.core.app import create_app

    return {
        (method, path)
        for path, operations in create_app().openapi()["paths"].items()
        for method in (verb.upper() for verb in operations)
    }


def test_device_routes_are_mounted_on_the_real_app() -> None:
    """The phase shipped once with none of these mounted and every test passing.

    The integration suite builds its own FastAPI app, so it proved the routers
    work without proving anything reaches them. This asserts against the app the
    deployment actually serves.
    """
    assert _app_routes() >= DEVICE_ROUTES


def test_the_ip_scoped_pairing_recovery_route_is_gone() -> None:
    """It filtered on request_ip alone, because a pairing has no owner yet.

    Behind an unallowlisted proxy that predicate matches every row, so the route
    listed every user's live pairing requests, pairing_id included.

    Restored after a test-suite reduction pass removed it as an "asserts a
    deleted thing stays deleted" test. That reasoning holds for retired
    features; it does not hold here. This is the only record that the repository
    methods below were removed because they leaked across tenants, and they are
    exactly the kind of helper someone re-adds while restoring a listing screen.
    """
    assert ("GET", "/devices/pairings") not in _app_routes()

    from backend.ml.application.pairing_service import DevicePairingService as _Service

    assert not hasattr(_Service, "list_pairings_from_ip")
    from backend.ml.infrastructure.device_repository import HelperDeviceRepository

    assert not hasattr(HelperDeviceRepository, "list_live_pairings_for_ip")
    assert not hasattr(HelperDeviceRepository, "count_live_pairings_for_ip")


# ---------------------------------------------------------------------------
# Feature flag
# ---------------------------------------------------------------------------


def test_pairing_defaults_off_in_production_and_on_elsewhere() -> None:
    """No /pair page exists yet; production must not serve the consent flow."""
    production = _settings(ENVIRONMENT="production")
    assert production.pairing_enabled() is False
    assert _settings(ENVIRONMENT="development").pairing_enabled() is True
    # And it is an explicit dial in both directions, without a redeploy.
    assert _settings(ENVIRONMENT="production", DEVICE_PAIRING_ENABLED="true").pairing_enabled()
    assert not _settings(
        ENVIRONMENT="development", DEVICE_PAIRING_ENABLED="false"
    ).pairing_enabled()


def test_disabled_pairing_404s_before_any_database_work(monkeypatch) -> None:
    from fastapi import HTTPException

    from backend.ml.api import device_dependencies
    from backend.ml.api.device_dependencies import require_device_pairing_enabled

    monkeypatch.setattr(
        device_dependencies,
        "get_device_settings",
        lambda: _settings(ENVIRONMENT="production"),
    )
    with pytest.raises(HTTPException) as excinfo:
        require_device_pairing_enabled()
    assert excinfo.value.status_code == 404

    monkeypatch.setattr(
        device_dependencies,
        "get_device_settings",
        lambda: _settings(DEVICE_PAIRING_ENABLED="true"),
    )
    assert require_device_pairing_enabled() is None


def test_every_device_router_carries_the_feature_gate() -> None:
    """A router added later must not quietly escape the kill switch."""
    from fastapi.routing import APIRoute

    from backend.ml.api.device_dependencies import require_device_pairing_enabled
    from backend.ml.api.device_pairing import router as pairing_router
    from backend.ml.api.device_self import router as self_router
    from backend.ml.api.devices import router as devices_router

    for router in (pairing_router, devices_router, self_router):
        for route in router.routes:
            if not isinstance(route, APIRoute):
                continue
            gates = [
                dependency.call
                for dependency in route.dependant.dependencies
                if dependency.call is require_device_pairing_enabled
            ]
            assert gates, f"{route.path} is not behind the device pairing feature gate"


# ---------------------------------------------------------------------------
# Consent screen: what the researcher can actually check
# ---------------------------------------------------------------------------


async def test_start_and_consent_show_the_same_confirmation_code(service, session, repo) -> None:
    """The one thing the researcher can compare between two screens."""
    started = await service.start_pairing(
        session,
        device_name="laptop",
        platform="darwin-arm64",
        helper_version="0.2.0",
        capabilities={},
        request_ip="203.0.113.30",
    )
    pairing = repo.pairings[started.pairing_id]
    consent = pairing_response_from_orm(pairing, service=service)

    assert consent.confirmation_code == started.confirmation_code
    assert len(started.confirmation_code) == 9 and started.confirmation_code[4] == "-"
    # It is a comparison aid, not a credential: it opens no door and leaks nothing.
    assert started.device_code not in started.confirmation_code
    assert started.verification_url.split("#", 1)[1] not in started.confirmation_code


def test_the_confirmation_code_cannot_be_derived_without_the_server_key() -> None:
    pairing_id = uuid.uuid4()
    ours = opaque_tokens.confirmation_code(pairing_id, HMAC_KEY)
    assert ours == opaque_tokens.confirmation_code(pairing_id, HMAC_KEY)  # stable
    assert ours != opaque_tokens.confirmation_code(pairing_id, "a-different-key-32-bytes-long!!!")
    assert ours != opaque_tokens.confirmation_code(uuid.uuid4(), HMAC_KEY)
    assert set(ours) <= set(opaque_tokens.CONFIRMATION_ALPHABET) | {"-"}
    # No ambiguous glyphs: this gets read off one screen and compared to another.
    assert not set("IO01") & set(ours)


async def test_the_consent_screen_carries_no_untrustworthy_network_signal(
    service, session, repo
) -> None:
    """`same_network` was unconditionally true in production, which is worse than absent."""
    started = await service.start_pairing(
        session,
        device_name="laptop",
        platform="darwin-arm64",
        helper_version="0.2.0",
        capabilities={},
        request_ip="203.0.113.31",
    )
    payload = pairing_response_from_orm(repo.pairings[started.pairing_id], service=service)
    fields = set(payload.model_dump())

    assert "same_network" not in fields
    assert "request_ip" not in fields
    # The row still records the observed address for support correlation.
    assert repo.pairings[started.pairing_id].request_ip == "203.0.113.31"


async def test_the_phishing_window_is_bounded_in_minutes_not_days(service, session, repo) -> None:
    """A transferable consent link must not survive for a day.

    The poller is whoever created the pairing, so an attacker keeps their own
    link alive to the hard cap. That cap - not the TTL - is the exposure.
    """
    settings = service.settings
    assert settings.device_pairing_ttl_seconds <= 300
    assert settings.device_pairing_max_lifetime_seconds <= 900

    now = datetime.now(UTC)
    started = await service.start_pairing(
        session,
        device_name="laptop",
        platform="darwin-arm64",
        helper_version="0.2.0",
        capabilities={},
        request_ip="203.0.113.32",
        now=now,
    )
    pairing = repo.pairings[started.pairing_id]
    # Poll it as an attacker would, well past the cap.
    for minute in range(1, 40):
        await service.collect_token(
            session,
            pairing_id=pairing.id,
            device_code=started.device_code,
            now=now + timedelta(minutes=minute),
        )
    assert pairing.expires_at <= now + timedelta(
        seconds=settings.device_pairing_max_lifetime_seconds
    )


# ---------------------------------------------------------------------------
# Device credential key
# ---------------------------------------------------------------------------


def test_production_refuses_to_key_device_tokens_off_the_jwt_secret(monkeypatch) -> None:
    """A JWT rotation logs browsers out; it must not also unpair every laptop."""
    from pydantic import ValidationError as PydanticValidationError

    from backend.core.settings.auth import get_auth_settings

    monkeypatch.setenv("JWT_SECRET", HMAC_KEY)
    get_auth_settings.cache_clear()
    try:
        with pytest.raises(PydanticValidationError, match="DEVICE_TOKEN_HMAC_SECRET"):
            DeviceSettings(ENVIRONMENT="production", DEVICE_PAIRING_ENABLED="true", _env_file=None)

        with pytest.raises(PydanticValidationError, match="DEVICE_TOKEN_HMAC_SECRET"):
            DeviceSettings(
                ENVIRONMENT="production",
                DEVICE_PAIRING_ENABLED="true",
                DEVICE_TOKEN_HMAC_SECRET=HMAC_KEY,
                _env_file=None,
            )

        # A distinct dedicated secret is accepted.
        accepted = DeviceSettings(
            ENVIRONMENT="production",
            DEVICE_PAIRING_ENABLED="true",
            DEVICE_TOKEN_HMAC_SECRET="a-dedicated-device-secret-at-least-32-bytes",
            _env_file=None,
        )
        assert accepted.hmac_key() != HMAC_KEY
    finally:
        get_auth_settings.cache_clear()


def test_production_only_warns_while_pairing_is_switched_off(monkeypatch, caplog) -> None:
    """Fail closed on the feature, not on every deployment that has not enabled it."""
    from backend.core.settings.auth import get_auth_settings

    monkeypatch.setenv("JWT_SECRET", HMAC_KEY)
    get_auth_settings.cache_clear()
    caplog.set_level(logging.WARNING)
    try:
        settings = DeviceSettings(
            ENVIRONMENT="production", DEVICE_PAIRING_ENABLED="false", _env_file=None
        )
    finally:
        get_auth_settings.cache_clear()

    assert settings.pairing_enabled() is False
    assert "DEVICE_TOKEN_HMAC_SECRET" in caplog.text
    assert HMAC_KEY not in caplog.text


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------


async def test_every_pairing_transition_is_logged(
    service, session, repo, owner, caplog, management
) -> None:
    """Ten modules shipped without one import of logging; this is the regression."""
    caplog.set_level(logging.INFO)
    now = datetime.now(UTC)

    started = await service.start_pairing(
        session,
        device_name="laptop",
        platform="darwin-arm64",
        helper_version="0.2.0",
        capabilities={},
        request_ip="203.0.113.33",
        now=now,
    )
    verification_token = started.verification_url.split("#", 1)[1]
    await service.approve_pairing(
        session,
        user=owner,
        pairing_id=started.pairing_id,
        verification_token=verification_token,
        now=now,
    )
    issued = await service.collect_token(
        session, pairing_id=started.pairing_id, device_code=started.device_code, now=now
    )
    await management.revoke_device(session, user=owner, device_id=issued.device_id, now=now)
    with pytest.raises(InvalidCredentialsError):
        await authenticate_device(
            session, issued.device_token, repository=repo, settings=service.settings, now=now
        )

    for event in (
        "device_pairing_started",
        "device_pairing_approved",
        "device_token_issued",
        "device_revoked",
        "device_auth_rejected",
    ):
        assert event in caplog.text, event
    # Correlatable without being exploitable.
    assert str(started.pairing_id) in caplog.text
    assert str(issued.device_id) in caplog.text
    for secret in (
        started.device_code,
        verification_token,
        issued.device_token,
        issued.device_token.split(".", 2)[2],
    ):
        assert secret not in caplog.text


async def test_a_burned_pairing_is_logged(service, session, repo, owner, caplog) -> None:
    caplog.set_level(logging.INFO)
    now = datetime.now(UTC)
    pairing, _ = await _pair(service, session, owner, now=now, request_ip="203.0.113.34")
    step = timedelta(seconds=service.settings.device_pairing_poll_interval_seconds)

    for attempt in range(service.settings.device_pairing_max_attempts):
        await service.collect_token(
            session,
            pairing_id=pairing.id,
            device_code=opaque_tokens.new_secret(),
            now=now + step * (attempt + 1),
        )

    assert "device_pairing_burned" in caplog.text
    assert "device_pairing_bad_device_code" in caplog.text


async def test_second_renew_keeps_a_live_predecessor(
    service, session, repo, owner, management
) -> None:
    """A lost renewal response must not brick the helper on the second retry."""
    now = datetime.now(UTC)
    pairing, device_code = await _pair(service, session, owner, now=now)
    await service.collect_token(session, pairing_id=pairing.id, device_code=device_code, now=now)
    device = repo.devices[pairing.device_id]

    await management.renew_token(session, device=device, now=now)
    first_predecessor = device.previous_token_hash
    assert first_predecessor

    # The predecessor is still inside its overlap window: rotating again keeps
    # it instead of replacing it with the token just rotated out.
    await management.renew_token(session, device=device, now=now)
    assert device.previous_token_hash == first_predecessor


async def test_approve_counts_the_device_cap_under_the_user_lock(
    service, session, repo, owner
) -> None:
    """Two parallel approves must not both pass the per-user device cap."""
    now = datetime.now(UTC)
    started = await service.start_pairing(
        session,
        device_name="laptop",
        platform="linux",
        helper_version="1",
        capabilities={},
        request_ip="203.0.113.10",
        now=now,
    )
    await service.approve_pairing(
        session,
        user=owner,
        pairing_id=started.pairing_id,
        verification_token=started.verification_url.split("#", 1)[1],
        now=now,
    )

    assert session.executed == [{"key": f"device-cap:{owner.id}"}]
