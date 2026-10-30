"""Shared pytest fixtures — integration tests against live Postgres and the real app."""

import os
import time
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError

# Safe above the environment setup below: `helpers` imports nothing from the
# backend at module scope, precisely so that this line cannot resolve a settings
# object before the environment it reads has been set.
from tests.nomikos.integration.helpers import DEVICE_SERVICE_TOKEN

os.environ.setdefault("JWT_SECRET", "test-secret-not-for-production-at-least-32-bytes")
os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:dev@localhost:5433/kalamos",
)
os.environ.setdefault(
    "SYNC_DATABASE_URL",
    "postgresql://postgres:dev@localhost:5433/kalamos",
)
os.environ.setdefault("AUTH_RATE_LIMIT_REQUESTS", "1000")
os.environ.setdefault("ENABLE_TEST_JOB_ROUTES", "true")
os.environ.setdefault("JOB_WORKER_ENABLED", "true")
os.environ.setdefault("INFERENCE_WEBHOOK_SECRET", "test-inference-webhook-secret")
# The hosted worker's claim credential. Held to the same 32-character floor as the
# device HMAC key, because unlike a device token it is scoped to no account.
os.environ.setdefault("INFERENCE_WORKER_SERVICE_TOKEN", DEVICE_SERVICE_TOKEN)
os.environ.setdefault(
    "MIGRATOR_DATABASE_URL",
    os.environ.get(
        "MIGRATOR_DATABASE_URL",
        "postgresql://postgres:dev@localhost:5433/kalamos",
    ),
)
os.environ.setdefault(
    "INFERENCE_REGISTRY_PATH",
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../nomikos_inference/registry.yaml")
    ),
)
# The device routers construct their service at import time, so the pairing poll
# cadence has to be collapsed before ``backend.core.app`` is imported below - the
# real 5s interval makes every back-to-back poll in test_device_pairing.py return
# ``slow_down``. It lives here rather than in that module because a test module is
# imported *after* this conftest, which is too late to matter.
#
# Setting it here is also what lets that module use the session-scoped client
# instead of assembling a second app. A second TestClient means a second event
# loop, and the asyncpg pool is bound to the loop that created it - which is what
# made four of these tests fail with "attached to a different loop" (issue #63).
# The cadence itself is covered by unit tests with an explicit clock.
os.environ.setdefault("DEVICE_PAIRING_POLL_INTERVAL_SECONDS", "1")
os.environ.setdefault("DEVICE_PAIRING_APP_ORIGIN", "https://app.nomikos.test")

import infrastructure.models  # noqa: F401 — register all mappers
from backend.core.app import create_app

# Only `reset_settings_caches` is used, and deliberately so: the five individual
# accessors that used to be imported alongside it were the hand-written list it
# replaced, left behind after the switch. Naming them here again suggested this
# module still cared which caches there were, when the whole point of
# `reset_settings_caches` is that it does not have to.
from backend.core.settings import reset_settings_caches
from infrastructure.db import Base, sync_engine

_truncate_engine = sync_engine
_TRUNCATE_ADVISORY_LOCK_ID = 73450123

#: The only hosts this suite is allowed to wipe. Loopback, and nothing else.
LOCAL_DATABASE_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def require_local_database(engine: Engine) -> None:
    """Refuse to touch a database that is not on this machine.

    Everything below TRUNCATEs every mapped table before every test, so the only
    thing separating this suite from a real database is which host
    ``SYNC_DATABASE_URL`` happens to name. The ``os.environ.setdefault`` above is
    not that separation: ``setdefault`` yields to a value already exported in the
    caller's shell, and this repo hands developers a Supabase env file whose
    ``SYNC_DATABASE_URL`` is meant to be exported (see
    ``scripts/platform/migrate_supabase.sh``). Sourcing it and then running
    pytest in the same shell was enough to truncate that database.

    Fails closed: a URL with no readable host is refused too, because a host that
    cannot be identified cannot be shown to be local.
    """
    host = engine.url.host
    normalized = (host or "").strip().strip("[]").casefold()
    if normalized in LOCAL_DATABASE_HOSTS:
        return

    target = engine.url.render_as_string(hide_password=True)
    where = f"host {host!r}" if host else "a URL with no host in it"
    raise RuntimeError(
        f"Refusing to wipe the database at {where}.\n"
        f"  resolved target: {target}\n"
        f"  allowed hosts:   {', '.join(sorted(LOCAL_DATABASE_HOSTS))}\n"
        "The integration suite TRUNCATEs every table before every test, so it "
        "only ever runs against a local Postgres. Something has pointed "
        "SYNC_DATABASE_URL somewhere else - most likely an export still live in "
        "this shell from a Supabase env file (scripts/platform/migrate_supabase.sh "
        "sources one). Run `unset SYNC_DATABASE_URL DATABASE_URL "
        "MIGRATOR_DATABASE_URL` or open a new shell, then re-run: the suite "
        "defaults to postgresql://postgres:dev@localhost:5433/kalamos.\n"
        "If you believe that target IS local, note that only loopback names are "
        "accepted - a hostname or container alias that resolves to 127.0.0.1 is "
        "not, on purpose, because this check must not depend on name resolution."
    )


# Checked once at import so a misdirected suite fails at collection, before any
# fixture has run, rather than at the first test. `_truncate_database` re-checks
# rather than trusting this, so the guard cannot be skipped by calling it
# directly, and so a test that repoints the engine mid-session is still caught.
require_local_database(_truncate_engine)


def _ensure_job_status_cancelled() -> None:
    """Dev/test DBs created before cancelled existed need the enum value."""
    require_local_database(_truncate_engine)
    with _truncate_engine.begin() as connection:
        connection.execute(text("ALTER TYPE job_status ADD VALUE IF NOT EXISTS 'cancelled'"))


def _truncate_database() -> None:
    require_local_database(_truncate_engine)
    table_names = [
        sync_engine.dialect.identifier_preparer.quote(table.name)
        for table in reversed(Base.metadata.sorted_tables)
    ]
    for attempt in range(8):
        try:
            with _truncate_engine.begin() as connection:
                connection.execute(
                    text("SELECT pg_advisory_xact_lock(:lock_id)"),
                    {"lock_id": _TRUNCATE_ADVISORY_LOCK_ID},
                )
                if table_names:
                    connection.execute(
                        text(f"TRUNCATE TABLE {', '.join(table_names)} RESTART IDENTITY CASCADE")
                    )
            return
        except OperationalError as exc:
            if "deadlock" not in str(exc).lower() or attempt == 7:
                raise
            time.sleep(0.1 * (attempt + 1))


@pytest.fixture(scope="session")
def client() -> TestClient:
    """Session TestClient — lifespan runs the platform job worker."""
    _ensure_job_status_cancelled()
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def isolated_platform_state(client: TestClient):
    # Every settings accessor at once. The hand-written list this replaces omitted
    # the storage and device accessors, so a test that repointed STORAGE_BACKEND or
    # any DEVICE_* value leaked it into whichever test ran next.
    reset_settings_caches()

    _truncate_database()

    yield


@pytest.fixture
def unique_user() -> dict[str, str]:
    """Unique credentials dict — used with live register HTTP calls."""
    suffix = uuid.uuid4().hex[:8]
    return {
        "email": f"user-{suffix}@test.kalamos",
        "username": f"user_{suffix}",
        "password": "test-pass-123",
    }


@pytest.fixture
def registered_user(client: TestClient, unique_user: dict[str, str]) -> dict[str, str]:
    """Register a user and return credentials plus access_token."""
    response = client.post(
        "/auth/register",
        json={
            "email": unique_user["email"],
            "username": unique_user["username"],
            "password": unique_user["password"],
        },
    )
    assert response.status_code == 201
    data = unique_user.copy()
    data["access_token"] = response.json()["access_token"]
    return data


@pytest.fixture
def auth_headers(registered_user: dict[str, str]) -> dict[str, str]:
    return {"Authorization": f"Bearer {registered_user['access_token']}"}


def _register_user(client: TestClient, suffix: str) -> dict[str, str]:
    creds = {
        "email": f"user-{suffix}@test.kalamos",
        "username": f"user_{suffix}",
        "password": "test-pass-123",
    }
    response = client.post(
        "/auth/register",
        json={
            "email": creds["email"],
            "username": creds["username"],
            "password": creds["password"],
        },
    )
    assert response.status_code == 201
    creds["access_token"] = response.json()["access_token"]
    return creds


@pytest.fixture
def owner_user(client: TestClient) -> dict[str, str]:
    return _register_user(client, f"owner-{uuid.uuid4().hex[:8]}")


@pytest.fixture
def collaborator_user(client: TestClient) -> dict[str, str]:
    return _register_user(client, f"collab-{uuid.uuid4().hex[:8]}")


@pytest.fixture
def outsider_user(client: TestClient) -> dict[str, str]:
    return _register_user(client, f"outsider-{uuid.uuid4().hex[:8]}")


@pytest.fixture
def owner_headers(owner_user: dict[str, str]) -> dict[str, str]:
    return {"Authorization": f"Bearer {owner_user['access_token']}"}


@pytest.fixture
def collaborator_headers(collaborator_user: dict[str, str]) -> dict[str, str]:
    return {"Authorization": f"Bearer {collaborator_user['access_token']}"}


@pytest.fixture
def outsider_headers(outsider_user: dict[str, str]) -> dict[str, str]:
    return {"Authorization": f"Bearer {outsider_user['access_token']}"}


@pytest.fixture
def owner_project(client: TestClient, owner_headers: dict[str, str]) -> dict:
    """Project owned by ``owner_user``."""
    slug = f"proj-{uuid.uuid4().hex[:8]}"
    response = client.post(
        "/projects",
        headers=owner_headers,
        json={"slug": slug, "name": "Test project"},
    )
    assert response.status_code == 201
    return response.json()
