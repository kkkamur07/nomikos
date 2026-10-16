"""Auth integration tests — real Postgres (kalamos)."""

from concurrent.futures import ThreadPoolExecutor

import psycopg2
import pytest
from pydantic import ValidationError

from backend.core.settings import get_infrastructure_settings
from backend.core.settings.auth import AuthSettings, get_auth_settings
from backend.users.api.rate_limit import clear_auth_rate_limit_state
from tests.nomikos.integration.helpers import assert_api_error

pytestmark = pytest.mark.integration


def _session_headers(response) -> dict[str, str]:
    """What a browser that can read the CSRF cookie sends: both cookies, plus the header."""
    settings = get_auth_settings()
    session_cookie = response.cookies.get(settings.session_cookie_name)
    csrf_cookie = response.cookies.get(settings.csrf_cookie_name)
    assert session_cookie
    assert csrf_cookie
    return {
        "Cookie": (
            f"{settings.session_cookie_name}={session_cookie}; "
            f"{settings.csrf_cookie_name}={csrf_cookie}"
        ),
        "X-CSRF-Token": csrf_cookie,
    }


def _headers_without_the_csrf_cookie(response) -> dict[str, str]:
    """What a browser that will not let script read that cookie can still send.

    The header comes from the response *body* instead. The CSRF cookie is left
    out of the request entirely, which is the strongest form of the failure:
    whatever a cookie policy does to it - hides it from script, partitions it,
    drops it - the request still has to work.
    """
    settings = get_auth_settings()
    session_cookie = response.cookies.get(settings.session_cookie_name)
    body_token = response.json()["csrf_token"]
    assert session_cookie
    assert body_token
    return {
        "Cookie": f"{settings.session_cookie_name}={session_cookie}",
        "X-CSRF-Token": body_token,
    }


def _register(client, unique_user):
    response = client.post(
        "/auth/register",
        json={
            "email": unique_user["email"],
            "username": unique_user["username"],
            "password": unique_user["password"],
        },
    )
    assert response.status_code == 201
    return response


# --- Register, login, and /me ---
# Tests happy-path auth and JWT bearer flow. Does not test rate limits or password policy edge cases.


@pytest.mark.integration
def test_register_login_me_success(client, unique_user):
    register = client.post(
        "/auth/register",
        json={
            "email": unique_user["email"],
            "username": unique_user["username"],
            "password": unique_user["password"],
        },
    )
    assert register.status_code == 201
    token = register.json()["access_token"]
    assert register.json()["token_type"] == "bearer"

    login = client.post(
        "/auth/login",
        json={"email": unique_user["email"], "password": unique_user["password"]},
    )
    assert login.status_code == 200
    assert login.json()["access_token"]

    me = client.get("/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    body = me.json()
    assert body["email"] == unique_user["email"]
    assert body["username"] == unique_user["username"]
    assert "id" in body
    assert "created_at" in body


# --- Password storage ---
# Tests bcrypt hashing at rest. Does not test login or token issuance.


@pytest.mark.integration
def test_register_stores_bcrypt_hash(client, unique_user):
    response = client.post(
        "/auth/register",
        json={
            "email": unique_user["email"],
            "username": unique_user["username"],
            "password": unique_user["password"],
        },
    )
    assert response.status_code == 201

    settings = get_infrastructure_settings()
    with psycopg2.connect(settings.sync_database_url) as conn, conn.cursor() as cur:
        cur.execute("SELECT hashed_password FROM users WHERE email = %s", (unique_user["email"],))
        hashed = cur.fetchone()[0]
    assert hashed != unique_user["password"]
    assert hashed.startswith("$2b$") or hashed.startswith("$2a$")
    assert "$12$" in hashed  # bcrypt cost factor 12 (see password.BCRYPT_ROUNDS)


# --- Login and token failures ---
# Tests bad credentials and missing/invalid JWT. Does not test account lockout or email verification.


@pytest.mark.integration
def test_login_wrong_password_fails(client, registered_user):
    response = client.post(
        "/auth/login",
        json={"email": registered_user["email"], "password": "wrong-password-xyz"},
    )
    assert response.status_code == 401
    assert_api_error(
        response,
        code="UNAUTHORIZED",
        message="Authentication failed",
    )


@pytest.mark.integration
def test_me_without_token_returns_401(client):
    response = client.get("/me")
    assert response.status_code == 401


@pytest.mark.integration
def test_me_invalid_token_returns_401(client):
    response = client.get("/me", headers={"Authorization": "Bearer not-a-valid-jwt"})
    assert response.status_code == 401


def test_login_sets_secure_http_only_session_and_csrf_cookies(client, unique_user):
    response = client.post(
        "/auth/register",
        json={
            "email": unique_user["email"],
            "username": unique_user["username"],
            "password": unique_user["password"],
        },
    )

    assert response.status_code == 201
    cookies = "\n".join(response.headers.get_list("set-cookie"))
    assert "__Host-greekocr-session=" in cookies
    assert "HttpOnly" in cookies
    assert "SameSite=lax" in cookies
    assert "Secure" in cookies
    assert "greekocr-csrf=" in cookies


def test_auth_responses_hand_back_the_csrf_token_as_well_as_the_cookie(client, unique_user):
    """Two delivery channels for one value.

    The cookie is set on the API host for ``.nomikos.app`` so that script on
    ``app.nomikos.app`` can read it back into ``X-CSRF-Token``; where a browser
    declines that read, the body copy is the only way the client can build the
    header at all. Both are asserted here, because the body copy is an addition
    and the cookie must keep working exactly as it did.
    """
    settings = get_auth_settings()
    register = _register(client, unique_user)

    login = client.post(
        "/auth/login",
        json={"email": unique_user["email"], "password": unique_user["password"]},
    )
    refresh = client.post("/auth/refresh", headers=_session_headers(login))

    assert refresh.status_code == 200
    for response in (register, login, refresh):
        body_token = response.json()["csrf_token"]
        assert body_token
        assert body_token == response.cookies.get(settings.csrf_cookie_name)
        assert f"{settings.csrf_cookie_name}=" in "\n".join(response.headers.get_list("set-cookie"))
    # And every response carries its own freshly rotated token.
    assert len({r.json()["csrf_token"] for r in (register, login, refresh)}) == 3


def test_refresh_authorises_on_the_body_token_with_no_csrf_cookie_sent(client, unique_user):
    """The candidate fix for the Safari 403, exercised end to end.

    A client that cannot read the CSRF cookie echoes the token the body gave it.
    The server checks that header against the hash it stored when it issued the
    session, which is proof of possession on its own - so the cookie's absence
    from the request is not a reason to refuse.
    """
    login = _register(client, unique_user)

    refresh = client.post("/auth/refresh", headers=_headers_without_the_csrf_cookie(login))

    assert refresh.status_code == 200
    assert refresh.json()["access_token"]
    # The rotated session is a real one: it refreshes again the same way.
    assert (
        client.post("/auth/refresh", headers=_headers_without_the_csrf_cookie(refresh)).status_code
        == 200
    )


def test_a_csrf_cookie_alone_still_authorises_nothing(client, unique_user):
    """What the header check buys that the old cookie-equality check did not.

    A caller who can plant a matching cookie pair but does not hold this
    session's token is refused - which is why dropping the cookie-equality check
    costs nothing. The value below is a well-formed token from a *different*
    session, presented in both places at once.
    """
    login = _register(client, unique_user)
    settings = get_auth_settings()
    other_session = client.post(
        "/auth/login",
        json={"email": unique_user["email"], "password": unique_user["password"]},
    )
    someone_elses_token = other_session.json()["csrf_token"]

    forged = client.post(
        "/auth/refresh",
        headers={
            "Cookie": (
                f"{settings.session_cookie_name}="
                f"{login.cookies.get(settings.session_cookie_name)}; "
                f"{settings.csrf_cookie_name}={someone_elses_token}"
            ),
            "X-CSRF-Token": someone_elses_token,
        },
    )

    assert forged.status_code == 403
    assert forged.json()["error"]["code"] == "FORBIDDEN"


def test_logout_revokes_on_the_body_token_with_no_csrf_cookie_sent(client, unique_user):
    """Signing out is CSRF-checked too, so it needs the same second channel.

    A browser that could not present the cookie would otherwise be unable to
    revoke its session server-side - it would only forget its own credentials.
    """
    login = _register(client, unique_user)

    logout = client.post("/auth/logout", headers=_headers_without_the_csrf_cookie(login))

    assert logout.status_code == 204
    assert client.post("/auth/refresh", headers=_session_headers(login)).status_code == 401


def test_refresh_requires_csrf_and_rotates_single_use_session(client, unique_user):
    login = client.post(
        "/auth/register",
        json={
            "email": unique_user["email"],
            "username": unique_user["username"],
            "password": unique_user["password"],
        },
    )
    old_headers = _session_headers(login)

    missing_csrf = client.post(
        "/auth/refresh",
        headers={"Cookie": old_headers["Cookie"]},
    )
    assert missing_csrf.status_code == 403
    assert missing_csrf.json()["error"]["code"] == "FORBIDDEN"

    refresh = client.post("/auth/refresh", headers=old_headers)
    assert refresh.status_code == 200
    new_headers = _session_headers(refresh)
    assert new_headers["Cookie"] != old_headers["Cookie"]

    replay = client.post("/auth/refresh", headers=old_headers)
    assert replay.status_code == 401
    # A replay revokes the session rather than letting the newer token continue.
    assert client.post("/auth/refresh", headers=new_headers).status_code == 401


def test_logout_revokes_session_and_prevents_refresh_replay(client, unique_user):
    login = client.post(
        "/auth/register",
        json={
            "email": unique_user["email"],
            "username": unique_user["username"],
            "password": unique_user["password"],
        },
    )
    headers = _session_headers(login)

    logout = client.post("/auth/logout", headers=headers)
    assert logout.status_code == 204
    assert "max-age=0" in "\n".join(logout.headers.get_list("set-cookie")).lower()
    assert client.post("/auth/refresh", headers=headers).status_code == 401


def test_sessions_are_isolated_between_users(client, unique_user):
    first = client.post(
        "/auth/register",
        json={
            "email": unique_user["email"],
            "username": unique_user["username"],
            "password": unique_user["password"],
        },
    )
    second_user = {
        "email": "second-" + unique_user["email"],
        "username": "second_" + unique_user["username"],
        "password": unique_user["password"],
    }
    second = client.post("/auth/register", json=second_user)

    first_refresh = client.post("/auth/refresh", headers=_session_headers(first))
    second_refresh = client.post("/auth/refresh", headers=_session_headers(second))
    assert first_refresh.status_code == second_refresh.status_code == 200

    first_me = client.get(
        "/me",
        headers={"Authorization": f"Bearer {first_refresh.json()['access_token']}"},
    )
    second_me = client.get(
        "/me",
        headers={"Authorization": f"Bearer {second_refresh.json()['access_token']}"},
    )
    assert first_me.json()["email"] == unique_user["email"]
    assert second_me.json()["email"] == second_user["email"]


def test_credentialed_cors_allows_only_configured_origins(client):
    origin = "http://localhost:5173"
    allowed = client.options(
        "/auth/refresh",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,x-csrf-token",
        },
    )
    assert allowed.headers["access-control-allow-origin"] == origin
    assert allowed.headers["access-control-allow-credentials"] == "true"
    assert "x-csrf-token" in allowed.headers["access-control-allow-headers"].lower()

    denied = client.options(
        "/auth/refresh",
        headers={
            "Origin": "https://attacker.example",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert "access-control-allow-origin" not in denied.headers


# --- Protected routes ---
# Tests auth is required for member APIs. Does not test per-resource authorization rules.


@pytest.mark.integration
def test_projects_without_auth_returns_401(client):
    response = client.get("/projects")
    assert response.status_code == 401


# --- Auth settings ---
# Tests required env config. Does not hit HTTP or the database.


def test_auth_settings_require_jwt_secret(monkeypatch):
    monkeypatch.delenv("JWT_SECRET", raising=False)
    with pytest.raises(ValidationError):
        AuthSettings(_env_file=None)


# --- Login rate limiting ---
# Tests per-client throttling on /auth/login. Does not test register rate limits or successful logins.


def test_login_rate_limit_returns_429(client, monkeypatch):
    monkeypatch.setenv("AUTH_RATE_LIMIT_REQUESTS", "1")
    monkeypatch.setenv("AUTH_RATE_LIMIT_WINDOW_SECONDS", "60")

    get_auth_settings.cache_clear()
    clear_auth_rate_limit_state()

    first = client.post(
        "/auth/login",
        json={"email": "missing@test.kalamos", "password": "wrong-password-xyz"},
    )
    assert first.status_code == 401

    limited = client.post(
        "/auth/login",
        json={"email": "missing@test.kalamos", "password": "wrong-password-xyz"},
    )
    assert limited.status_code == 429


def test_login_rate_limit_is_atomic_for_parallel_requests(client, monkeypatch):
    monkeypatch.setenv("AUTH_RATE_LIMIT_REQUESTS", "1")
    monkeypatch.setenv("AUTH_RATE_LIMIT_WINDOW_SECONDS", "60")
    get_auth_settings.cache_clear()
    clear_auth_rate_limit_state()

    def login_attempt() -> int:
        return client.post(
            "/auth/login",
            json={"email": "parallel@test.kalamos", "password": "wrong-password-xyz"},
        ).status_code

    with ThreadPoolExecutor(max_workers=2) as executor:
        status_codes = list(executor.map(lambda _unused: login_attempt(), range(2)))

    assert sorted(status_codes) == [401, 429]


def test_auth_rejects_passwords_over_bcrypt_utf8_byte_limit(client, unique_user):
    oversized_password = "é" * 37
    response = client.post(
        "/auth/register",
        json={
            "email": unique_user["email"],
            "username": unique_user["username"],
            "password": oversized_password,
        },
    )

    assert response.status_code == 422
    assert_api_error(
        response,
        code="VALIDATION_ERROR",
        message="Invalid request",
    )
    assert oversized_password not in response.text
    assert response.headers["x-error-id"]
