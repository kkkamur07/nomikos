"""Auth routes: register, login, refresh, logout, and me."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.settings.auth import get_auth_settings
from backend.users.api.dependencies import get_current_user
from backend.users.api.rate_limit import (
    UNATTRIBUTABLE_AUTH_RATE_LIMIT,
    attributable_client_ip,
    consume_rate_limit,
    throttle_auth_attempts,
)
from backend.users.api.schemas import LoginRequest, RegisterRequest, TokenResponse, UserResponse
from backend.users.application.auth_service import AuthService
from backend.users.application.browser_sessions import BrowserSessionService, BrowserSessionTokens
from backend.users.infrastructure.orm_models import User
from infrastructure.db import get_db

router = APIRouter(tags=["auth"])
_auth = AuthService()


async def _throttle_session_use(request: Request) -> None:
    """Per-IP cap on session rotation; unattributable callers share one bucket.

    Refresh and logout had no limiter at all. The IP dimension carries the
    load where the peer identifies one client; where it does not, the coarse
    shared bucket sheds the excess, which is safe here because rotation never
    locks anyone out of signing in.
    """
    settings = get_auth_settings()
    client_ip = attributable_client_ip(request)
    if client_ip is not None:
        key, limit = f"session:{client_ip}", settings.auth_rate_limit_requests
    else:
        key, limit = "session:unattributable", UNATTRIBUTABLE_AUTH_RATE_LIMIT
    await consume_rate_limit(
        [key],
        limit=limit,
        window_seconds=settings.auth_rate_limit_window_seconds,
        detail="Too many requests; try again later",
    )


def _set_session_cookies(response: Response, tokens: BrowserSessionTokens) -> None:
    settings = get_auth_settings()
    max_age = settings.session_expire_days * 24 * 60 * 60
    response.set_cookie(
        settings.session_cookie_name,
        tokens.session_cookie,
        max_age=max_age,
        secure=True,
        httponly=True,
        samesite=settings.cookie_same_site,
        path="/",
    )
    response.set_cookie(
        settings.csrf_cookie_name,
        tokens.csrf_token,
        max_age=max_age,
        secure=True,
        httponly=False,
        samesite=settings.cookie_same_site,
        path="/",
        domain=settings.csrf_cookie_domain,
    )


def _issue_session(response: Response, tokens: BrowserSessionTokens) -> TokenResponse:
    """Hand a freshly issued session to the browser, by cookie and by body.

    The two deliveries are made together, in one place, so that a future route
    cannot set the cookies and forget the body copy of the CSRF token (or the
    reverse). See ``TokenResponse.csrf_token`` for why there are two.
    """
    _set_session_cookies(response, tokens)
    return TokenResponse(access_token=tokens.access_token, csrf_token=tokens.csrf_token)


def _clear_session_cookies(response: Response) -> None:
    settings = get_auth_settings()
    response.delete_cookie(
        settings.session_cookie_name,
        secure=True,
        httponly=True,
        samesite=settings.cookie_same_site,
        path="/",
    )
    response.delete_cookie(
        settings.csrf_cookie_name,
        secure=True,
        httponly=False,
        samesite=settings.cookie_same_site,
        path="/",
        domain=settings.csrf_cookie_domain,
    )


@router.post("/auth/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(
    body: RegisterRequest,
    _rate_limit: Annotated[None, Depends(throttle_auth_attempts)],
    db: Annotated[AsyncSession, Depends(get_db)],
    response: Response,
) -> TokenResponse:
    user, _token = await _auth.register(
        db,
        email=body.email,
        username=body.username,
        password=body.password,
    )
    tokens = await BrowserSessionService(get_auth_settings()).create(db, user)
    return _issue_session(response, tokens)


@router.post("/auth/login", response_model=TokenResponse)
async def login(
    body: LoginRequest,
    _rate_limit: Annotated[None, Depends(throttle_auth_attempts)],
    db: Annotated[AsyncSession, Depends(get_db)],
    response: Response,
) -> TokenResponse:
    user, _token = await _auth.login(db, email=body.email, password=body.password)
    tokens = await BrowserSessionService(get_auth_settings()).create(db, user)
    return _issue_session(response, tokens)


@router.post("/auth/refresh", response_model=TokenResponse)
async def refresh(
    request: Request,
    response: Response,
    db: Annotated[AsyncSession, Depends(get_db)],
    _rate_limit: Annotated[None, Depends(_throttle_session_use)],
) -> TokenResponse:
    settings = get_auth_settings()
    tokens = await BrowserSessionService(settings).rotate(
        db,
        session_cookie=request.cookies.get(settings.session_cookie_name),
        csrf_header=request.headers.get("X-CSRF-Token"),
    )
    if tokens is None:
        _clear_session_cookies(response)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session expired")
    return _issue_session(response, tokens)


@router.post("/auth/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    request: Request,
    response: Response,
    db: Annotated[AsyncSession, Depends(get_db)],
    _rate_limit: Annotated[None, Depends(_throttle_session_use)],
) -> None:
    settings = get_auth_settings()
    await BrowserSessionService(settings).revoke(
        db,
        session_cookie=request.cookies.get(settings.session_cookie_name),
        csrf_header=request.headers.get("X-CSRF-Token"),
    )
    _clear_session_cookies(response)
    response.status_code = status.HTTP_204_NO_CONTENT


@router.get("/me", response_model=UserResponse)
async def me(current_user: Annotated[User, Depends(get_current_user)]) -> UserResponse:
    return UserResponse.model_validate(current_user)
