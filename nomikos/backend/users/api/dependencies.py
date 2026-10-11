"""FastAPI dependencies for authenticated routes."""

from typing import Annotated

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials
from jwt import InvalidTokenError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.exceptions import InvalidCredentialsError
from backend.core.settings.auth import get_auth_settings
from backend.ml.application.device_auth import (
    DEVICE_TOKEN_HEADER,
    AuthenticatedDevice,
    authenticate_device,
)
from backend.users.api.security import bearer_scheme
from backend.users.application.browser_sessions import BrowserSessionService
from backend.users.application.jwt_tokens import decode_access_token
from backend.users.infrastructure.orm_models import User
from infrastructure.db import get_db


def _unauthenticated(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> User:
    """Resolve the bearer token to a user, refusing revoked browser sessions.

    The token is only half the credential: the ``sid`` claim must still name a
    live ``auth_sessions`` row. Logout revokes that row, so an access token
    stolen before logout stops working on the next request rather than at
    ``exp``. Tokens without a ``sid`` are refused outright - nothing that
    reaches a browser is minted without a session.
    """
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise _unauthenticated("Not authenticated")
    settings = get_auth_settings()
    try:
        claims = decode_access_token(credentials.credentials, settings)
    except InvalidTokenError:
        raise _unauthenticated("Invalid or expired token") from None
    if claims.session_id is None:
        raise _unauthenticated("Invalid or expired token")
    user = await BrowserSessionService(settings).resolve_active_session_user(
        db, session_id=claims.session_id, user_id=claims.user_id
    )
    if user is None:
        raise _unauthenticated("Not authenticated")
    return user


async def require_admin(
    current_user: Annotated[User, Depends(get_current_user)],
) -> User:
    """Let only accounts with ``users.is_admin`` through; everyone else gets 403.

    Built on :func:`get_current_user`, so a missing or revoked session is still a
    401 and only an authenticated non-admin sees the 403. The flag is read from
    the row the session resolved to, never from the token, so demoting an account
    takes effect on its next request.
    """
    if not current_user.is_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    return current_user


async def get_current_device(
    db: Annotated[AsyncSession, Depends(get_db)],
    x_nomikos_device_token: Annotated[str | None, Header(alias=DEVICE_TOKEN_HEADER)] = None,
) -> AuthenticatedDevice:
    """Authenticate a paired helper and resolve it to ``(device, user)``.

    Sits alongside :func:`get_current_user` rather than inside it. The two are
    intentionally not interchangeable: this reads a dedicated header and returns
    an :class:`AuthenticatedDevice`, so a device token can neither satisfy a
    ``Depends(get_current_user)`` route nor widen its own scope past the single
    user recorded on ``helper_devices.user_id``.
    """
    try:
        return await authenticate_device(db, x_nomikos_device_token)
    except InvalidCredentialsError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid device token",
            headers={"WWW-Authenticate": DEVICE_TOKEN_HEADER},
        ) from None
