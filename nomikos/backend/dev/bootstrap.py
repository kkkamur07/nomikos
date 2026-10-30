"""Development-only bootstrap helpers (idempotent, no DB restarts)."""

from __future__ import annotations

import os

from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.settings import get_infrastructure_settings
from backend.users.application.auth_service import AuthService
from backend.users.application.password import hash_password_async

# Must pass Pydantic EmailStr (`.local` / `.test` TLDs are rejected).
DEV_EMAIL = os.environ.get("DEV_USER_EMAIL", "dev@example.com")
DEV_USERNAME = os.environ.get("DEV_USER_USERNAME", "dev")
DEV_PASSWORD = os.environ.get("DEV_USER_PASSWORD", "dev-pass-123")


def require_development_environment() -> None:
    """Refuse to seed the well-known dev account outside development.

    These credentials are published in the repository, so the seed is a backdoor
    anywhere else. The guard lives here rather than only at the call sites so a
    maintenance script or a stray import cannot bypass it.
    """
    settings = get_infrastructure_settings()
    if not settings.is_development:
        raise RuntimeError(
            "Refusing to seed the development user: ENVIRONMENT is "
            f"{settings.environment!r}, not 'development'"
        )


async def ensure_dev_user_exists(session: AsyncSession) -> bool:
    """Create the dev user when missing. Returns True if a user was created."""
    require_development_environment()
    service = AuthService()
    if await service.find_by_email(session, DEV_EMAIL):
        return False
    await service.register(
        session,
        email=DEV_EMAIL,
        username=DEV_USERNAME,
        password=DEV_PASSWORD,
    )
    return True


async def reset_dev_user_password(session: AsyncSession) -> None:
    """Ensure dev user exists and reset its password to the configured dev default."""
    require_development_environment()
    service = AuthService()
    user = await service.find_by_email(session, DEV_EMAIL)
    if user is None:
        await service.register(
            session,
            email=DEV_EMAIL,
            username=DEV_USERNAME,
            password=DEV_PASSWORD,
        )
        return
    user.hashed_password = await hash_password_async(DEV_PASSWORD)
    await session.commit()
