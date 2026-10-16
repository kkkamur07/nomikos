"""Register, login, and user lookup."""

import logging
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.exceptions import ConflictError, InvalidCredentialsError, NotFoundError
from backend.core.settings.auth import AuthSettings, get_auth_settings
from backend.users.application.jwt_tokens import create_access_token
from backend.users.application.password import hash_password_async, verify_password_async
from backend.users.infrastructure.orm_models import User
from backend.users.infrastructure.user_repository import UserRepository

logger = logging.getLogger(__name__)

#: A bcrypt hash of nothing, verified against when the email does not exist so a
#: login for an unknown account costs the same as one for a known account. It is
#: a decoy, not a credential: no password produces it and nothing accepts it.
_DUMMY_PASSWORD_HASH = "$2b$12$t7YSQy5g4YoP4Bfr5DXh0eUg2kUE4qavr20ibunY9EEWibESvTARu"  # noqa: S105

#: One message for every registration conflict. Distinct "email already
#: registered" / "username already taken" responses let an anonymous caller
#: enumerate which accounts exist; the specific reason stays in the log.
REGISTRATION_CONFLICT_MESSAGE = "Registration could not be completed with those details"


class AuthService:
    def __init__(
        self,
        repository: UserRepository | None = None,
        auth_settings: AuthSettings | None = None,
    ) -> None:
        self._repo = repository or UserRepository()
        self._auth_settings = auth_settings or get_auth_settings()

    async def find_by_email(self, session: AsyncSession, email: str) -> User | None:
        return await self._repo.get_by_email(session, email)

    async def register(
        self,
        session: AsyncSession,
        *,
        email: str,
        username: str,
        password: str,
    ) -> tuple[User, str]:
        if await self._repo.get_by_email(session, email):
            logger.info("registration_conflict field=email")
            raise ConflictError(REGISTRATION_CONFLICT_MESSAGE)
        if await self._repo.get_by_username(session, username):
            logger.info("registration_conflict field=username")
            raise ConflictError(REGISTRATION_CONFLICT_MESSAGE)
        user = await self._repo.create(
            session,
            email=email,
            username=username,
            hashed_password=await hash_password_async(password),
        )
        try:
            await session.commit()
        except IntegrityError as exc:
            await session.rollback()
            # Concurrent insert lost the race; the response stays identical to the
            # pre-checks above so timing is the only distinguishable signal.
            logger.info("registration_conflict field=concurrent_insert")
            raise ConflictError(REGISTRATION_CONFLICT_MESSAGE) from exc
        token = create_access_token(user.id, self._auth_settings)
        return user, token

    async def login(
        self,
        session: AsyncSession,
        *,
        email: str,
        password: str,
    ) -> tuple[User, str]:
        user = await self._repo.get_by_email(session, email)
        hash_to_check = user.hashed_password if user is not None else _DUMMY_PASSWORD_HASH
        password_matches = await verify_password_async(password, hash_to_check)
        if user is None or not password_matches:
            raise InvalidCredentialsError("Invalid email or password")
        token = create_access_token(user.id, self._auth_settings)
        return user, token

    async def get_user(self, session: AsyncSession, user_id: UUID) -> User:
        user = await self._repo.get_by_id(session, user_id)
        if user is None:
            raise NotFoundError("User not found")
        return user

    async def register_if_absent(
        self,
        session: AsyncSession,
        *,
        email: str,
        username: str,
        password: str,
    ) -> tuple[User | None, str | None]:
        """Create user when email is free; return (None, None) if already exists."""
        if await self.find_by_email(session, email):
            return None, None
        user, token = await self.register(
            session,
            email=email,
            username=username,
            password=password,
        )
        return user, token
