from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, Boolean, DateTime, ForeignKey, Index, String, func, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.project.infrastructure.orm_models import project_shared_users
from infrastructure.db import Base

if TYPE_CHECKING:
    from backend.project.infrastructure.orm_models import Project


class User(Base):
    __tablename__ = "users"
    # Uniqueness on lower(email), not the raw column: get_by_email looks up with
    # func.lower(...), so this backs that query and stops case-variant duplicates
    # (Victim@x.com vs victim@x.com) at the database, matching the schema-layer
    # normalisation.
    __table_args__ = (Index("uq_users_email_lower", text("lower(email)"), unique=True),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(255))
    username: Mapped[str] = mapped_column(String(150), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    # "Use my computer when it is available." The whole of **execution target**
    # selection, chosen once at the account level. There is deliberately no
    # per-job toggle: a researcher cannot know which host is faster for a given
    # page, so asking at every action is a decision without a basis - and it is
    # what regrows the three-mode complexity ADR 0002 deletes.
    prefer_local_inference: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="false", default=False
    )
    # Opens ``/admin`` (aggregate usage and database load, nothing per-document).
    # No route writes it: promotion is an UPDATE by whoever operates the database,
    # so an account cannot grant itself the dashboard. See migration 007.
    is_admin: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="false", default=False
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    owned_projects: Mapped[list[Project]] = relationship("Project", back_populates="owner")
    shared_projects: Mapped[list[Project]] = relationship(
        "Project",
        secondary=project_shared_users,
        back_populates="shared_users",
    )
    sessions: Mapped[list[AuthSession]] = relationship(
        "AuthSession", back_populates="user", cascade="all, delete-orphan"
    )


class AuthSession(Base):
    """Durable rotating browser session; only credential hashes are stored."""

    __tablename__ = "auth_sessions"
    __table_args__ = (
        Index("ix_auth_sessions_user_id", "user_id"),
        Index("ix_auth_sessions_expires_at", "expires_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    token_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    csrf_token_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    user: Mapped[User] = relationship("User", back_populates="sessions")


class AuthRateLimitAttempt(Base):
    """One row per auth attempt; used by the shared-state rate limiter."""

    __tablename__ = "auth_rate_limit_attempts"
    __table_args__ = (Index("ix_auth_rate_limit_key_time", "key", "attempted_at"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String(255), nullable=False)
    attempted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
