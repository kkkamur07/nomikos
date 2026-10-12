"""User feedback: one write path, one admin read path.

The write is capped per user (20 notes an hour) with a plain count query, so
no new dependency. The admin read joins the author for the dashboard table.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.users.infrastructure.orm_models import Feedback, User

#: Notes per user per hour before POST /feedback answers 429.
FEEDBACK_HOURLY_LIMIT = 20


async def recent_count(db: AsyncSession, *, user_id: UUID, hours: int = 1) -> int:
    since = datetime.now(UTC) - timedelta(hours=hours)
    result = await db.execute(
        select(func.count())
        .select_from(Feedback)
        .where(Feedback.user_id == user_id, Feedback.created_at >= since)
    )
    return int(result.scalar() or 0)


async def over_limit(db: AsyncSession, *, user_id: UUID) -> bool:
    return await recent_count(db, user_id=user_id) >= FEEDBACK_HOURLY_LIMIT


async def create_feedback(
    db: AsyncSession, *, user_id: UUID, message: str, page_path: str | None
) -> Feedback:
    row = Feedback(user_id=user_id, message=message, page_path=page_path)
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return row


async def list_feedback(db: AsyncSession, *, limit: int) -> list[dict]:
    rows = (
        await db.execute(
            select(Feedback, User.email, User.username)
            .join(User, User.id == Feedback.user_id)
            .order_by(Feedback.created_at.desc())
            .limit(limit)
        )
    ).all()
    return [
        {
            "id": feedback.id,
            "created_at": feedback.created_at,
            "email": email,
            "username": username,
            "message": feedback.message,
            "page_path": feedback.page_path,
        }
        for feedback, email, username in rows
    ]
