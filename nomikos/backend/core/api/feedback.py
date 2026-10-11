"""Short feedback notes from any signed-in user.

One button in the top navigation opens a small dialog; this route takes the
note, the app route the user was on, and stores both. The message is trimmed
and must leave 1 to 2000 characters, else 422. Each user is capped at 20
notes an hour (a plain count query, 429 past that) so a stuck client cannot
flood the admin table.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.application import feedback_service
from backend.users.api.dependencies import get_current_user
from backend.users.infrastructure.orm_models import User
from infrastructure.db import get_db

router = APIRouter(tags=["feedback"])


class FeedbackCreateRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    page_path: str | None = Field(default=None, max_length=512)

    @field_validator("message", mode="before")
    @classmethod
    def _trim_message(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("page_path", mode="before")
    @classmethod
    def _trim_page_path(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        return stripped or None


class FeedbackResponse(BaseModel):
    id: UUID
    created_at: datetime
    message: str
    page_path: str | None = None


@router.post("/feedback", response_model=FeedbackResponse, status_code=201)
async def submit_feedback(
    body: FeedbackCreateRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> FeedbackResponse:
    if await feedback_service.over_limit(db, user_id=current_user.id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many feedback notes; try again later",
        )
    row = await feedback_service.create_feedback(
        db, user_id=current_user.id, message=body.message, page_path=body.page_path
    )
    return FeedbackResponse.model_validate(row, from_attributes=True)
