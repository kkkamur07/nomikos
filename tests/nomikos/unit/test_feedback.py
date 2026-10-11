"""User feedback (ADR 0012): validation, the hourly cap, and the admin guard.

The SQL itself runs against Postgres in the integration lane; what is pinned
here is the part a refactor would quietly break: trimming and length rules,
the 429 past 20 notes an hour, and that only ``is_admin`` reads the table.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from fastapi import HTTPException
from pydantic import ValidationError as PydanticValidationError

from backend.core.api.admin import admin_feedback
from backend.core.api.feedback import FeedbackCreateRequest, submit_feedback
from backend.core.application import feedback_service
from backend.users.api.dependencies import get_current_user, require_admin
from backend.users.infrastructure.orm_models import Feedback, User


def _user(*, is_admin: bool = False) -> User:
    return User(
        id=uuid.uuid4(),
        email="someone@example.com",
        username="someone",
        hashed_password="x",
        prefer_local_inference=False,
        is_admin=is_admin,
        created_at=datetime.now(UTC),
    )


class _ScalarResult:
    def __init__(self, value: object) -> None:
        self._value = value

    def scalar(self) -> object:
        return self._value


class _RowsResult:
    def __init__(self, rows: list) -> None:
        self._rows = rows

    def all(self) -> list:
        return self._rows


class _Session:
    """Canned count for the rate check; records the added feedback row."""

    def __init__(self, *, count: int = 0) -> None:
        self.count = count
        self.added: list[Feedback] = []

    async def execute(self, statement: object, params: dict | None = None) -> _ScalarResult:
        return _ScalarResult(self.count)

    def add(self, row: Feedback) -> None:
        self.added.append(row)

    async def commit(self) -> None:
        return None

    async def refresh(self, row: Feedback) -> None:
        row.id = row.id or uuid.uuid4()
        row.created_at = row.created_at or datetime.now(UTC)


def test_message_is_trimmed_and_must_leave_text() -> None:
    assert FeedbackCreateRequest.model_validate({"message": "  hello  "}).message == "hello"
    with pytest.raises(PydanticValidationError):
        FeedbackCreateRequest.model_validate({"message": "   "})
    with pytest.raises(PydanticValidationError):
        FeedbackCreateRequest.model_validate({"message": ""})


def test_message_longer_than_the_column_is_refused() -> None:
    with pytest.raises(PydanticValidationError):
        FeedbackCreateRequest.model_validate({"message": "x" * 2001})
    assert len(FeedbackCreateRequest.model_validate({"message": "x" * 2000}).message) == 2000


def test_page_path_is_optional_trimmed_and_capped() -> None:
    assert FeedbackCreateRequest.model_validate({"message": "hi"}).page_path is None
    assert (
        FeedbackCreateRequest.model_validate(
            {"message": "hi", "page_path": "  /projects/1  "}
        ).page_path
        == "/projects/1"
    )
    assert (
        FeedbackCreateRequest.model_validate({"message": "hi", "page_path": "   "}).page_path
        is None
    )
    with pytest.raises(PydanticValidationError):
        FeedbackCreateRequest.model_validate({"message": "hi", "page_path": "x" * 513})


async def test_submit_stores_the_note_for_a_signed_in_user() -> None:
    user = _user()
    session = _Session(count=3)
    response = await submit_feedback(
        FeedbackCreateRequest(message="  fix the contrast  ", page_path="/projects/1"),
        user,
        session,  # type: ignore[arg-type]
    )
    assert response.message == "fix the contrast"
    assert response.page_path == "/projects/1"
    assert len(session.added) == 1
    assert session.added[0].user_id == user.id


async def test_submit_answers_429_past_twenty_notes_an_hour() -> None:
    session = _Session(count=20)
    with pytest.raises(HTTPException) as refused:
        await submit_feedback(
            FeedbackCreateRequest(message="one more"),
            _user(),
            session,  # type: ignore[arg-type]
        )
    assert refused.value.status_code == 429
    assert session.added == []


async def test_nineteen_notes_still_go_through() -> None:
    session = _Session(count=19)
    await submit_feedback(
        FeedbackCreateRequest(message="just under"),
        _user(),
        session,  # type: ignore[arg-type]
    )
    assert len(session.added) == 1


async def test_admin_feedback_needs_an_admin() -> None:
    # Route dependencies do not run on a direct call, so the guard is pinned
    # where it lives: require_admin refuses a signed-in non-admin with 403.
    with pytest.raises(HTTPException) as refused:
        await require_admin(_user(is_admin=False))
    assert refused.value.status_code == 403
    assert await require_admin(_user(is_admin=True)) is not None


async def test_admin_feedback_needs_a_signed_in_user() -> None:
    with pytest.raises(HTTPException) as refused:
        await get_current_user(None, _Session())  # type: ignore[arg-type]
    assert refused.value.status_code == 401


async def test_admin_feedback_lists_newest_rows_with_authors() -> None:
    user = _user(is_admin=True)
    first = Feedback(
        id=uuid.uuid4(),
        user_id=user.id,
        message="first",
        page_path="/a",
        created_at=datetime(2026, 10, 10, tzinfo=UTC),
    )
    second = Feedback(
        id=uuid.uuid4(),
        user_id=user.id,
        message="second",
        page_path=None,
        created_at=datetime(2026, 10, 11, tzinfo=UTC),
    )

    class _ListSession(_Session):
        async def execute(self, statement: object, params: dict | None = None) -> _RowsResult:
            return _RowsResult(
                [(second, user.email, user.username), (first, user.email, user.username)]
            )

    rows = await admin_feedback(user, _ListSession(), 50)  # type: ignore[arg-type]
    assert [row.message for row in rows] == ["second", "first"]
    assert rows[0].email == "someone@example.com"
    assert rows[0].username == "someone"
    assert rows[1].page_path == "/a"


def test_hourly_cap_is_twenty() -> None:
    assert feedback_service.FEEDBACK_HOURLY_LIMIT == 20
