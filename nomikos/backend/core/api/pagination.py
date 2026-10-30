"""Keyset cursor pagination helpers."""

from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Callable
from datetime import datetime
from typing import TypeVar
from uuid import UUID

from pydantic import BaseModel

from backend.core.exceptions import ValidationError

T = TypeVar("T")
MAX_CURSOR_LENGTH = 1024


class PageCursor(BaseModel):
    created_at: datetime
    id: UUID


def encode_cursor(created_at: datetime, row_id: UUID) -> str:
    payload = {
        "created_at": created_at.isoformat(),
        "id": str(row_id),
    }
    return base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")


def decode_cursor(cursor: str) -> PageCursor:
    if not cursor or len(cursor) > MAX_CURSOR_LENGTH:
        raise ValidationError("Invalid pagination cursor")
    try:
        raw = base64.b64decode(cursor.encode("ascii"), altchars=b"-_", validate=True)
        payload = json.loads(raw.decode("utf-8"))
        return PageCursor(
            created_at=datetime.fromisoformat(payload["created_at"]),
            id=UUID(payload["id"]),
        )
    except (
        UnicodeEncodeError,
        binascii.Error,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        raise ValidationError("Invalid pagination cursor") from exc


def paginate_rows(
    rows: list[T],
    *,
    limit: int,
    created_at_getter: Callable[[T], datetime],
    id_getter: Callable[[T], UUID],
) -> tuple[list[T], str | None]:
    """Slice *limit + 1* rows and emit next cursor when more pages exist."""
    has_more = len(rows) > limit
    page = rows[:limit]
    if not has_more or not page:
        return page, None
    last = page[-1]
    return page, encode_cursor(created_at_getter(last), id_getter(last))
