"""Frontend failure beacon - logging-first observability (no Prometheus)."""

from __future__ import annotations

import logging
import re
import uuid

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field, field_validator

from backend.core.api import request_metrics
from backend.users.api.rate_limit import attributable_client_ip, consume_rate_limit

logger = logging.getLogger(__name__)
router = APIRouter(tags=["observability"])

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
CLIENT_FAILURE_RATE_LIMIT = 30
#: Applied when no address identifies a single client. A beacon is best-effort
#: logging, so a shared ceiling that sheds excess reports is an acceptable
#: outcome here - unlike on login, where a global bucket would lock everyone out.
CLIENT_FAILURE_GLOBAL_RATE_LIMIT = 300
CLIENT_FAILURE_RATE_WINDOW_SECONDS = 60


def clear_client_failure_rate_limit_state() -> None:
    """No-op - state lives in Postgres and is cleared by database truncation in tests."""


def _sanitize_log_field(value: str, *, max_len: int) -> str:
    cleaned = _CONTROL_CHARS.sub(" ", value).replace("\r", " ").replace("\n", " ")
    return cleaned.strip()[:max_len]


async def _throttle_client_failure(request: Request) -> None:
    """Rate-limit before body validation (route dependency).

    Shares the auth limiter's Postgres-backed store: an in-process dict would
    reset on every serverless cold start and wouldn't be shared across workers.
    """
    client_ip = attributable_client_ip(request)
    if client_ip:
        key, limit = f"client-failure:ip:{client_ip}", CLIENT_FAILURE_RATE_LIMIT
    else:
        key, limit = "client-failure:global", CLIENT_FAILURE_GLOBAL_RATE_LIMIT
    await consume_rate_limit(
        [key],
        limit=limit,
        window_seconds=CLIENT_FAILURE_RATE_WINDOW_SECONDS,
        detail="Too many client failure reports; try again later",
    )


class ClientFailureRequest(BaseModel):
    message: str = Field(min_length=1, max_length=500)
    ref: str | None = Field(default=None, max_length=64)
    path: str | None = Field(default=None, max_length=512)
    status: int | None = Field(default=None, ge=100, le=599)
    source: str | None = Field(default=None, max_length=64)

    @field_validator("message", "ref", "path", "source", mode="before")
    @classmethod
    def _reject_control_characters(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        if _CONTROL_CHARS.search(value) or "\r" in value or "\n" in value:
            raise ValueError("control characters are not allowed")
        return value


class ClientFailureResponse(BaseModel):
    accepted: bool = True
    ref: str


@router.post(
    "/client-failures",
    response_model=ClientFailureResponse,
    status_code=202,
    dependencies=[Depends(_throttle_client_failure)],
)
async def report_client_failure(
    body: ClientFailureRequest,
    request: Request,
) -> ClientFailureResponse:
    correlation_id = _sanitize_log_field((body.ref or "").strip(), max_len=64) or uuid.uuid4().hex
    logger.warning(
        "client_failure correlation_id=%s method=%s path=%s ui_path=%s status=%s source=%s message=%s",
        correlation_id,
        request.method,
        request.url.path,
        _sanitize_log_field(body.path or "-", max_len=512),
        body.status if body.status is not None else "-",
        _sanitize_log_field(body.source or "ui", max_len=64),
        _sanitize_log_field(body.message, max_len=200),
    )
    # Also kept in memory for the admin dashboard; this route has no table.
    request_metrics.record_client_failure(
        path=_sanitize_log_field(body.path, max_len=512) if body.path else None,
        status=body.status,
        source=_sanitize_log_field(body.source or "ui", max_len=64),
        message=_sanitize_log_field(body.message, max_len=200),
    )
    return ClientFailureResponse(accepted=True, ref=correlation_id)
