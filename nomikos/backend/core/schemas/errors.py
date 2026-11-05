"""Shared, intentionally minimal public API error response schemas."""

from pydantic import BaseModel, Field


class ApiErrorDetail(BaseModel):
    code: str = Field(description="Stable machine-readable error code")
    message: str = Field(description="Allowlisted user-safe error message")
    ref: str | None = Field(
        default=None,
        description="Correlation id for support / log lookup",
    )
    summary: str | None = Field(
        default=None,
        description=(
            "Short user-safe cause of a server error. In production an allowlisted "
            "reason plus the root exception type, e.g. `database schema out of date "
            "(UndefinedColumnError)`, never the exception text; elsewhere a redacted "
            "first line. Never carries SQL, parameters, credentials, paths or tracebacks"
        ),
    )


class ApiErrorResponse(BaseModel):
    error: ApiErrorDetail
