"""Health endpoint DTOs."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    # ``schema`` is the wire name, but a field called that shadows the deprecated
    # ``BaseModel.schema`` method, so the attribute is ``schema_state`` behind an alias.
    model_config = ConfigDict(populate_by_name=True)

    status: str
    database: Literal["ok", "error"]
    oldest_pending_job_seconds: float | None = Field(
        default=None,
        description=(
            "Age of the oldest job still in `pending`, in seconds. `null` means "
            "the queue is empty, or - on a 503 - that it could not be read. "
            "Reported, never gated on: see the route docstring."
        ),
    )
    schema_state: Literal["ok", "behind", "ahead", "unknown"] | None = Field(
        default=None,
        alias="schema",
        description=(
            "Database revision against the alembic head the code ships: `ok` (equal), "
            "`behind` (an older known revision, which answers 503), `ahead` (a revision "
            "the code does not know, which is the apply-SQL-before-merge window), "
            "`unknown` (either side unreadable). `null` when the database is unreachable."
        ),
    )
    schema_revision: str | None = Field(
        default=None, description="`alembic_version.version_num` in the database, if readable."
    )
    expected_schema_revision: str | None = Field(
        default=None, description="The alembic head shipped with the running code, if readable."
    )
