"""ML model catalog and binding API DTOs."""

import json
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from backend.ml.infrastructure.orm_models import InferenceTask


class InferenceModelResponse(BaseModel):
    id: UUID
    name: str
    provider: str
    task: InferenceTask
    artifact_ref: str
    default_params: dict
    created_at: datetime

    model_config = {"from_attributes": True}


MAX_OVERRIDE_ENTRIES = 64
MAX_OVERRIDE_KEY_LENGTH = 128
#: Overrides are merged into the params an agent executes, so the bound keeps
#: one binding from smuggling an arbitrarily large payload into every job.
MAX_OVERRIDE_BYTES = 16 * 1024


def bound_overrides(value: dict | None) -> dict | None:
    """Reject override maps that exceed the count, key-length or size ceilings."""
    if value is None:
        return None
    if len(value) > MAX_OVERRIDE_ENTRIES:
        raise ValueError(f"at most {MAX_OVERRIDE_ENTRIES} overrides")
    for key in value:
        if len(key) > MAX_OVERRIDE_KEY_LENGTH:
            raise ValueError("override names must be short")
    if len(json.dumps(value, default=str).encode("utf-8")) > MAX_OVERRIDE_BYTES:
        raise ValueError("overrides must fit in 16 KiB of JSON")
    return value


class ModelBindingCreateRequest(BaseModel):
    task: InferenceTask
    model_id: UUID
    overrides: dict = Field(default_factory=dict)

    @field_validator("overrides")
    @classmethod
    def _bound_overrides(cls, value: dict) -> dict:
        return bound_overrides(value) or {}


class ModelBindingUpdateRequest(BaseModel):
    model_id: UUID | None = None
    overrides: dict | None = None

    @field_validator("overrides")
    @classmethod
    def _bound_overrides(cls, value: dict | None) -> dict | None:
        return bound_overrides(value)


class ModelBindingResponse(BaseModel):
    id: UUID
    task: InferenceTask
    model_id: UUID
    project_id: UUID | None
    document_id: UUID | None
    document_part_id: UUID | None
    overrides: dict
    created_at: datetime

    model_config = {"from_attributes": True}


class ResolvedModelBindingResponse(BaseModel):
    binding: ModelBindingResponse
    model: InferenceModelResponse
    effective_params: dict
