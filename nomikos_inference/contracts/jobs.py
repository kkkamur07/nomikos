"""Async job submit and completion callback contracts."""

from __future__ import annotations

from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from nomikos_inference.admission import validate_job_payload
from nomikos_inference.contracts.common import ImageBytes, InferenceJobStatus, InferenceTask
from nomikos_inference.contracts.segment import SegmentRunResponse
from nomikos_inference.contracts.transcribe import TranscribeBatchRunResponse
from nomikos_inference.settings import get_inference_settings

SUPPORTED_JOB_TASKS = frozenset({InferenceTask.segment, InferenceTask.transcribe})


class JobSubmitRequest(BaseModel):
    task: InferenceTask
    registry_model_id: str = Field(min_length=1)
    registry_tag: str = Field(default="stable", min_length=1)
    product_job_id: UUID
    image_bytes: ImageBytes
    params: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_supported_task(self) -> JobSubmitRequest:
        if self.task not in SUPPORTED_JOB_TASKS:
            raise ValueError(f"unsupported job task: {self.task.value}")
        if self.task == InferenceTask.transcribe:
            lines = self.params.get("lines")
            if not isinstance(lines, list) or not lines:
                raise ValueError("transcribe jobs require non-empty params.lines")
        validate_job_payload(self.image_bytes, self.params, get_inference_settings())
        return self


class SegmentJobOutput(BaseModel):
    kind: Literal["segment"]
    data: SegmentRunResponse


class TranscribeJobOutput(BaseModel):
    kind: Literal["transcribe"]
    data: TranscribeBatchRunResponse


JobOutput = Annotated[
    SegmentJobOutput | TranscribeJobOutput,
    Field(discriminator="kind"),
]


class JobCallbackRequest(BaseModel):
    inference_job_id: UUID
    product_job_id: UUID
    task: InferenceTask
    status: InferenceJobStatus
    output: JobOutput | None = None
    error: str | None = None

    # These are request-body contract errors. FastAPI returns them as 422
    # validation responses when this model is used as an endpoint body.
    @model_validator(mode="after")
    def validate_terminal_payload(self) -> JobCallbackRequest:
        if self.task not in SUPPORTED_JOB_TASKS:
            raise ValueError(f"unsupported job task: {self.task.value}")

        if self.status == InferenceJobStatus.done:
            if self.output is None:
                raise ValueError("done callbacks require structured output")
            if self.error is not None:
                raise ValueError("done callbacks must not include error")
            if self.output.kind != self.task.value:
                raise ValueError(f"{self.task.value} task requires {self.task.value} output")
            return self

        if self.status == InferenceJobStatus.failed:
            if not self.error:
                raise ValueError("failed callbacks require error message")
            if self.output is not None:
                raise ValueError("failed callbacks must not include output")
            return self
        raise ValueError("callback status must be done or failed")


__all__ = [
    "JobCallbackRequest",
    "JobOutput",
    "JobSubmitRequest",
    "SUPPORTED_JOB_TASKS",
    "SegmentJobOutput",
    "TranscribeJobOutput",
]
