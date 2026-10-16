"""Enqueue ML jobs for document parts.

One responsibility: turning "segment this page" or "transcribe these lines" into a
``pending`` row an inference agent will claim. It is the only module in this context that
knows about the ML catalog, and the only one that writes ``jobs``.

Model resolution is the substance behind the small interface: an explicit ``model_id``
wins and contributes its defaults, otherwise the nearest binding (part, then document,
then project) is consulted, and *no* binding is not an error: the job goes out with a
null model and the agent's own default applies. That last part is why ``NotFoundError``
from the resolver is swallowed here rather than propagated.

**Execution target** is fixed here, and only here, because this is the first moment all
three inputs exist at once: the caller's account preference and the **capacity** reading
arrive as an :class:`ExecutionRequest` from the route, and **host eligibility** is only
knowable once the model has been resolved. The request is a value, not a collaborator:
capacity is read once at the top of submission and then carried down, so nothing further
along can re-decide. When no eligible host has capacity this raises instead of writing a
row: a job created for a host nobody claims from has no terminal outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.exceptions import ConflictError, NotFoundError, ValidationError
from backend.document.application.document_access import DocumentAccess
from backend.document.infrastructure.document_repository import DocumentRepository
from backend.jobs.infrastructure.orm_models import Job, JobStatus, JobType
from backend.ml.application.model_hosts import eligible_targets_for_model
from backend.ml.application.model_service import InferenceModelService
from backend.ml.domain.execution import (
    ExecutionDecision,
    ExecutionRequest,
    choose_execution_target,
)
from backend.ml.infrastructure.orm_models import InferenceModel, InferenceTask
from backend.project.infrastructure.project_repository import ProjectRepository
from backend.users.infrastructure.orm_models import User


def _decide_execution(
    execution: ExecutionRequest, model: InferenceModel | None, *, task: InferenceTask
) -> ExecutionDecision:
    return choose_execution_target(execution, eligible=eligible_targets_for_model(model, task=task))


@dataclass(frozen=True)
class _ResolvedModel:
    model: InferenceModel | None
    model_id: UUID | None
    binding_id: UUID | None
    params: dict[str, object]


class DocumentJobEnqueueService:
    def __init__(
        self,
        documents: DocumentRepository | None = None,
        projects: ProjectRepository | None = None,
        access: DocumentAccess | None = None,
        inference_models: InferenceModelService | None = None,
    ) -> None:
        self._documents = documents or DocumentRepository()
        self._projects = projects or ProjectRepository()
        self._access = access or DocumentAccess(documents=self._documents, projects=self._projects)
        self._inference_models = inference_models or InferenceModelService()

    async def enqueue_transcribe_part(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        *,
        execution: ExecutionRequest,
        model_id: UUID | None = None,
        line_ids: list[UUID] | None = None,
    ) -> Job:
        context = await self._access.require_part(session, user, project_id, document_id, part_id)
        document, part = context.document, context.part
        lines = await self._documents.list_part_lines(session, part.id)
        if not lines:
            raise ConflictError("Cannot transcribe a document part without layout lines")
        if line_ids is not None:
            selected_ids = set(line_ids)
            known_ids = {line.id for line in lines}
            if selected_ids - known_ids:
                raise NotFoundError("Line not found")
            if not selected_ids:
                raise ValidationError("At least one line must be selected for transcription")
        resolved = await self._resolve_model(
            session, user, project_id, document_id, part_id, model_id, InferenceTask.transcribe
        )
        decision = _decide_execution(execution, resolved.model, task=InferenceTask.transcribe)
        payload: dict[str, object] = {
            "ml_params": resolved.params,
            "execution": decision.target.value,
        }
        if line_ids is not None:
            payload["line_ids"] = [str(line_id) for line_id in line_ids]
        job = Job(
            type=JobType.transcribe,
            status=JobStatus.pending,
            user_id=user.id,
            document_id=document.id,
            document_part_id=part.id,
            model_id=resolved.model_id,
            binding_id=resolved.binding_id,
            execution_target=decision.target,
            preferred_execution_target=decision.preferred,
            payload=payload,
        )
        session.add(job)
        await session.commit()
        await session.refresh(job)
        return job

    async def enqueue_segment_part(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        *,
        execution: ExecutionRequest,
        model_id: UUID | None = None,
        ml_params: dict[str, object] | None = None,
    ) -> Job:
        context = await self._access.require_part(session, user, project_id, document_id, part_id)
        document, part = context.document, context.part
        resolved = await self._resolve_model(
            session, user, project_id, document_id, part_id, model_id, InferenceTask.segment
        )
        # Caller-supplied params override the model's defaults key by key.
        effective_params = {**resolved.params, **(ml_params or {})}
        decision = _decide_execution(execution, resolved.model, task=InferenceTask.segment)
        job = Job(
            type=JobType.segment,
            status=JobStatus.pending,
            user_id=user.id,
            document_id=document.id,
            document_part_id=part.id,
            model_id=resolved.model_id,
            binding_id=resolved.binding_id,
            execution_target=decision.target,
            preferred_execution_target=decision.preferred,
            payload={"ml_params": effective_params, "execution": decision.target.value},
        )
        session.add(job)
        await session.commit()
        await session.refresh(job)
        return job

    async def _resolve_model(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        model_id: UUID | None,
        task: InferenceTask,
    ) -> _ResolvedModel:
        """The explicit model and its defaults, else the nearest binding, else none."""
        if model_id is not None:
            model = await self._inference_models.get_model_for_task(session, model_id, task)
            return _ResolvedModel(model, model_id, None, dict(model.default_params or {}))
        try:
            resolved = await self._inference_models.resolve_for_part(
                session, user, project_id, document_id, part_id, task=task
            )
        except NotFoundError:
            return _ResolvedModel(None, None, None, {})
        return _ResolvedModel(
            resolved.model,
            resolved.model.id,
            resolved.binding.id,
            dict(resolved.effective_params),
        )
