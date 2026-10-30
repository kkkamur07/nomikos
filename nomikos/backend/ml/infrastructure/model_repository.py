"""ML catalog and binding persistence."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.core.exceptions import ConflictError
from backend.ml.infrastructure.orm_models import (
    InferenceModel,
    InferenceTask,
    ModelBinding,
)


class MlRepository:
    async def list_models(self, session: AsyncSession) -> list[InferenceModel]:
        result = await session.execute(
            select(InferenceModel).order_by(InferenceModel.provider, InferenceModel.name)
        )
        return list(result.scalars().all())

    async def get_model(self, session: AsyncSession, model_id: UUID) -> InferenceModel | None:
        result = await session.execute(select(InferenceModel).where(InferenceModel.id == model_id))
        return result.scalar_one_or_none()

    async def get_binding(self, session: AsyncSession, binding_id: UUID) -> ModelBinding | None:
        result = await session.execute(
            select(ModelBinding)
            .options(selectinload(ModelBinding.model))
            .where(ModelBinding.id == binding_id)
        )
        return result.scalar_one_or_none()

    async def list_bindings(
        self,
        session: AsyncSession,
        *,
        project_id: UUID | None = None,
        document_id: UUID | None = None,
        document_part_id: UUID | None = None,
    ) -> list[ModelBinding]:
        result = await session.execute(
            select(ModelBinding)
            .options(selectinload(ModelBinding.model))
            .where(
                ModelBinding.project_id == project_id,
                ModelBinding.document_id == document_id,
                ModelBinding.document_part_id == document_part_id,
            )
            .order_by(ModelBinding.task, ModelBinding.created_at)
        )
        return list(result.scalars().all())

    async def find_binding(
        self,
        session: AsyncSession,
        *,
        task: InferenceTask,
        project_id: UUID | None = None,
        document_id: UUID | None = None,
        document_part_id: UUID | None = None,
    ) -> ModelBinding | None:
        result = await session.execute(
            select(ModelBinding)
            .options(selectinload(ModelBinding.model))
            .where(
                ModelBinding.task == task,
                ModelBinding.project_id == project_id,
                ModelBinding.document_id == document_id,
                ModelBinding.document_part_id == document_part_id,
            )
            .order_by(ModelBinding.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def create_binding(
        self,
        session: AsyncSession,
        *,
        task: InferenceTask,
        model_id: UUID,
        project_id: UUID | None = None,
        document_id: UUID | None = None,
        document_part_id: UUID | None = None,
        overrides: dict | None = None,
    ) -> ModelBinding:
        binding = ModelBinding(
            task=task,
            model_id=model_id,
            project_id=project_id,
            document_id=document_id,
            document_part_id=document_part_id,
            overrides=overrides or {},
        )
        try:
            session.add(binding)
            await session.commit()
            await session.refresh(binding)
        except IntegrityError as exc:
            # The application-level duplicate check in
            # ``InferenceModelService._require_no_existing_binding`` lost a
            # concurrent create. This maps whatever the database rejects into
            # the same 409 the pre-check raises.
            await session.rollback()
            raise ConflictError("A binding already exists for this scope and task") from exc
        return binding

    async def update_binding(
        self,
        session: AsyncSession,
        binding: ModelBinding,
        *,
        model_id: UUID | None = None,
        overrides: dict | None = None,
    ) -> ModelBinding:
        if model_id is not None:
            binding.model_id = model_id
        if overrides is not None:
            binding.overrides = overrides
        await session.commit()
        await session.refresh(binding)
        return binding

    async def delete_binding(self, session: AsyncSession, binding: ModelBinding) -> None:
        await session.delete(binding)
        await session.commit()
