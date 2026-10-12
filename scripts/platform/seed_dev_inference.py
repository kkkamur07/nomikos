#!/usr/bin/env python3
"""Seed inference models and an optional project-level segment binding.

The editor's model pickers list rows from ``inference_models``, not from
``registry.yaml``, so a registry entry is invisible in dev until this script
puts a row beside it. It seeds four transcribe models (three Calamari and one
PP-OCR) and two segment models (BLLA and PP-OCR detection), and is an upsert:
re-running it after a registry edit
rewrites the existing rows rather than duplicating them.

Each row carries a registry id (what dispatch resolves through
``artifact_ref``) and a short display name (what the pickers print,
``docs/inference/model-catalog-names.md``): ``seg-v1``/``seg-v2`` for the
segmenters, ``greek-v1``, ``armenian-v1``, ``coptic-v1``, ``syriac-v2`` for
the transcribers.

Task and provider are read out of ``registry.yaml`` rather than restated here.
A catalog row whose ``task`` disagrees with the registry entry its
``artifact_ref`` points at is a row that fails at dispatch, and the only way to
be sure the two agree is to have one of them come from the other.

Binds to the project named by ``BINDING_PROJECT_SLUG`` (default:
byzantine-greek-manuscripts) if it exists. Transcribe bindings are skipped by
default since script-specific models get attached per project when available.

Overrides, both optional and both comma-separated ``registry_id[:display_name]``
pairs (an entry without ``:display_name`` keeps the registry id as its name):

* ``DEFAULT_SEGMENT_MODEL`` - the segment rows, and the models the
  project-level segment binding points at (first entry wins).
* ``DEFAULT_TRANSCRIBE_MODEL`` - the transcribe rows to seed *instead of* the
  four defaults, for a dev database that should only see one script.
"""

import asyncio
import os
import uuid
from collections.abc import Sequence

from _bootstrap import ensure_nomikos_on_path
from sqlalchemy import delete, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

ensure_nomikos_on_path()

from backend.jobs.infrastructure.orm_models import Job  # noqa: E402
from backend.ml.infrastructure.orm_models import (  # noqa: E402
    InferenceModel,
    InferenceTask,
    ModelBinding,
)
from backend.project.infrastructure.orm_models import Project  # noqa: E402
from infrastructure import models as _orm_models  # noqa: E402, F401 - register all mappers
from infrastructure.db import system_session  # noqa: E402
from nomikos_inference.contracts.common import RegistryArchitecture  # noqa: E402
from nomikos_inference.registry import load_registry  # noqa: E402


def _parse_entry(part: str) -> tuple[str, str]:
    """Split one ``registry_id[:display_name]`` override entry."""
    registry_id, _, display_name = part.partition(":")
    registry_id = registry_id.strip()
    display_name = display_name.strip()
    return (registry_id, display_name or registry_id)


def _pairs(env_var: str, default: tuple[tuple[str, str], ...]) -> tuple[tuple[str, str], ...]:
    raw = os.environ.get(env_var, "")
    override = tuple(
        _parse_entry(part) for part in (item.strip() for item in raw.split(",")) if part
    )
    return override or default


SEGMENT_MODELS = _pairs(
    "DEFAULT_SEGMENT_MODEL",
    (
        ("blla-segment", "seg-v1"),
        ("ppocr-segment", "seg-v2"),
    ),
)


def artifact_ref_for(registry_id: str) -> str:
    """Return the dispatch ref for a registry id (never a display name)."""
    return f"registry://{registry_id}?tag=stable"


def candidate_names_for(registry_id: str, display_name: str) -> tuple[str, ...]:
    """Return every name a catalog row for this id may still carry."""
    if display_name == registry_id:
        return (registry_id,)
    return (registry_id, display_name)


def split_duplicate_models(
    rows: Sequence[InferenceModel],
    display_name: str,
) -> tuple[InferenceModel | None, list[InferenceModel]]:
    """Keep one of several duplicate rows, mark the rest for deletion.

    A dev database can hold both the old registry-id name and the new display
    name (an old seed plus a hand-made row). The row already carrying the
    display name wins, else the oldest wins, with ties broken by name so the
    choice stays deterministic.
    """
    candidates = list(rows)
    if not candidates:
        return None, []
    named = [row for row in candidates if row.name == display_name]
    pool = named or candidates
    keep = min(pool, key=lambda row: (row.created_at, row.name))
    return keep, [row for row in candidates if row is not keep]


TRANSCRIBE_MODELS = _pairs(
    "DEFAULT_TRANSCRIBE_MODEL",
    (
        ("greek-calamari-v1", "greek-v1"),
        ("armenian-calamari-v1", "armenian-v1"),
        ("coptic-calamari-v1", "coptic-v1"),
        ("syriac-ppocr-v1", "syriac-v2"),
    ),
)
BINDING_PROJECT_SLUG = os.environ.get(
    "BINDING_PROJECT_SLUG",
    os.environ.get("DEV_ANNOTATED_PROJECT_SLUG", "byzantine-greek-manuscripts"),
)

# ``provider`` is a catalog label, not a dispatch key: nothing reads it back
# except the ordering in ``model_repository.list_models``. It was hardcoded to
# "kraken" for every row, which was already only half true for BLLA (the
# topology is Kraken's, the runtime is our own ONNX graph) and plainly wrong for
# a Calamari model. Naming the architecture is the reading that survives someone
# scanning the table to work out what a row runs on.
#
# ``blla`` and ``blla-segment`` are both spellings the enum accepts. registry.yaml
# uses the first; both mean the same runtime, so both map to the same provider.
_PROVIDER_BY_ARCHITECTURE: dict[RegistryArchitecture, str] = {
    RegistryArchitecture.blla: "kraken",
    RegistryArchitecture.blla_segment: "kraken",
    RegistryArchitecture.calamari: "calamari",
    RegistryArchitecture.ppocr_rec: "ppocr",
    RegistryArchitecture.ppocr_det: "ppocr",
}


def _binding_scope(binding: ModelBinding) -> tuple:
    """Return the scope that makes two bindings for one task collide."""
    return (binding.task, binding.project_id, binding.document_id, binding.document_part_id)


async def repoint_model_references(
    session: AsyncSession, duplicate_id: uuid.UUID, keep_id: uuid.UUID
) -> None:
    """Move bindings and jobs from a duplicate model row to the kept row.

    Runs before the duplicate row is deleted so its bindings are not
    cascade-deleted and its jobs are not nulled. A duplicate binding whose
    scope the kept row already covers is deleted instead of repointed, since
    one scope holds one binding; only that conflicting row is removed.
    """
    keep_result = await session.execute(
        select(ModelBinding).where(ModelBinding.model_id == keep_id)
    )
    keep_scopes = {_binding_scope(binding) for binding in keep_result.scalars().all()}
    duplicate_result = await session.execute(
        select(ModelBinding).where(ModelBinding.model_id == duplicate_id)
    )
    for binding in duplicate_result.scalars().all():
        scope = _binding_scope(binding)
        if scope in keep_scopes:
            await session.execute(delete(ModelBinding).where(ModelBinding.id == binding.id))
        else:
            await session.execute(
                update(ModelBinding).where(ModelBinding.id == binding.id).values(model_id=keep_id)
            )
            keep_scopes.add(scope)
    await session.execute(update(Job).where(Job.model_id == duplicate_id).values(model_id=keep_id))


async def _upsert_model(
    *, registry_id: str, display_name: str, task: InferenceTask
) -> InferenceModel:
    """Write one catalog row for a registry model id, creating or rewriting it.

    The row is found by ``artifact_ref`` so re-seeding after a display-name
    change renames the existing row instead of duplicating it. An id the
    Registry does not know raises here rather than seeding a row that
    resolves to nothing at dispatch time.
    """
    entry = load_registry().models.get(registry_id)
    if entry is None:
        raise SystemExit(
            f"{registry_id!r} is not in registry.yaml; add the entry before seeding it"
        )
    if entry.task.value != task.value:
        raise SystemExit(
            f"{registry_id!r} is a {entry.task.value} model in registry.yaml, "
            f"seeded here as {task.value}"
        )
    provider = _PROVIDER_BY_ARCHITECTURE[entry.architecture]
    artifact_ref = artifact_ref_for(registry_id)
    default_params = {"device": entry.device.value}

    async with system_session() as session:
        # artifact_ref is not unique, so the old name and the new name can
        # both be present with the same ref. One query fetches every row
        # matching the ref or the candidate names, then all but one are
        # removed: the row already carrying the display name wins, else the
        # oldest row wins. This never calls scalar_one_or_none on the ref,
        # which would raise before the duplicates could be removed.
        result = await session.execute(
            select(InferenceModel).where(
                or_(
                    InferenceModel.artifact_ref == artifact_ref,
                    InferenceModel.name.in_(candidate_names_for(registry_id, display_name)),
                )
            )
        )
        rows = list(result.scalars().all())
        model, duplicates = split_duplicate_models(rows, display_name)
        for duplicate in duplicates:
            await repoint_model_references(session, duplicate.id, model.id)
            await session.delete(duplicate)
        if duplicates:
            await session.flush()
        if model is None:
            model = InferenceModel(
                name=display_name,
                provider=provider,
                task=task,
                artifact_ref=artifact_ref,
                default_params=default_params,
            )
            session.add(model)
        else:
            model.name = display_name
            model.provider = provider
            model.task = task
            model.artifact_ref = artifact_ref
            model.default_params = default_params
        await session.commit()
        await session.refresh(model)
        return model


async def _upsert_project_binding(
    *, project: Project, task: InferenceTask, model: InferenceModel
) -> ModelBinding:
    async with system_session() as session:
        result = await session.execute(
            select(ModelBinding).where(
                ModelBinding.project_id == project.id,
                ModelBinding.document_id.is_(None),
                ModelBinding.document_part_id.is_(None),
                ModelBinding.task == task,
            )
        )
        binding = result.scalar_one_or_none()
        if binding is None:
            binding = ModelBinding(
                project_id=project.id,
                task=task,
                model_id=model.id,
                overrides={},
            )
            session.add(binding)
        else:
            binding.model_id = model.id
            binding.overrides = {}
        await session.commit()
        await session.refresh(binding)
        return binding


async def main() -> None:
    segment_models = [
        await _upsert_model(
            registry_id=registry_id, display_name=display_name, task=InferenceTask.segment
        )
        for registry_id, display_name in SEGMENT_MODELS
    ]
    transcribe_models = [
        await _upsert_model(
            registry_id=registry_id, display_name=display_name, task=InferenceTask.transcribe
        )
        for registry_id, display_name in TRANSCRIBE_MODELS
    ]

    for model in segment_models:
        print(f"Seeded segment model: {model.name} ({model.provider}) -> {model.artifact_ref}")
    for model in transcribe_models:
        print(f"Seeded transcribe model: {model.name} ({model.provider}) -> {model.artifact_ref}")
    print(
        "Note: no project-level transcribe binding "
        "(attach script-specific models per project when available)."
    )

    async with system_session() as session:
        project = (
            await session.execute(select(Project).where(Project.slug == BINDING_PROJECT_SLUG))
        ).scalar_one_or_none()

    if project is None:
        print(
            f"No project slug={BINDING_PROJECT_SLUG!r} yet - "
            "run annotated seed first, then re-run this script for bindings."
        )
        return

    # The project-level segment binding still points at one model, the first
    # seeded. Transcribe stays unbound on purpose: four scripts are now in the
    # catalog and picking one of them as a workspace default here would put a
    # Greek model in front of a Syriac manuscript.
    segment_binding = await _upsert_project_binding(
        project=project,
        task=InferenceTask.segment,
        model=segment_models[0],
    )
    print(f"Segment binding on {project.slug} ({project.id}): {segment_binding.id}")


if __name__ == "__main__":
    asyncio.run(main())
