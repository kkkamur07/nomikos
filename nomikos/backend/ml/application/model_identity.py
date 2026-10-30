"""Who produced this output, in a form that survives the registry and the job table.

The platform already records *which model was selected* - ``jobs.model_id`` points
at an ``inference_models`` row, and the effective parameters land in
``jobs.payload["ml_params"]``. Neither survives what users actually do. Finished
jobs are deletable (``clear_project_job_history``) and ``transcriptions.created_by_job_id``
is ``ON DELETE SET NULL``, so one click leaves a model layer with no model; the
registry row itself can be edited or repointed at a new artifact, after which the
join would answer with today's model about last month's output.

So identity is read once, at the moment the output is written, and copied onto the
row that holds the output. This module is that read. It is deliberately total: a
missing or malformed field yields nulls rather than an exception, because a merge
that has already run inference must not fail over a provenance note.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import parse_qs, urlparse

from backend.jobs.infrastructure.orm_models import Job
from backend.ml.infrastructure.orm_models import InferenceModel


@dataclass(frozen=True)
class ModelIdentity:
    """A denormalised snapshot of the model behind one run."""

    name: str | None = None
    provider: str | None = None
    artifact_ref: str | None = None
    registry_model_id: str | None = None
    registry_tag: str | None = None
    params: dict = field(default_factory=dict)


EMPTY_MODEL_IDENTITY = ModelIdentity()


def split_artifact_ref(artifact_ref: str | None) -> tuple[str | None, str | None]:
    """``registry://blla-segment?tag=stable`` -> ``("blla-segment", "stable")``.

    The tolerant twin of ``inference_dispatcher._registry_selection_from_artifact_ref``,
    which refuses anything that is not a registry URL because it is about to run it.
    Here the ref has already been run; a shape this cannot parse is recorded verbatim
    in ``artifact_ref`` and the two split-out columns stay null.
    """
    if not artifact_ref:
        return None, None
    parsed = urlparse(artifact_ref)
    if parsed.scheme != "registry" or not parsed.netloc:
        return None, None
    tag = parse_qs(parsed.query).get("tag", ["stable"])[0] or "stable"
    return parsed.netloc, tag


def identity_for_job(job: Job) -> ModelIdentity:
    """Resolve the identity of the model a job ran.

    ``params`` comes from the job rather than the registry row: it is the *effective*
    set, with the model's defaults, the binding's overrides and the caller's own
    ``ml_params`` already merged by ``document_job_enqueue``. The registry defaults
    alone would describe a run that did not happen.

    The model is read through ``job.model``, which the callback's sync session loads
    on demand. A job that is not attached to a session, or that was submitted with no
    explicit model, answers ``None`` and leaves the name columns null.
    """
    params = dict((job.payload or {}).get("ml_params") or {})
    # A job submitted with no explicit model ran the dispatcher's fallback registry
    # entry, which has no ``inference_models`` row. The params are still the truth
    # about the run, so they are kept and the names left null.
    model: InferenceModel | None = job.model if job.model_id is not None else None
    if model is None:
        return ModelIdentity(params=params)
    registry_model_id, registry_tag = split_artifact_ref(model.artifact_ref)
    return ModelIdentity(
        name=model.name,
        provider=model.provider,
        artifact_ref=model.artifact_ref,
        registry_model_id=registry_model_id,
        registry_tag=registry_tag,
        params=params,
    )


__all__ = [
    "EMPTY_MODEL_IDENTITY",
    "ModelIdentity",
    "identity_for_job",
    "split_artifact_ref",
]
