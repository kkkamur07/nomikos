"""Reading a model's identity off a job, and keeping it readable when things are missing.

This runs at the moment a merge writes output, so it is on the critical path of
every finished inference job. It must never raise: a provenance note that can fail
turns a successful page of OCR into a failed job.
"""

from __future__ import annotations

import uuid

import pytest

from backend.jobs.infrastructure.orm_models import Job, JobType
from backend.ml.application.model_identity import (
    EMPTY_MODEL_IDENTITY,
    identity_for_job,
    split_artifact_ref,
)
from backend.ml.infrastructure.orm_models import InferenceModel, InferenceTask


def _model(artifact_ref: str = "registry://blla-segment?tag=stable") -> InferenceModel:
    return InferenceModel(
        id=uuid.uuid4(),
        name="blla-segment",
        provider="kraken",
        task=InferenceTask.segment,
        artifact_ref=artifact_ref,
        default_params={"heatmap_threshold": 0.17},
    )


def _job(model: InferenceModel | None, payload: dict | None) -> Job:
    """A job as the callback's session hands it over, with ``model`` already loaded.

    ``model_id`` set while ``model`` is ``None`` is the row-deleted case: the FK is
    ``ON DELETE SET NULL``, so it cannot outlive its model in the database, but a job
    read before that commit can.
    """
    job = Job(
        id=uuid.uuid4(),
        type=JobType.segment,
        payload=payload,
        model_id=model.id if model is not None else None,
    )
    job.model = model
    return job


@pytest.mark.parametrize(
    ("artifact_ref", "expected"),
    [
        ("registry://blla-segment?tag=stable", ("blla-segment", "stable")),
        ("registry://greek-calamari-v1?tag=2026-09", ("greek-calamari-v1", "2026-09")),
        # No tag in the ref means the registry's own default, which is "stable".
        ("registry://blla-segment", ("blla-segment", "stable")),
        # Shapes this cannot read keep the whole ref in ``model_artifact_ref`` and
        # leave the split-out columns null rather than guessing.
        ("https://example.test/model.onnx", (None, None)),
        ("", (None, None)),
        (None, (None, None)),
    ],
)
def test_the_registry_ref_splits_into_a_model_and_a_tag(artifact_ref, expected):
    assert split_artifact_ref(artifact_ref) == expected


def test_a_job_with_a_model_carries_its_name_ref_and_effective_params():
    job = _job(_model(), {"ml_params": {"heatmap_threshold": 0.3}})

    identity = identity_for_job(job)

    assert identity.name == "blla-segment"
    assert identity.provider == "kraken"
    assert identity.artifact_ref == "registry://blla-segment?tag=stable"
    assert identity.registry_model_id == "blla-segment"
    assert identity.registry_tag == "stable"
    # The job's params, not the registry defaults: this is what actually ran, with
    # the model defaults, the binding overrides and the caller's value already merged.
    assert identity.params == {"heatmap_threshold": 0.3}


def test_a_job_with_no_model_still_records_the_params_it_ran_with():
    # No ``model_id`` means the dispatcher's fallback registry entry, which has no
    # ``inference_models`` row to name. The params are still true about the run.
    job = _job(None, {"ml_params": {"heatmap_threshold": 0.2}})

    identity = identity_for_job(job)

    assert identity.name is None
    assert identity.params == {"heatmap_threshold": 0.2}


def test_a_model_row_deleted_since_submission_yields_nulls_not_an_error():
    job = _job(None, {"ml_params": {}})
    job.model_id = uuid.uuid4()

    identity = identity_for_job(job)

    assert identity == EMPTY_MODEL_IDENTITY


def test_a_job_with_no_payload_at_all_yields_empty_params():
    job = _job(None, None)

    assert identity_for_job(job).params == {}
