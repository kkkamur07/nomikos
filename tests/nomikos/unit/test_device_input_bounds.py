"""Bounds on attacker-controlled JSON: pairing capabilities and model overrides."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.ml.api.device_schemas import PairingStartRequest
from backend.ml.api.schemas import ModelBindingCreateRequest, ModelBindingUpdateRequest
from backend.ml.infrastructure.orm_models import InferenceTask

MODEL_ID = "12345678-1234-5678-1234-567812345678"


def _pairing(**kwargs) -> PairingStartRequest:
    return PairingStartRequest(
        device_name="laptop", platform="linux", helper_version="1.0", **kwargs
    )


def test_capabilities_accept_string_tags() -> None:
    assert _pairing(capabilities={"runtime": "torch"}).capabilities == {"runtime": "torch"}
    assert _pairing().capabilities == {}


def test_capabilities_reject_non_strings_oversize_and_overflow() -> None:
    with pytest.raises(ValidationError):
        _pairing(capabilities={"runtime": ["torch"]})
    with pytest.raises(ValidationError):
        _pairing(capabilities={"k": "v" * 513})
    with pytest.raises(ValidationError):
        _pairing(capabilities={"k" * 65: "v"})
    with pytest.raises(ValidationError):
        _pairing(capabilities={f"k{i}": "v" for i in range(33)})


def test_overrides_accept_small_maps() -> None:
    body = ModelBindingCreateRequest(
        task=InferenceTask.transcribe, model_id=MODEL_ID, overrides={"level": "project"}
    )
    assert body.overrides == {"level": "project"}


def test_overrides_reject_overflow() -> None:
    with pytest.raises(ValidationError):
        ModelBindingCreateRequest(
            task=InferenceTask.transcribe,
            model_id=MODEL_ID,
            overrides={f"k{i}": "v" for i in range(65)},
        )
    with pytest.raises(ValidationError):
        ModelBindingCreateRequest(
            task=InferenceTask.transcribe, model_id=MODEL_ID, overrides={"k" * 129: "v"}
        )
    with pytest.raises(ValidationError):
        ModelBindingCreateRequest(
            task=InferenceTask.transcribe, model_id=MODEL_ID, overrides={"k": "v" * 20000}
        )
    with pytest.raises(ValidationError):
        ModelBindingUpdateRequest(overrides={"k": "v" * 20000})
    assert ModelBindingUpdateRequest(overrides=None).overrides is None
