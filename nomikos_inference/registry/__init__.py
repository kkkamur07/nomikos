"""Registry model catalog loaded from registry.yaml."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from nomikos_inference import INFERENCE_ROOT
from nomikos_inference.contracts.common import (
    ComputeDevice,
    HostEligibility,
    InferenceTask,
    LineCrop,
    RegistryArchitecture,
)
from nomikos_inference.hub.pins import COMMIT_SHA_PATTERN, SHA256_PATTERN

DEFAULT_REGISTRY_PATH = INFERENCE_ROOT / "registry.yaml"


class RegistryVersionEntry(BaseModel):
    weights_source: str = Field(min_length=1)
    hub_revision: str | None = None
    artifact_sha256: str | None = None

    @field_validator("hub_revision")
    @classmethod
    def validate_hub_revision(cls, value: str | None) -> str | None:
        if value is not None and not COMMIT_SHA_PATTERN.fullmatch(value):
            raise ValueError("hub_revision must be a 40-character lowercase commit SHA")
        return value

    @field_validator("artifact_sha256")
    @classmethod
    def validate_artifact_sha256(cls, value: str | None) -> str | None:
        if value is not None and not SHA256_PATTERN.fullmatch(value):
            raise ValueError("artifact_sha256 must be a 64-character lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def validate_hf_provenance_pair(self) -> RegistryVersionEntry:
        if self.weights_source.startswith("hf://") and not (
            self.hub_revision and self.artifact_sha256
        ):
            raise ValueError("hf weights_source must provide both hub_revision and artifact_sha256")
        if not self.weights_source.startswith("hf://") and self.hub_revision is not None:
            raise ValueError("hub_revision is only valid for hf weights_source")
        return self


class RegistryModelEntry(BaseModel):
    task: InferenceTask
    architecture: RegistryArchitecture
    device: ComputeDevice
    host_eligibility: HostEligibility = HostEligibility.local
    #: How this model's training line crops were cut, and by how much the
    #: polygon's box was widened before masking. Both are required for transcribe
    #: entries and rejected for the others; see ``RegistryDocument`` for why the
    #: rule lives one level up. Optional here only so the error can name the
    #: model id rather than a field path.
    line_crop: LineCrop | None = None
    line_crop_padding: int | None = None
    versions: dict[str, RegistryVersionEntry] = Field(min_length=1)


def _require_transcribe_crop(model_id: str, entry: RegistryModelEntry) -> None:
    """A transcribe entry must state the crop recipe its training used."""
    if entry.line_crop is None:
        raise ValueError(
            f"registry model {model_id!r} has task 'transcribe' and must set "
            "line_crop to one of: " + ", ".join(repr(option.value) for option in LineCrop)
        )
    if entry.line_crop_padding is None:
        raise ValueError(
            f"registry model {model_id!r} has task 'transcribe' and must set "
            "line_crop_padding, the pixel padding its training crops were "
            "exported with"
        )
    if entry.line_crop_padding < 0:
        raise ValueError(
            f"registry model {model_id!r} has a negative line_crop_padding "
            f"({entry.line_crop_padding}); padding widens the crop and cannot "
            "shrink it"
        )


def _reject_non_transcribe_crop(model_id: str, entry: RegistryModelEntry) -> None:
    """Crop fields only apply to transcribe models."""
    for field in ("line_crop", "line_crop_padding"):
        if getattr(entry, field) is not None:
            raise ValueError(
                f"registry model {model_id!r} has task {entry.task.value!r} and "
                f"must not set {field}, which only applies to transcribe models"
            )


class RegistryDocument(BaseModel):
    models: dict[str, RegistryModelEntry] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_line_crop_per_task(self) -> RegistryDocument:
        """Every transcribe model has to state how its training crops were cut.

        There is no safe default for the padding. Serving Greek at Armenian's 12
        px costs it 125 exact lines out of 204 down to 0, CER 0.050 to 0.304, and
        it fails silently: plausible-looking text, quietly wrong. So a transcribe
        entry without both fields is a registry error rather than an assumption,
        and the message names the model id because that is what the person
        editing this YAML is looking at.
        """
        for model_id, entry in self.models.items():
            if entry.task == InferenceTask.transcribe:
                _require_transcribe_crop(model_id, entry)
            else:
                _reject_non_transcribe_crop(model_id, entry)
        return self


@lru_cache(maxsize=8)
def _load_registry_cached(resolved_path: str, mtime_ns: int) -> RegistryDocument:
    """Parse + validate one registry file. Keyed on path and mtime so an edited
    ``registry.yaml`` is re-read, but the common hot path (unchanged file read
    once per page) skips the YAML parse and full Pydantic validation entirely.
    """
    raw = yaml.safe_load(Path(resolved_path).read_text(encoding="utf-8"))
    return RegistryDocument.model_validate(raw)


def load_registry(path: Path | None = None) -> RegistryDocument:
    registry_path = (path or DEFAULT_REGISTRY_PATH).resolve()
    return _load_registry_cached(str(registry_path), registry_path.stat().st_mtime_ns)


def get_model_entry(
    registry: RegistryDocument,
    registry_model_id: str,
    registry_tag: str = "stable",
) -> RegistryModelEntry:
    try:
        model = registry.models[registry_model_id]
    except KeyError as exc:
        raise KeyError(f"unknown registry model id: {registry_model_id}") from exc

    try:
        model.versions[registry_tag]
    except KeyError as exc:
        raise KeyError(
            f"unknown registry tag {registry_tag!r} for model {registry_model_id!r}"
        ) from exc
    return model
