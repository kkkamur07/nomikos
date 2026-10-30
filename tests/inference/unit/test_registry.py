"""registry.yaml validation and inference-local weight path helpers."""

import pytest

from nomikos_inference.contracts import ComputeDevice, InferenceTask, RegistryArchitecture
from nomikos_inference.registry import RegistryVersionEntry, get_model_entry, load_registry
from nomikos_inference.weights import resolve_weights_source

# --- registry.yaml entries ---
# Tests bundled model metadata loads correctly. Does not run inference.


# The four scripts are one model family: same graph, same device, same
# eligibility, one Hub repo and codec each. Parametrizing says that out loud, and
# means adding a fifth script is a one-line change here rather than a fifth
# copy of the same six assertions.
TRANSCRIBE_MODEL_IDS = (
    "greek-calamari-v1",
    "armenian-calamari-v1",
    "syriac-calamari-v2",
    "coptic-calamari-v1",
)


@pytest.mark.parametrize("registry_model_id", TRANSCRIBE_MODEL_IDS)
def test_registry_yaml_validates_calamari_transcribe_entries(registry_model_id: str):
    """The schema each entry must satisfy, not the values one publish happened to have.

    The literal ``hub_revision`` and ``artifact_sha256`` this used to inline made a
    legitimate model republish a test edit. What the digests are actually *worth* is
    checked against real bytes by ``test_onnx_runtime.py::
    test_the_registry_pins_the_digest_of_the_artifact_the_loader_opens``; here only
    their presence and shape matter.
    """
    entry = get_model_entry(load_registry(), registry_model_id, "stable")

    assert entry.task == InferenceTask.transcribe
    assert entry.architecture == RegistryArchitecture.calamari
    assert entry.device == ComputeDevice.cpu
    assert entry.host_eligibility.value == "local"

    version = entry.versions["stable"]
    assert version.weights_source.startswith("hf://")
    assert len(version.hub_revision) == 40
    assert len(version.artifact_sha256) == 64


def test_every_transcribe_entry_names_its_own_hub_repo():
    """Three entries copied from one another is exactly how two of them end up
    resolving the same weights. A shared ``weights_source`` would still validate,
    load, and transcribe: it would just silently read Greek pages with the Syriac
    codec, so the distinctness is worth an assertion of its own."""
    registry = load_registry()
    sources = {
        registry.models[registry_model_id].versions["stable"].weights_source
        for registry_model_id in TRANSCRIBE_MODEL_IDS
    }
    assert len(sources) == len(TRANSCRIBE_MODEL_IDS)


def test_registry_yaml_validates_the_blla_segment_entry():
    registry = load_registry()

    blla = get_model_entry(registry, "blla-segment", "stable")
    assert blla.task == InferenceTask.segment
    assert blla.architecture == RegistryArchitecture.blla
    assert blla.device == ComputeDevice.cpu
    assert blla.versions["stable"].weights_source.startswith("hf://")
    assert len(blla.versions["stable"].hub_revision) == 40
    assert len(blla.versions["stable"].artifact_sha256) == 64


# --- Weight path resolution ---
# Hub weights are resolved in tests/hf; this verifies the native artifact shape.


def test_registry_rejects_partial_hf_provenance():
    with pytest.raises(ValueError, match="both hub_revision and artifact_sha256"):
        RegistryVersionEntry(
            weights_source="hf://example/demo@stable",
            hub_revision="a" * 40,
        )


def test_registry_rejects_hf_entry_with_neither_pin():
    """An hf:// entry with neither pin must fail validation, not resolve time."""
    with pytest.raises(ValueError, match="both hub_revision and artifact_sha256"):
        RegistryVersionEntry(weights_source="hf://example/demo@stable")


def _transcribe_entry(**overrides: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "task": "transcribe",
        "architecture": "calamari",
        "device": "cpu",
        "line_crop": "polygon-white",
        "line_crop_padding": 12,
        "versions": {
            "stable": {
                "weights_source": "hf://example/demo@stable",
                "hub_revision": "a" * 40,
                "artifact_sha256": "b" * 64,
            }
        },
    }
    entry.update(overrides)
    return entry


def _transcribe_doc(**overrides: object) -> dict[str, object]:
    return {"models": {"demo": _transcribe_entry(**overrides)}}


def test_registry_rejects_transcribe_entry_without_crop() -> None:
    from nomikos_inference.registry import RegistryDocument

    entry = _transcribe_entry()
    entry.pop("line_crop")
    with pytest.raises(ValueError, match="must set line_crop"):
        RegistryDocument.model_validate({"models": {"demo": entry}})


def test_registry_rejects_transcribe_entry_without_padding() -> None:
    from nomikos_inference.registry import RegistryDocument

    entry = _transcribe_entry()
    entry.pop("line_crop_padding")
    with pytest.raises(ValueError, match="must set line_crop_padding"):
        RegistryDocument.model_validate({"models": {"demo": entry}})


def test_registry_rejects_negative_padding() -> None:
    from nomikos_inference.registry import RegistryDocument

    with pytest.raises(ValueError, match="negative line_crop_padding"):
        RegistryDocument.model_validate(_transcribe_doc(line_crop_padding=-1))


def test_registry_rejects_crop_on_segment_entry() -> None:
    from nomikos_inference.registry import RegistryDocument

    with pytest.raises(ValueError, match="must not set line_crop"):
        RegistryDocument.model_validate(
            _transcribe_doc(task="segment", architecture="blla", device="cpu")
        )


# `test_blla_rejects_digest_mismatch_before_runtime_load` stood here. The same ordering
# claim -- digest before open, integrity failure rather than a parse error -- is
# `test_architecture_contract.py::test_corrupt_artifact_is_a_service_error_not_a_client_error`,
# which runs it over both architectures instead of BLLA alone.


# --- Path safety ---
# Tests weights cannot escape INFERENCE_ROOT. Does not test Hub download caching.


def test_weights_source_rejects_paths_outside_ml_root():
    with pytest.raises(ValueError, match="relative to INFERENCE_ROOT"):
        resolve_weights_source("file:///etc/passwd")

    with pytest.raises(ValueError, match="within INFERENCE_ROOT"):
        resolve_weights_source("file://../pyproject.toml")


# `test_interim_weights_layout` stood here and asserted `nomikos_inference/weights/kraken/` is a
# directory. pyproject.toml calls that tree an empty placeholder from the pre-Hub weights
# layout and excludes its `.gitkeep` from the wheel; `DEFAULT_WEIGHTS_ROOT` has no
# production caller.
