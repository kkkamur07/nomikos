"""The ONNX Runtime contract, exercised on real weights (ADR 0006).

Everything here runs a real graph. The rules under test are the ones ADR 0006
makes load-bearing and that nothing else pins:

* both architectures execute through ONNX Runtime on CPU,
* the artifact carries its own codec, so nothing but the ``.onnx`` is needed,
* the **artifact SHA-256** is verified *before* the artifact is opened,
* a retired native checkpoint sitting in the same directory is refused rather
  than loaded,
* the outputs are deterministic run to run, so a local result and a cloud
  worker result are comparable.

No stubbed sessions: a test that cannot run a real graph does not belong in
this file. Nothing here imports Torch - that is the point of the split, and
``test_architecture_contract`` asserts it of the package rather than of this
module.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from PIL import Image

from nomikos_inference.architectures.blla.blla import run_blla_segment
from nomikos_inference.architectures.calamari.adapter import (
    CalamariUnavailableError,
    _load_session,
    run_calamari_transcribe,
)
from nomikos_inference.hub.artifacts import ArtifactIntegrityError, find_hub_artifact
from tests.fixtures.paths import REPO_ROOT, SEGMENT_PAGE, TRANSCRIBE_LINE

# The published artifacts, as fetched from the Hub revisions the registry pins.
# They are gitignored (``/nomikos_inference/publish/artifacts/cache/``), so a
# checkout that has not fetched
# them exports an equivalent graph from the tracked native checkpoints instead -
# see ``_exported``. Either way these tests run a real graph; what varies is
# whether it is the published one or one built from the same weights.
PUBLISHED_CALAMARI = (
    REPO_ROOT / "nomikos_inference/publish/artifacts/cache/syriac-calamari-v2/stable/best.onnx"
)
PUBLISHED_BLLA = (
    REPO_ROOT / "nomikos_inference/publish/artifacts/cache/blla-segment/stable/blla.onnx"
)
CALAMARI_CHECKPOINT = (
    REPO_ROOT / "nomikos_inference/publish/artifacts/local/syriac/calamari/v1/stable/best.pt"
)
BLLA_CHECKPOINT = (
    REPO_ROOT
    / "nomikos_inference/publish/artifacts/staging"
    / "models/segmentation/blla/v1/stable/blla.safetensors"
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="session")
def calamari_artifact(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if PUBLISHED_CALAMARI.is_file():
        return PUBLISHED_CALAMARI
    return _exported("calamari", tmp_path_factory)


@pytest.fixture(scope="session")
def blla_artifact(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if PUBLISHED_BLLA.is_file():
        return PUBLISHED_BLLA
    return _exported("blla", tmp_path_factory)


def _exported(architecture: str, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the artifact from the tracked checkpoint, or skip.

    The import is inside the fixture on purpose: ``export`` pulls in Torch, and
    a module-level import would put it in this file's import graph even on the
    machines that never need it.
    """
    pytest.importorskip("torch", reason="no published .onnx cached and Torch is unavailable")
    destination = tmp_path_factory.mktemp("onnx") / f"{architecture}.onnx"
    if architecture == "calamari":
        from nomikos_inference.export.calamari import export_calamari_onnx

        export_calamari_onnx(CALAMARI_CHECKPOINT, destination)
    else:
        from nomikos_inference.export.blla import export_blla_onnx

        export_blla_onnx(BLLA_CHECKPOINT, destination, example_width=64)
    return destination


# --- Both architectures run, on ONNX Runtime, on CPU --------------------------


def test_transcribe_runs_the_graph_on_real_weights(calamari_artifact: Path) -> None:
    response = run_calamari_transcribe(
        TRANSCRIBE_LINE.read_bytes(),
        checkpoint_path=calamari_artifact,
        artifact_sha256=_digest(calamari_artifact),
    )

    assert response.text
    assert 0.0 <= response.confidence <= 1.0
    assert len(response.character_confidences) == len(response.text)


def test_segment_runs_the_graph_on_real_weights(blla_artifact: Path) -> None:
    response = run_blla_segment(
        SEGMENT_PAGE.read_bytes(),
        model_path=blla_artifact,
        artifact_sha256=_digest(blla_artifact),
    )

    assert len(response.blocks) == 1
    assert len(response.lines) > 10
    assert all(len(line.points) >= 4 for line in response.lines)


def test_the_runtime_never_selects_an_accelerator(calamari_artifact: Path) -> None:
    """CPU only: a CoreML or CUDA provider would make a laptop result differ."""
    _load_session.cache_clear()
    session, _, _ = _load_session(str(calamari_artifact))

    assert session.get_providers() == ["CPUExecutionProvider"]


def test_both_runtimes_are_deterministic_across_runs(
    calamari_artifact: Path,
    blla_artifact: Path,
) -> None:
    """A researcher's result must be reproducible against the cloud worker.

    Under ADR 0004 this caught a model left in training mode. There is no
    dropout to leave on in a traced graph, which is one of the things the
    conversion buys - but the property a researcher depends on is the same, so
    it stays asserted rather than assumed.
    """
    first = run_calamari_transcribe(TRANSCRIBE_LINE.read_bytes(), checkpoint_path=calamari_artifact)
    second = run_calamari_transcribe(
        TRANSCRIBE_LINE.read_bytes(), checkpoint_path=calamari_artifact
    )
    assert first.text == second.text
    assert first.confidence == second.confidence

    page = SEGMENT_PAGE.read_bytes()
    first_segment = run_blla_segment(page, model_path=blla_artifact)
    second_segment = run_blla_segment(page, model_path=blla_artifact)
    assert [line.points for line in first_segment.lines] == [
        line.points for line in second_segment.lines
    ]


# --- The artifact is self-contained -------------------------------------------


def test_the_graph_carries_the_codec_the_decoder_needs(calamari_artifact: Path) -> None:
    """No sidecar, no ``.pt``: the metadata is why one file is enough.

    This is what replaces ADR 0004's ``strict=True`` state-dict load. There the
    graph and the checkpoint had to agree at load time; here the graph *is* the
    checkpoint, and what has to be checked is that the exporter stamped
    everything the decoder reads.
    """
    _load_session.cache_clear()
    _, charset, line_height = _load_session(str(calamari_artifact))

    # Deliberately not `line_height == 48` and `len(charset) == 47`: those are
    # properties of one published checkpoint, and pinning them turns a legitimate
    # republish red. What the decoder needs is that the stamps are *there* and
    # sane, plus the one value that is a real decode invariant rather than a
    # property of these weights.
    assert isinstance(line_height, int) and line_height > 0
    assert charset
    assert charset[0] == ""  # the CTC blank, at index 0


def test_a_graph_without_its_metadata_is_refused(tmp_path: Path) -> None:
    """A stripped or foreign graph must fail at load, not decode into nonsense."""
    onnx = pytest.importorskip("onnx")
    if not PUBLISHED_CALAMARI.is_file():
        pytest.skip("published Calamari artifact is not cached locally")

    model = onnx.load(str(PUBLISHED_CALAMARI))
    del model.metadata_props[:]
    stripped = tmp_path / "stripped.onnx"
    onnx.save(model, str(stripped))

    with pytest.raises(CalamariUnavailableError, match="unsupported Calamari ONNX artifact"):
        _load_session(str(stripped))


# --- The retired native formats are refused, not loaded -----------------------
# The per-architecture "a native checkpoint is refused" pair stood here. Both are
# the suffix branch, and `test_architecture_contract.py::
# test_foreign_artifact_is_a_service_error_on_every_path` takes it over both
# architectures *and* asserts the failure family, which neither of these did.


def test_the_cache_directory_resolves_to_the_graph_not_the_checkpoint(tmp_path: Path) -> None:
    """``snapshot_download`` fetches both formats; only one may be picked up."""
    (tmp_path / "best.pt").write_bytes(b"native checkpoint")
    (tmp_path / "best.onnx").write_bytes(b"graph")

    assert find_hub_artifact(tmp_path, architecture="calamari").name == "best.onnx"

    blla_dir = tmp_path / "blla"
    blla_dir.mkdir()
    (blla_dir / "blla.safetensors").write_bytes(b"native checkpoint")
    (blla_dir / "blla.onnx").write_bytes(b"graph")

    assert find_hub_artifact(blla_dir, architecture="blla").name == "blla.onnx"

    ppocr_dir = tmp_path / "ppocr"
    ppocr_dir.mkdir()
    (ppocr_dir / "a.onnx").write_bytes(b"alphabetically first graph")
    (ppocr_dir / "ppocrv6-det.onnx").write_bytes(b"graph")

    assert find_hub_artifact(ppocr_dir, architecture="ppocr-det").name == "ppocrv6-det.onnx"


# --- Integrity is checked before the artifact is opened -----------------------


def test_a_digest_mismatch_stops_the_load_before_the_runtime_sees_the_file(
    calamari_artifact: Path,
    tmp_path: Path,
) -> None:
    """Verification is ordered ahead of the load, not merely present."""
    tampered = tmp_path / "best.onnx"
    tampered.write_bytes(calamari_artifact.read_bytes() + b"\x00")

    with pytest.raises(ArtifactIntegrityError):
        run_calamari_transcribe(
            TRANSCRIBE_LINE.read_bytes(),
            checkpoint_path=tampered,
            artifact_sha256=_digest(calamari_artifact),
        )


def test_the_registry_pins_the_digest_of_the_artifact_the_loader_opens() -> None:
    """The pinned digest must be the file the runtime will actually run."""
    import yaml

    if not (PUBLISHED_CALAMARI.is_file() and PUBLISHED_BLLA.is_file()):
        pytest.skip("published artifacts are not cached locally")

    registry = yaml.safe_load(
        (REPO_ROOT / "nomikos_inference/registry.yaml").read_text(encoding="utf-8")
    )

    calamari = registry["models"]["syriac-calamari-v2"]["versions"]["stable"]
    assert calamari["artifact_sha256"] == _digest(PUBLISHED_CALAMARI)

    blla = registry["models"]["blla-segment"]["versions"]["stable"]
    assert blla["artifact_sha256"] == _digest(PUBLISHED_BLLA)


# --- The two architectures compose --------------------------------------------
# `test_a_page_that_segments_can_be_transcribed_line_by_line` stood here. Its only
# claim about the composed result was `hasattr(result, "text")` on a pydantic model,
# which holds whatever the crops contained. The composition is run for real by
# `test_published_package.py`'s `real_page_run` and end to end by `test_cli_run.py`;
# cutting it removes a third full model execution from this file.


def test_segment_geometry_stays_inside_the_page(blla_artifact: Path) -> None:
    with Image.open(SEGMENT_PAGE) as image:
        width, height = image.size

    response = run_blla_segment(SEGMENT_PAGE.read_bytes(), model_path=blla_artifact)

    assert response.lines
    for line in response.lines:
        metadata = line.source_metadata
        assert isinstance(metadata["simplification_status"], str)
        assert isinstance(metadata["raw_point_count"], int)
        assert isinstance(metadata["simplified_point_count"], int)
        assert isinstance(metadata["min_vertex_spacing"], float)
        assert isinstance(metadata["epsilon"], float)
        for x, y in line.points:
            assert -1 <= x <= width + 1
            assert -1 <= y <= height + 1
