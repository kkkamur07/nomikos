"""Essential hf:// resolution and cache coverage."""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from nomikos_inference.hub import resolve_hf_weights_source, set_default_hub_client
from nomikos_inference.hub.manifest import load_manifest
from nomikos_inference.hub.uri import parse_hf_weights_uri

REPO_ROOT = Path(__file__).resolve().parents[2]
MOCK_CHECKPOINT = (
    REPO_ROOT / "nomikos_inference/publish/artifacts/local/syriac/calamari/v1/stable/best.pt"
)
# A stand-in for the graph, not a real one: nothing on this path opens a
# session, it hashes bytes and moves files. What matters is that the mock
# snapshot puts *both* formats in the cache directory, the way a real
# `snapshot_download` of the whole revision does - so these tests exercise the
# rule that resolution picks the `.onnx` and never the checkpoint beside it.
MOCK_GRAPH = b"onnx graph bytes, published beside the checkpoint"
HUB_REVISION = "a" * 40
ARTIFACT_SHA256 = hashlib.sha256(MOCK_GRAPH).hexdigest()


@dataclass
class MockHubClient:
    downloads: list[tuple[str, str, Path]] = field(default_factory=list)

    def snapshot_download(self, repo_id: str, revision: str, local_dir: Path) -> None:
        self.downloads.append((repo_id, revision, local_dir))
        local_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(MOCK_CHECKPOINT, local_dir / "best.pt")
        (local_dir / "best.onnx").write_bytes(MOCK_GRAPH)


@pytest.fixture(autouse=True)
def reset_hub_client():
    set_default_hub_client(None)
    yield
    set_default_hub_client(None)


def test_parse_hf_weights_uri():
    parsed = parse_hf_weights_uri("hf://nomikos/greek-htr-calamari@stable")
    assert parsed.repo_id == "nomikos/greek-htr-calamari"
    assert parsed.registry_tag == "stable"


def test_resolve_uses_pinned_revision_not_moved_registry_tag(tmp_path: Path):
    client = MockHubClient()
    uri = "hf://nomikos/greek-htr-calamari@stable"
    kwargs = dict(
        uri=uri,
        registry_model_id="greek-calamari-v1",
        registry_tag="stable",
        hub_revision=HUB_REVISION,
        artifact_sha256=ARTIFACT_SHA256,
        architecture="calamari",
        hub_client=client,
        cache_root=tmp_path,
    )

    first = resolve_hf_weights_source(**kwargs)
    second = resolve_hf_weights_source(**kwargs)

    assert first == second
    assert len(client.downloads) == 1
    assert client.downloads[0][1] == HUB_REVISION
    manifest = load_manifest(tmp_path / "greek-calamari-v1" / "stable")
    assert manifest is not None
    assert manifest.hub_revision == HUB_REVISION
    assert manifest.artifact_sha256 == ARTIFACT_SHA256


def test_resolve_replaces_tampered_cache_before_reuse(tmp_path: Path):
    client = MockHubClient()
    kwargs = dict(
        uri="hf://nomikos/greek-htr-calamari@stable",
        registry_model_id="greek-calamari-v1",
        registry_tag="stable",
        hub_revision=HUB_REVISION,
        artifact_sha256=ARTIFACT_SHA256,
        architecture="calamari",
        hub_client=client,
        cache_root=tmp_path,
    )

    artifact = resolve_hf_weights_source(**kwargs)
    artifact.write_bytes(b"tampered")
    repaired = resolve_hf_weights_source(**kwargs)

    assert repaired == artifact
    assert repaired.name == "best.onnx"  # never the .pt sitting beside it
    assert repaired.read_bytes() == MOCK_GRAPH
    assert len(client.downloads) == 2


def test_resolve_rejects_downloaded_digest_mismatch(tmp_path: Path):
    with pytest.raises(ValueError, match="artifact SHA-256 mismatch"):
        resolve_hf_weights_source(
            "hf://nomikos/greek-htr-calamari@stable",
            registry_model_id="greek-calamari-v1",
            registry_tag="stable",
            hub_revision=HUB_REVISION,
            artifact_sha256="0" * 64,
            architecture="calamari",
            hub_client=MockHubClient(),
            cache_root=tmp_path,
        )

    assert not (tmp_path / "greek-calamari-v1" / "stable").exists()


def test_resolve_rejects_mutable_only_hf_reference(tmp_path: Path):
    with pytest.raises(ValueError, match="immutable.*hub_revision"):
        resolve_hf_weights_source(
            "hf://nomikos/greek-htr-calamari@stable",
            registry_model_id="greek-calamari-v1",
            registry_tag="stable",
            hub_revision=None,
            artifact_sha256=ARTIFACT_SHA256,
            architecture="calamari",
            hub_client=MockHubClient(),
            cache_root=tmp_path,
        )


def test_tampered_cache_logs_a_warning_before_redownload(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """On-disk tampering of a pinned artifact must leave a trace."""
    import logging

    kwargs = dict(
        uri="hf://nomikos/greek-htr-calamari@stable",
        registry_model_id="greek-calamari-v1",
        registry_tag="stable",
        hub_revision=HUB_REVISION,
        artifact_sha256=ARTIFACT_SHA256,
        architecture="calamari",
        hub_client=MockHubClient(),
        cache_root=tmp_path,
    )
    artifact = resolve_hf_weights_source(**kwargs)
    artifact.write_bytes(b"tampered")
    with caplog.at_level(logging.WARNING, logger="nomikos_inference.hub.cache"):
        repaired = resolve_hf_weights_source(**kwargs)
    assert repaired.read_bytes() == MOCK_GRAPH
    assert "re-downloading" in caplog.text


def test_raw_hub_error_is_wrapped_once(tmp_path: Path):
    """The client raises raw; resolve maps it to one ValueError."""

    class HfHubHTTPError(Exception):
        pass

    @dataclass
    class FailingHubClient:
        def snapshot_download(self, repo_id: str, revision: str, local_dir: Path) -> None:
            raise HfHubHTTPError("401 Unauthorized for a gated repo")

    with pytest.raises(ValueError, match="access denied"):
        resolve_hf_weights_source(
            "hf://nomikos/greek-htr-calamari@stable",
            registry_model_id="greek-calamari-v1",
            registry_tag="stable",
            hub_revision=HUB_REVISION,
            artifact_sha256=ARTIFACT_SHA256,
            architecture="calamari",
            hub_client=FailingHubClient(),
            cache_root=tmp_path,
        )
