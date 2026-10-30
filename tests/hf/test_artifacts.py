"""Artifact integrity pinning: memoized, but never weakened."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from nomikos_inference.hub import artifacts
from nomikos_inference.hub.artifacts import (
    ArtifactIntegrityError,
    find_hub_artifact,
    verify_artifact_sha256,
)

CONTENT = b"checkpoint bytes" * 64
DIGEST = hashlib.sha256(CONTENT).hexdigest()


@pytest.fixture(autouse=True)
def clear_verification_cache():
    artifacts._VERIFIED_DIGESTS.clear()
    yield
    artifacts._VERIFIED_DIGESTS.clear()


@pytest.fixture
def hash_calls(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    calls: list[Path] = []
    real = artifacts.sha256_file

    def counting_sha256_file(path: Path) -> str:
        calls.append(path)
        return real(path)

    monkeypatch.setattr(artifacts, "sha256_file", counting_sha256_file)
    return calls


def _write(path: Path, data: bytes = CONTENT) -> Path:
    path.write_bytes(data)
    return path


def test_corrupted_artifact_raises(tmp_path: Path):
    artifact = _write(tmp_path / "model.pt", b"corrupted")

    with pytest.raises(ArtifactIntegrityError, match="artifact SHA-256 mismatch"):
        verify_artifact_sha256(artifact, DIGEST)


def test_corruption_after_a_successful_verification_still_raises(
    tmp_path: Path, hash_calls: list[Path]
):
    """The memoized pass must not vouch for a file that changed afterwards."""
    artifact = _write(tmp_path / "model.pt")
    verify_artifact_sha256(artifact, DIGEST)

    artifact.write_bytes(b"corrupted")
    with pytest.raises(ArtifactIntegrityError, match="artifact SHA-256 mismatch"):
        verify_artifact_sha256(artifact, DIGEST)
    assert len(hash_calls) == 2


def test_same_size_replacement_is_reverified(tmp_path: Path, hash_calls: list[Path]):
    """Size alone cannot be trusted: a swap of equal length must be re-hashed."""
    artifact = _write(tmp_path / "model.pt")
    verify_artifact_sha256(artifact, DIGEST)
    stat = artifact.stat()

    swapped = bytes(len(CONTENT))
    assert len(swapped) == stat.st_size
    artifact.write_bytes(swapped)
    # Guarantee a distinct mtime even on a filesystem with coarse timestamps.
    os.utime(artifact, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

    with pytest.raises(ArtifactIntegrityError, match="artifact SHA-256 mismatch"):
        verify_artifact_sha256(artifact, DIGEST)
    assert len(hash_calls) == 2


def test_distinct_artifacts_are_verified_separately(tmp_path: Path, hash_calls: list[Path]):
    first = _write(tmp_path / "a.pt")
    second = _write(tmp_path / "b.pt", b"other bytes")

    verify_artifact_sha256(first, DIGEST)
    with pytest.raises(ArtifactIntegrityError):
        verify_artifact_sha256(second, DIGEST)
    assert len(hash_calls) == 2


def test_new_expected_digest_is_not_served_from_the_memo(tmp_path: Path):
    artifact = _write(tmp_path / "model.pt")
    verify_artifact_sha256(artifact, DIGEST)

    with pytest.raises(ArtifactIntegrityError):
        verify_artifact_sha256(artifact, "0" * 64)


def test_missing_artifact_raises_file_not_found(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        verify_artifact_sha256(tmp_path / "absent.pt", DIGEST)


def test_memo_stays_bounded(tmp_path: Path):
    for index in range(artifacts._VERIFIED_DIGESTS_MAX + 5):
        verify_artifact_sha256(_write(tmp_path / f"model-{index}.pt"), DIGEST)

    assert len(artifacts._VERIFIED_DIGESTS) <= artifacts._VERIFIED_DIGESTS_MAX


def test_known_architecture_missing_its_named_file_raises(tmp_path: Path):
    """A wrong pick must read as a missing artifact, not an integrity failure."""
    _write(tmp_path / "other.onnx")
    with pytest.raises(FileNotFoundError, match="model.onnx"):
        find_hub_artifact(tmp_path, architecture="ppocr_rec")


def test_legacy_architecture_spellings_resolve(tmp_path: Path):
    _write(tmp_path / "blla.onnx")
    assert find_hub_artifact(tmp_path, architecture="blla_segment").name == "blla.onnx"
    assert find_hub_artifact(tmp_path, architecture="kraken_segment").name == "blla.onnx"
    _write(tmp_path / "ppocrv6-det.onnx")
    assert find_hub_artifact(tmp_path, architecture="ppocr_det").name == "ppocrv6-det.onnx"


def test_unknown_architecture_raises_instead_of_guessing(tmp_path: Path):
    _write(tmp_path / "model.onnx")
    with pytest.raises(FileNotFoundError, match="unknown architecture"):
        find_hub_artifact(tmp_path, architecture="trocr")


def test_no_architecture_still_picks_the_first_onnx(tmp_path: Path):
    _write(tmp_path / "b.onnx")
    first = _write(tmp_path / "a.onnx")
    assert find_hub_artifact(tmp_path, architecture=None) == first
