"""Locate architecture-native Hub artifacts inside a cache directory."""

from __future__ import annotations

import hashlib
from pathlib import Path

from nomikos_inference.contracts.common import RegistryArchitecture


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ArtifactIntegrityError(ValueError):
    """A weights artifact does not match its pinned SHA-256 digest.

    Subclasses ``ValueError`` because the adapters raise and sort on
    ``ValueError``, but callers must never treat it as a bad request: nothing
    about the submitted job is wrong, the weights on this machine are. The agent
    reports it as a failed page with its own reason, and any surface mapping
    exceptions to statuses owes it a service error, not a client one.
    """


# Digest of artifacts already hashed in this process, keyed by identity.
# Bounded so a long-lived process cannot grow it without limit.
_VERIFIED_DIGESTS: dict[tuple[str, int, int], str] = {}
_VERIFIED_DIGESTS_MAX = 64


def _artifact_identity(path: Path) -> tuple[str, int, int]:
    """Identify file *content* well enough to reuse a digest.

    Any write to the artifact changes its size or its mtime, so a rewritten,
    truncated, or swapped file misses the cache and gets hashed again.
    """
    stat = path.stat()
    return (str(path), stat.st_size, stat.st_mtime_ns)


def verify_artifact_sha256(path: Path, expected_sha256: str) -> None:
    """Raise ``ArtifactIntegrityError`` unless ``path`` matches its pinned digest.

    The same artifact is verified from several call sites per run: weights
    source resolution (``nomikos_inference/weights/__init__.py``), the Hub cache
    (``nomikos_inference/hub/cache.py``), and the architecture adapter
    (``nomikos_inference/architectures/artifact.py``). Hashing is memoized per
    ``(path, size, mtime_ns)`` so those cost one read of the file rather than one
    each; a file that changed on disk is always re-read.
    """
    # stat() first: a missing artifact must still raise FileNotFoundError here,
    # exactly as the unmemoized read did.
    identity = _artifact_identity(path)
    if _VERIFIED_DIGESTS.get(identity) == expected_sha256:
        return

    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        _VERIFIED_DIGESTS.pop(identity, None)
        raise ArtifactIntegrityError(
            f"artifact SHA-256 mismatch for {path}: expected {expected_sha256}, got {actual_sha256}"
        )
    if len(_VERIFIED_DIGESTS) >= _VERIFIED_DIGESTS_MAX:
        _VERIFIED_DIGESTS.clear()
    _VERIFIED_DIGESTS[identity] = actual_sha256


_ARTIFACT_FILENAMES: dict[RegistryArchitecture, tuple[str, ...]] = {
    RegistryArchitecture.calamari: ("best.onnx", "stable.onnx", "model.onnx"),
    RegistryArchitecture.ppocr_rec: ("model.onnx",),
    RegistryArchitecture.blla: ("blla.onnx",),
    RegistryArchitecture.blla_segment: ("blla.onnx",),
    RegistryArchitecture.ppocr_det: ("ppocrv6-det.onnx",),
}

# Spellings the cache saw before the enum existed; they name the same files.
_ARCHITECTURE_ALIASES = {
    "blla_segment": RegistryArchitecture.blla_segment,
    "kraken_segment": RegistryArchitecture.blla,
    "ppocr_det": RegistryArchitecture.ppocr_det,
}


def _normalize_architecture(architecture: str) -> RegistryArchitecture:
    """Map a caller-supplied architecture name onto the enum."""
    if architecture in _ARCHITECTURE_ALIASES:
        return _ARCHITECTURE_ALIASES[architecture]
    return RegistryArchitecture(architecture)


def _first_named_file(cache_dir: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        candidate = cache_dir / name
        if candidate.is_file():
            return candidate
    return None


def find_hub_artifact(cache_dir: Path, *, architecture: str | None) -> Path:
    """Locate the one runtime **Hub artifact** in a cache directory.

    There is exactly one runtime format under ADR 0006, ``.onnx`` for both
    architectures. Ranking multiple candidate formats per architecture is a
    real risk here: ``snapshot_download`` fetches the whole repo revision, and
    these repos publish the native checkpoint beside the graph, so every cache
    directory also holds a ``.pt`` or ``.safetensors`` this runtime must not
    pick up. Naming only ``.onnx`` keeps the choice from being made by
    directory contents instead of by design.
    """
    if architecture is None:
        for path in sorted(cache_dir.glob("*.onnx")):
            if path.is_file():
                return path
        raise FileNotFoundError(f"no supported Hub artifact found in cache directory: {cache_dir}")
    try:
        expected = _ARTIFACT_FILENAMES[_normalize_architecture(architecture)]
    except ValueError as exc:
        raise FileNotFoundError(
            f"unknown architecture {architecture!r}: no Hub artifact mapping"
        ) from exc
    found = _first_named_file(cache_dir, expected)
    if found is None:
        names = ", ".join(expected)
        raise FileNotFoundError(
            f"Hub artifact for architecture {architecture!r} not found in "
            f"cache directory {cache_dir}: expected one of {names}"
        )
    return found
