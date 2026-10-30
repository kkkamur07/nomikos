"""Resolve hf:// weights sources to local Hub cache paths."""

from __future__ import annotations

import logging
import os
import shutil
from contextlib import contextmanager
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX; the research tooling is macOS/Linux
    fcntl = None  # type: ignore[assignment]

from nomikos_inference.hub.artifacts import find_hub_artifact, verify_artifact_sha256
from nomikos_inference.hub.client import HubClient, _hub_error_message, get_default_hub_client
from nomikos_inference.hub.manifest import (
    HubCacheManifest,
    load_manifest,
    manifest_matches_expected,
    save_manifest,
)
from nomikos_inference.hub.pins import COMMIT_SHA_PATTERN, SHA256_PATTERN
from nomikos_inference.hub.uri import parse_hf_weights_uri

logger = logging.getLogger(__name__)

# The **Hub cache** lives under the researcher's home directory, not beside the
# code, because this module ships inside the published `nomikos-inference`
# wheel: a repository-relative path would write into site-packages, which is
# unwritable for a system install and gets silently discarded on upgrade.
# `~/.nomikos/` is the same root the CLI keeps its device credential under.
DEFAULT_CACHE_ROOT = Path.home() / ".nomikos" / "hf" / "cache"
CACHE_ROOT_ENV = "HF_CACHE_ROOT"


def default_cache_root() -> Path:
    """Where **Hub artifact**s are cached, overridable by ``HF_CACHE_ROOT``.

    Read from the environment on every call rather than at import time: tests and
    the CLI both set it after this module is already loaded.
    """
    override = os.environ.get(CACHE_ROOT_ENV)
    if override:
        return Path(override).expanduser()
    return DEFAULT_CACHE_ROOT


def cache_dir_for(
    registry_model_id: str,
    registry_tag: str,
    *,
    cache_root: Path | None = None,
) -> Path:
    # The registry lookup already insists these are exact keys, but this function
    # is also callable directly: a segment containing a separator or ``..`` must
    # not be able to write outside the cache root.
    for label, value in (
        ("registry_model_id", registry_model_id),
        ("registry_tag", registry_tag),
    ):
        if not value or value in {".", ".."} or "/" in value or "\\" in value:
            raise ValueError(f"{label} must be a single path segment")
    root = cache_root or default_cache_root()
    return root / registry_model_id / registry_tag


def _validate_provenance(
    *, hub_revision: str | None, artifact_sha256: str | None
) -> tuple[str, str]:
    """Both pins, proven present and well-formed, or ``ValueError``.

    Returns them instead of ``None`` so the caller gets narrowing from the
    signature rather than needing its own ``assert`` calls, which `python -O`
    strips (and would silently allow an unpinned download).
    """
    if not hub_revision or not COMMIT_SHA_PATTERN.fullmatch(hub_revision):
        raise ValueError(
            "hf weights source requires an immutable 40-character lowercase Hub commit in hub_revision"
        )
    if not artifact_sha256 or not SHA256_PATTERN.fullmatch(artifact_sha256):
        raise ValueError("hf weights source requires a 64-character lowercase artifact_sha256")
    return hub_revision, artifact_sha256


@contextmanager
def _cache_lock(cache_dir: Path):
    """Serialize resolve of one ``(model, tag)`` across processes and threads.

    Without it, two first-time resolves of the same model both see no manifest,
    both ``rmtree`` the cache dir and both ``snapshot_download`` into it - one
    deleting the other's half-written snapshot. The lock file lives *beside*
    ``cache_dir`` (not inside it) because the resolve body ``rmtree``s the dir
    itself. A no-op when ``fcntl`` is unavailable rather than a hard failure.
    """
    if fcntl is None:
        yield
        return
    lock_path = cache_dir.parent / f".{cache_dir.name}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _snapshot_download(
    client: HubClient,
    repo_id: str,
    revision: str,
    local_dir: Path,
) -> None:
    try:
        client.snapshot_download(repo_id, revision, local_dir)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(_hub_error_message(exc, repo_id=repo_id, revision=revision)) from exc


def _try_cached_artifact(
    cache_dir: Path,
    repo_id: str,
    hub_revision: str,
    artifact_sha256: str,
    architecture: str | None,
) -> Path | None:
    """A verified cached artifact, or ``None`` (with a warning) on any failure."""
    manifest = load_manifest(cache_dir)
    if manifest is None or not manifest_matches_expected(
        manifest,
        repo_id=repo_id,
        hub_revision=hub_revision,
        artifact_sha256=artifact_sha256,
    ):
        return None
    try:
        artifact = find_hub_artifact(cache_dir, architecture=architecture)
        if str(artifact.relative_to(cache_dir)) != manifest.artifact_path:
            raise ValueError("cached Hub artifact path does not match its manifest")
        verify_artifact_sha256(artifact, artifact_sha256)
        return artifact
    except (FileNotFoundError, ValueError) as exc:
        logger.warning(
            "cached Hub artifact failed verification, re-downloading: %s",
            exc,
        )
        return None


def _download_artifact(
    cache_dir: Path,
    client: HubClient,
    repo_id: str,
    hub_revision: str,
    artifact_sha256: str,
    architecture: str | None,
) -> Path:
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    try:
        _snapshot_download(client, repo_id, hub_revision, cache_dir)
        artifact = find_hub_artifact(cache_dir, architecture=architecture)
        verify_artifact_sha256(artifact, artifact_sha256)
        save_manifest(
            cache_dir,
            HubCacheManifest(
                repo_id=repo_id,
                hub_revision=hub_revision,
                artifact_path=str(artifact.relative_to(cache_dir)),
                artifact_sha256=artifact_sha256,
            ),
        )
        return artifact
    except Exception:
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        raise


def resolve_hf_weights_source(
    uri: str,
    *,
    registry_model_id: str,
    registry_tag: str,
    hub_revision: str | None,
    artifact_sha256: str | None,
    architecture: str | None = None,
    hub_client: HubClient | None = None,
    cache_root: Path | None = None,
) -> Path:
    parsed = parse_hf_weights_uri(uri)
    if parsed.registry_tag != registry_tag:
        raise ValueError(
            f"hf weights source registry tag {parsed.registry_tag!r} "
            f"does not match requested registry tag {registry_tag!r}"
        )
    hub_revision, artifact_sha256 = _validate_provenance(
        hub_revision=hub_revision,
        artifact_sha256=artifact_sha256,
    )

    client = hub_client or get_default_hub_client()
    resolved_cache_root = cache_root or default_cache_root()
    cache_dir = cache_dir_for(registry_model_id, registry_tag, cache_root=resolved_cache_root)

    # Held across the whole check-then-download so a concurrent resolve of the
    # same model cannot rmtree the snapshot this one is verifying or writing.
    with _cache_lock(cache_dir):
        cached = _try_cached_artifact(
            cache_dir, parsed.repo_id, hub_revision, artifact_sha256, architecture
        )
        if cached is not None:
            return cached
        return _download_artifact(
            cache_dir, client, parsed.repo_id, hub_revision, artifact_sha256, architecture
        )
