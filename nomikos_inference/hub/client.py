"""Mockable Hugging Face Hub download client."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class HubClient(Protocol):
    def snapshot_download(self, repo_id: str, revision: str, local_dir: Path) -> None: ...


def _hub_error_message(exc: Exception, *, repo_id: str, revision: str) -> str:
    name = type(exc).__name__
    message = str(exc).strip()

    if name == "RepositoryNotFoundError":
        return (
            f"Hub model repo not found: {repo_id}. "
            "Check the namespace and hub repo slug in the registry weights source."
        )
    if name == "RevisionNotFoundError":
        return (
            f"Hub revision not found: {repo_id}@{revision}. "
            "Check the registry tag maps to an existing git tag or commit on the Hub repo."
        )
    if name in {"GatedRepoError", "HfHubHTTPError"} and (
        "401" in message or "403" in message or "authorized" in message.lower()
    ):
        return (
            f"Hub access denied for {repo_id}@{revision}. "
            "Set HF_TOKEN only for private or gated repos; public nomikos repos do not require a token."
        )
    if name == "OfflineModeIsEnabled":
        return (
            f"Hub is offline and {repo_id}@{revision} is not cached. "
            "Run scripts/hf/fetch_model.py while online or use local bundled weights."
        )

    return (
        f"Hub download failed for {repo_id}@{revision}: {message or name}. "
        "Verify the repo is public, the revision exists, and network access is available."
    )


class HuggingFaceHubClient:
    def snapshot_download(self, repo_id: str, revision: str, local_dir: Path) -> None:
        """Download one pinned revision; errors are wrapped by ``_snapshot_download``."""
        from huggingface_hub import snapshot_download

        local_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=str(local_dir),
            repo_type="model",
        )


_default_client: HubClient | None = None


def get_default_hub_client() -> HubClient:
    global _default_client
    if _default_client is None:
        _default_client = HuggingFaceHubClient()
    return _default_client


def set_default_hub_client(client: HubClient | None) -> None:
    global _default_client
    _default_client = client
