"""Hugging Face Hub publish client."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

# The two item kinds the `nomos` **Hub collection** holds. Narrower than `str`
# because `HfApi.add_collection_item` types this parameter as a Literal, and
# a plain `str` here only moved the mismatch to the one call site that cannot
# check it. `sync.py` passes these two spellings and nothing else.
CollectionItemType = Literal["model", "dataset"]


class PublishClient(Protocol):
    def create_repo(
        self,
        repo_id: str,
        *,
        repo_type: str,
        private: bool = False,
    ) -> None: ...

    def upload_folder(
        self,
        folder_path: Path,
        repo_id: str,
        *,
        repo_type: str,
        commit_message: str,
    ) -> str: ...

    def create_tag(
        self,
        repo_id: str,
        *,
        tag: str,
        revision: str,
        repo_type: str,
    ) -> None: ...

    def update_collection_metadata(
        self,
        collection_slug: str,
        *,
        title: str | None = None,
        description: str | None = None,
    ) -> None: ...

    def add_collection_item(
        self,
        collection_slug: str,
        *,
        item_id: str,
        item_type: CollectionItemType,
        note: str | None = None,
    ) -> None: ...

    def get_collection(self, collection_slug: str): ...


class HuggingFacePublishClient:
    def create_repo(
        self,
        repo_id: str,
        *,
        repo_type: str,
        private: bool = False,
    ) -> None:
        from huggingface_hub import HfApi

        HfApi().create_repo(
            repo_id=repo_id,
            repo_type=repo_type,
            private=private,
            exist_ok=True,
        )

    def upload_folder(
        self,
        folder_path: Path,
        repo_id: str,
        *,
        repo_type: str,
        commit_message: str,
    ) -> str:
        from huggingface_hub import HfApi

        info = HfApi().upload_folder(
            folder_path=str(folder_path),
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=commit_message,
        )
        sha = getattr(info, "oid", None) or getattr(info, "commit_sha", None)
        if not sha and hasattr(info, "commit_url"):
            sha = str(info.commit_url).rsplit("/", 1)[-1]
        if not sha:
            raise ValueError(f"upload to {repo_id} did not return a revision sha")
        return str(sha)

    def create_tag(
        self,
        repo_id: str,
        *,
        tag: str,
        revision: str,
        repo_type: str,
    ) -> None:
        """Point ``tag`` at ``revision``, moving it when it already exists.

        The tag is created first so a failure leaves the previous tag in
        place; only when the tag already exists is it deleted and recreated.
        Serving pins commits, but ``@stable`` readers break while the tag
        is missing.
        """
        from huggingface_hub import HfApi
        from huggingface_hub.utils import HfHubHTTPError

        api = HfApi()
        try:
            api.create_tag(
                repo_id=repo_id,
                tag=tag,
                revision=revision,
                repo_type=repo_type,
                exist_ok=False,
            )
        except HfHubHTTPError:
            try:
                api.delete_tag(repo_id=repo_id, tag=tag, repo_type=repo_type)
            except HfHubHTTPError as error:
                if error.response is None or error.response.status_code != 404:
                    raise
            api.create_tag(
                repo_id=repo_id,
                tag=tag,
                revision=revision,
                repo_type=repo_type,
                exist_ok=False,
            )

    def update_collection_metadata(
        self,
        collection_slug: str,
        *,
        title: str | None = None,
        description: str | None = None,
    ) -> None:
        from huggingface_hub import HfApi

        HfApi().update_collection_metadata(
            collection_slug=collection_slug,
            title=title,
            description=description,
        )

    def add_collection_item(
        self,
        collection_slug: str,
        *,
        item_id: str,
        item_type: CollectionItemType,
        note: str | None = None,
    ) -> None:
        from huggingface_hub import HfApi

        HfApi().add_collection_item(
            collection_slug=collection_slug,
            item_id=item_id,
            item_type=item_type,
            note=note,
            exists_ok=True,
        )

    def get_collection(self, collection_slug: str):
        from huggingface_hub import HfApi

        return HfApi().get_collection(collection_slug)


_default_client: PublishClient | None = None


def get_default_publish_client() -> PublishClient:
    global _default_client
    if _default_client is None:
        _default_client = HuggingFacePublishClient()
    return _default_client


def set_default_publish_client(client: PublishClient | None) -> None:
    global _default_client
    _default_client = client


def upload_enabled(*, upload_flag: bool) -> bool:
    import os

    if upload_flag:
        return True
    return os.environ.get("HF_PUBLISH", "").strip().lower() in {"1", "true", "yes"}
