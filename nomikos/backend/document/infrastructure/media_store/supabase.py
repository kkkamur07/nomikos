"""Supabase Storage backend for document part page images."""

from datetime import UTC, datetime
from math import ceil
from typing import cast
from uuid import UUID

from supabase import Client, create_client

from backend.core.settings import get_storage_settings
from backend.document.infrastructure.media_store.keys import (
    DEFAULT_PART_IMAGE_SUFFIX,
    part_image_key,
    validate_image_key,
)


class SupabaseMediaStore:
    def __init__(
        self,
        *,
        url: str | None = None,
        service_role_key: str | None = None,
        bucket: str | None = None,
        client: Client | None = None,
    ) -> None:
        settings = get_storage_settings()
        self._bucket = bucket or settings.supabase_storage_bucket
        resolved_url = url or settings.supabase_url
        resolved_key = service_role_key or settings.supabase_service_role_key
        if client is not None:
            self._client = client
            return
        if not resolved_url or not resolved_key:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required when STORAGE_BACKEND=supabase"
            )
        self._client = create_client(resolved_url, resolved_key)

    def part_image_key(
        self,
        part_id: UUID,
        *,
        suffix: str = DEFAULT_PART_IMAGE_SUFFIX,
        filename_stem: str | None = None,
    ) -> str:
        return part_image_key(part_id, suffix=suffix, filename_stem=filename_stem)

    def create_upload_url(self, image_key: str, *, expires_at: datetime) -> tuple[str, str]:
        """A presigned URL the browser PUTs this one object to, plus its token.

        Vercel Functions cap a request body at 4.5 MB, so a manuscript page scan
        (TIFF, multi-MB JPEG, or a full-resolution PNG) cannot be POSTed through
        the API. Presigning an upload lets the browser stream the bytes straight
        to Supabase Storage instead, while the API still owns the object key, the
        validation, and the part row the key is derived from. The returned URL is
        scoped to exactly *image_key* - nothing else in the bucket is writable
        with it. *expires_at* is accepted for protocol symmetry but not honored:
        Storage's signed upload URLs live for its fixed window (currently two
        hours) and the API offers no shorter one.
        """
        validate_image_key(image_key)
        bucket = self._client.storage.from_(self._bucket)
        signed = cast(dict[str, str], bucket.create_signed_upload_url(image_key, options=None))
        url = signed.get("signedUrl") or signed.get("signedURL")
        token = signed.get("token")
        if not url or not token:
            raise RuntimeError(f"Supabase Storage returned no signed upload URL for {image_key}")
        return url, token

    def signed_object_url(self, image_key: str, *, expires_at: datetime) -> str:
        """Storage's own signed link to this one object.

        ``create_signed_url`` signs a single object path, so the link an agent
        receives reaches that page image and nothing else - not the bucket, not
        the ``parts/`` prefix, not the document's other pages. The bytes come
        from Storage directly, which is the point of the whole decision: the
        production API is serverless, and streaming manuscript scans through it
        would cost money for nothing.
        """
        validate_image_key(image_key)
        # Supabase takes a duration, not a deadline. Ceiling rather than floor so
        # a link is never issued already dead; the minimum of one second is what
        # a deliberately-expired deadline collapses to.
        expires_in = max(1, ceil((expires_at - datetime.now(UTC)).total_seconds()))
        signed = self._client.storage.from_(self._bucket).create_signed_url(image_key, expires_in)
        url = signed.get("signedURL") or signed.get("signedUrl")
        if not url:
            raise RuntimeError(f"Supabase Storage returned no signed URL for {image_key}")
        return url

    def write(self, image_key: str, data: bytes) -> None:
        validate_image_key(image_key)
        suffix = image_key.rsplit(".", 1)[-1].lower()
        if suffix in ("png", "jpg", "jpeg", "gif", "webp"):
            content_type = f"image/{'jpeg' if suffix == 'jpg' else suffix}"
        else:
            content_type = "application/octet-stream"
        self._client.storage.from_(self._bucket).upload(
            image_key,
            data,
            file_options={
                "content-type": content_type,
                "upsert": "true",
            },
        )

    def read(self, image_key: str) -> bytes:
        validate_image_key(image_key)
        try:
            return self._client.storage.from_(self._bucket).download(image_key)
        except Exception as exc:
            if _is_not_found(exc):
                raise FileNotFoundError(image_key) from exc
            raise

    def size(self, image_key: str) -> int:
        """Stored byte length from Storage metadata, no download.

        Lets finalize reject an over-cap blob before ``read`` pulls the whole
        object into the API process. Uses ``list`` with an exact-name search on
        the object's own prefix and reads ``metadata.size``.
        """
        validate_image_key(image_key)
        prefix, _, name = image_key.rpartition("/")
        try:
            entries = self._client.storage.from_(self._bucket).list(
                path=prefix or None,
                options={"search": name, "limit": 100},
            )
        except Exception as exc:
            if _is_not_found(exc):
                raise FileNotFoundError(image_key) from exc
            raise
        stored = _listed_size(entries or [], name)
        if stored is None:
            raise FileNotFoundError(image_key)
        return stored

    def delete(self, image_key: str) -> None:
        validate_image_key(image_key)
        self._client.storage.from_(self._bucket).remove([image_key])


def _is_not_found(exc: Exception) -> bool:
    """Whether a Storage error means the object is missing.

    The status code wins where the SDK carries one (``StorageApiError.status``). The
    message is still checked because Storage also answers a missing object with a
    400 whose message says "not found", and older SDK errors carry no status at all.
    """
    if str(getattr(exc, "status", "")) == "404":
        return True
    message = str(exc).lower()
    return "not found" in message or "404" in message


def _listed_size(entries: list[dict], name: str) -> int | None:
    """``metadata.size`` (or ``contentLength``) of the entry called exactly ``name``."""
    for entry in entries:
        if entry.get("name") != name:
            continue
        metadata = entry.get("metadata") or {}
        stored = metadata.get("size", metadata.get("contentLength"))
        if stored is not None:
            return int(stored)
    return None
