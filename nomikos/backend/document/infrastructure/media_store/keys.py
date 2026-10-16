"""Logical object keys for document part page images."""

import re
from uuid import UUID

_SAFE_IMAGE_KEY = re.compile(
    r"^parts/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    r"(?:/[a-z0-9][a-z0-9_-]{0,127})?\.[a-z0-9]{1,16}$"
)

DEFAULT_PART_IMAGE_SUFFIX = "webp"


def part_image_key(
    part_id: UUID, *, suffix: str = DEFAULT_PART_IMAGE_SUFFIX, filename_stem: str | None = None
) -> str:
    safe = re.sub(r"[^a-z0-9]", "", suffix.lstrip(".").lower())[:16] or DEFAULT_PART_IMAGE_SUFFIX
    if filename_stem is None:
        return f"parts/{part_id}.{safe}"
    stem = re.sub(r"[^a-z0-9_-]", "-", filename_stem.lower()).strip("-_")[:128]
    if not stem:
        stem = str(part_id)
    return f"parts/{part_id}/{stem}.{safe}"


def validate_image_key(image_key: str) -> None:
    if not image_key or image_key.startswith("/"):
        raise ValueError("Invalid image key")
    if ".." in image_key.split("/"):
        raise ValueError("Invalid image key")
    if not _SAFE_IMAGE_KEY.match(image_key):
        raise ValueError("Invalid image key")


def derived_image_key(image_key: str, *, width: int, encoder_version: str) -> str:
    """Where a persisted thumbnail of ``image_key`` is stored.

    Same grammar as a part key, under the same part's folder, so it passes
    ``validate_image_key`` on every backend and is deleted by key like any other
    object. The encoder version is part of the name: a new encoder never serves
    a stale rendering, it just leaves the old file for the part's deletion to
    sweep.

    ``parts/<id>.webp`` -> ``parts/<id>/thumb-w200-webp-q85-v1.webp``
    ``parts/<id>/page.webp`` -> ``parts/<id>/page-thumb-w200-webp-q85-v1.webp``
    """
    validate_image_key(image_key)
    if width <= 0:
        raise ValueError("Thumbnail width must be positive")
    version = re.sub(r"[^a-z0-9_-]", "-", encoder_version.lower()).strip("-_") or "v0"
    head = image_key.rsplit(".", 1)[0]
    folder, _, stem = head.partition("/")
    part_id, _, filename_stem = stem.partition("/")
    variant = f"thumb-w{width}-{version}"
    if filename_stem:
        variant = f"{filename_stem[:96]}-{variant}"
    derived = f"{folder}/{part_id}/{variant}.webp"
    validate_image_key(derived)
    return derived


def is_pending_key(image_key: str) -> bool:
    """Whether a part row is still waiting for its direct upload to be finalized."""
    return image_key.startswith("pending")


def minted_pending_key(image_key: str) -> str | None:
    """The object key a pending row's ``pending:<key>`` sentinel carries, if any.

    A bare ``pending`` row never minted a key, so it has no blob to delete; anything
    that is not a sentinel at all is not a pending key either.
    """
    minted = image_key.removeprefix("pending:")
    if not minted or minted == image_key:
        return None
    return minted
