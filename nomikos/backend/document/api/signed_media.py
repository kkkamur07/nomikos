"""Serve one object to whoever holds a valid signature for it.

**The signature is the authorization.** There is no session dependency here, no
device token, and no database session at all - deliberately. ADR 0002 rejected an
authenticated ``GET /device/v1/jobs/{id}/image`` for two reasons, and both are
visible in the shape of this module: it would have put a route on the device
credential that must independently re-derive job ownership, and it would have
made the platform the thing that streams manuscript scans. This route derives
nothing and, in the deployment that matters, is never reached.

That last part is why it refuses unless ``STORAGE_BACKEND=local``. Production
runs on Supabase, where Storage checks its own signature and hands the bytes over
directly; the production API is serverless, so a scan flowing through it costs
money for nothing. A filesystem cannot answer an HTTP request on its own, so the
platform answers for it - on a development and self-hosted backend only, and the
guard is what keeps that from quietly becoming true everywhere.

Everything that can be wrong with a request answers 403: a malformed key, a
forged digest, an expired deadline. Distinguishing them would turn the route into
an oracle for which object keys exist.
"""

from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import Response

from backend.core.schemas.errors import ApiErrorResponse
from backend.core.settings import get_storage_settings
from backend.document.api.media_responses import media_type_for_image_key
from backend.document.infrastructure.media_store import get_media_store, signature_is_valid

router = APIRouter(prefix="/media/signed", tags=["media"])

#: A URL carrying a bearer signature must not be written to a shared cache, and
#: there is nothing to revalidate against inside its one-minute life anyway.
SIGNED_MEDIA_CACHE_CONTROL = "private, no-store"

SIGNED_MEDIA_RESPONSES: dict = {
    200: {
        "content": {
            "image/png": {"schema": {"type": "string", "format": "binary"}},
            "image/jpeg": {"schema": {"type": "string", "format": "binary"}},
            "image/webp": {"schema": {"type": "string", "format": "binary"}},
            "application/octet-stream": {"schema": {"type": "string", "format": "binary"}},
        },
        "description": "Page image bytes",
    },
    403: {"model": ApiErrorResponse, "description": "Missing, forged, or expired signature"},
    404: {"model": ApiErrorResponse, "description": "No such object"},
}


@router.get("/{image_key:path}", response_class=Response, responses=SIGNED_MEDIA_RESPONSES)
async def get_signed_object(
    image_key: str,
    expires: Annotated[int, Query(description="Unix seconds after which the link is dead.")],
    signature: Annotated[
        str,
        Query(min_length=1, max_length=256, description="HMAC over the object key and expiry."),
    ],
) -> Response:
    if get_storage_settings().storage_backend != "local":
        # Nothing mints one of these against object storage, so answering here
        # would only be a second, unsigned-by-Storage way to reach a bucket.
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    if not signature_is_valid(image_key, expires=expires, signature=signature):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
    try:
        # A blocking filesystem read; off the event loop like every other media read.
        data = await asyncio.to_thread(get_media_store().read, image_key)
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from None
    return Response(
        content=data,
        media_type=media_type_for_image_key(image_key),
        headers={"Cache-Control": SIGNED_MEDIA_CACHE_CONTROL},
    )
