"""Document and DocumentPart routes under projects.

One ``router`` aggregates the per-resource routers; the app includes it as before.
"""

from __future__ import annotations

from fastapi import APIRouter

from backend.document.api.documents import (
    documents_crud,
    exports,
    layout,
    parts,
    transcription,
)

router = APIRouter()
for _resource in (documents_crud, parts, layout, transcription, exports):
    router.include_router(_resource.router)
