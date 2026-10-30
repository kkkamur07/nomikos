"""Public inference registry for helper sync (no auth - metadata only)."""

from __future__ import annotations

import hashlib

from fastapi import APIRouter, Request, Response
from fastapi.responses import PlainTextResponse

from backend.core.settings.ml import get_inference_settings
from nomikos_inference.registry import load_registry

router = APIRouter(tags=["ml"])


def _registry_etag(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


@router.get("/inference/v1/registry")
def get_inference_registry(request: Request) -> Response:
    """Serve registry.yaml for installed helpers to sync at startup."""
    registry_path = get_inference_settings().inference_registry_path
    content = registry_path.read_text(encoding="utf-8")
    # Fail fast on deploy if the on-disk registry is invalid.
    load_registry(registry_path)
    etag = _registry_etag(content)
    if request.headers.get("if-none-match") == f'"{etag}"':
        return Response(status_code=304)
    return PlainTextResponse(
        content=content,
        media_type="application/yaml",
        headers={"ETag": f'"{etag}"'},
    )
