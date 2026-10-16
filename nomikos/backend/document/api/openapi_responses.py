"""OpenAPI ``responses=`` entries for the routes that answer with raw bytes.

Shared by the member routes in ``documents`` and the public routes in ``public`` so the
two describe the same artifact the same way.
"""

from __future__ import annotations

#: The shape FastAPI's ``responses=`` argument takes: status code to an OpenAPI
#: response object.
OpenApiResponses = dict[int | str, dict[str, object]]


def _binary_response(media_type: str, description: str) -> OpenApiResponses:
    return {
        200: {
            "content": {media_type: {"schema": {"type": "string", "format": "binary"}}},
            "description": description,
        }
    }


PDF_RESPONSE = _binary_response("application/pdf", "Transcription PDF bytes")
XML_RESPONSE = _binary_response("application/xml", "PAGE XML bytes")
ZIP_RESPONSE = _binary_response(
    "application/zip",
    "Zip of the PAGE XML and the full-resolution page image it describes",
)
DOCUMENT_ZIP_RESPONSE = _binary_response(
    "application/zip",
    "Zip of one PAGE XML and one full-resolution image per page",
)
TEXT_RESPONSE: OpenApiResponses = {
    200: {
        "content": {"text/plain": {"schema": {"type": "string"}}},
        "description": "Plain-text transcription, one page after another",
    }
}
