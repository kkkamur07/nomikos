"""Per-client ceilings on the unauthenticated read surface.

The thumbnail route was the only public route that metered anything, and it is the
cheapest one on the surface: a decode plus a resize, served from a render cache. The
expensive ones - an fpdf2 render of a whole page, a PAGE XML export, a layout read
that joins lines to their transcriptions - answered anonymous callers as fast as they
could be asked. Each of them is a better lever on the process than the one route that
was guarded.

Where the address does not identify one client, nothing is charged. That is the same
answer the thumbnail throttle already gave, and the reason is worth keeping in front of
whoever changes this: behind a proxy tier the deployment cannot allowlist, every visitor
shares the peer address, so a bucket keyed on it is a single global bucket. Filling it is
then an outage for every reader at once, which is a worse outcome than the flood it was
meant to stop. The bounds that still hold in that case are the page size on the layout
read and the render cache on the thumbnail.
"""

from __future__ import annotations

from fastapi import Request

from backend.users.api.rate_limit import attributable_client_ip, consume_rate_limit

PUBLIC_RATE_LIMIT_WINDOW_SECONDS = 60

#: Generous enough for a reader opening a long document (each page is one request, and
#: revalidations are not charged), tight enough that scripted enumeration of published
#: parts cannot keep the encoder saturated.
THUMBNAIL_RATE_LIMIT_REQUESTS = 240

#: Metadata and geometry reads. One published document costs a handful of these, so a
#: reader browsing normally never approaches it.
PUBLIC_READ_RATE_LIMIT_REQUESTS = 120

#: Whole-page artifact renders: the transcription PDF and the PAGE XML export. Each one
#: is a synchronous render over every line on the page, so the budget is small - a reader
#: downloads one of these per page they care about, not per page they look at.
PUBLIC_EXPORT_RATE_LIMIT_REQUESTS = 20


async def _throttle(request: Request, *, bucket: str, limit: int, detail: str) -> None:
    client_ip = attributable_client_ip(request)
    if client_ip is None:
        return
    await consume_rate_limit(
        [f"{bucket}:{client_ip}"],
        limit=limit,
        window_seconds=PUBLIC_RATE_LIMIT_WINDOW_SECONDS,
        detail=detail,
    )


async def throttle_public_read(request: Request) -> None:
    """Cap anonymous reads of a published document per client."""
    await _throttle(
        request,
        bucket="public-read",
        limit=PUBLIC_READ_RATE_LIMIT_REQUESTS,
        detail="Too many requests; try again later",
    )


async def throttle_public_export(request: Request) -> None:
    """Cap anonymous whole-page artifact renders per client.

    Charged on top of the read budget, because these are the two routes on the surface
    that cost a full render: the PDF reloads a TrueType face per span and lays out every
    line, and the PAGE XML walks the same geometry.
    """
    await _throttle(
        request,
        bucket="public-export",
        limit=PUBLIC_EXPORT_RATE_LIMIT_REQUESTS,
        detail="Too many export requests; try again later",
    )


async def throttle_public_thumbnail(request: Request) -> None:
    """Cap anonymous thumbnail renders per client.

    Only the resized variant is charged: a full-size read streams bytes that already
    exist in storage, while a thumbnail can cost a decode plus a LANCZOS resize whenever
    the render cache misses.
    """
    await _throttle(
        request,
        bucket="public-thumbnail",
        limit=THUMBNAIL_RATE_LIMIT_REQUESTS,
        detail="Too many thumbnail requests; try again later",
    )
