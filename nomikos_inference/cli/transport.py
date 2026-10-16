"""How the CLI reaches the platform: URL policy, and the HTTP transport.

The URL policy is checked once, where a client is built, rather than per call:
`--api-url`, `$NOMIKOS_API_URL`, and the stored credential are three doors into
the same room, and a check on one of them is a check on none of them.

The transport never follows redirects. The stdlib's redirect handler copies
every non-content header - including the bearer tokens this CLI sends - to
whatever `Location` names, so one 302 from a proxy, CDN rule, or misconfigured
gateway would deliver the credential to another host (and possibly over
cleartext). A platform that answers with a redirect is misbehaving, and the
CLI says so instead of following it.
"""

from __future__ import annotations

import os
import platform as platform_module
import socket
import urllib.error
import urllib.request
from email.message import Message
from typing import Protocol
from urllib.parse import urlsplit

from nomikos_inference.cli.errors import InsecurePlatformURL, PlatformError

DEFAULT_PLATFORM_URL = "https://api.nomikos.app"
PLATFORM_URL_ENV = "NOMIKOS_API_URL"

# The platform caps these on the way in (`PairingStartRequest`). Truncating
# here keeps a long hostname from turning into a 422 the researcher can't act on.
DEVICE_NAME_LIMIT = 120
PLATFORM_LIMIT = 32
VERSION_LIMIT = 32

REQUEST_TIMEOUT_SECONDS = 30.0

JSON_RESPONSE_LIMIT = 1 << 20
"""Ceiling on a decoded JSON body. Claim, self, callback, and floor answers are
all small records; a response that does not fit in a megabyte is not one of
them, and reading it unbounded would let a hostile or broken endpoint force an
arbitrarily large allocation on a researcher's laptop."""

IMAGE_TIMEOUT_SECONDS = 120.0
"""Generous because this fetches a scan, not a JSON round trip, but still
bounded: the **signed page image link** dies in about a minute anyway, so a
fetch that hasn't finished well past that isn't going to."""

MAX_PAGE_IMAGE_BYTES_ENV = "NOMIKOS_MAX_PAGE_IMAGE_BYTES"


def max_page_image_bytes() -> int:
    """Ceiling on a downloaded page image, loose and overridable.

    A signed link names exactly one object and expires in about a minute, so
    the practical bound is the platform's own upload cap, not this number. It
    exists only to stop a slow-drip or hostile response from exhausting memory,
    so the default is generous; raise it via ``NOMIKOS_MAX_PAGE_IMAGE_BYTES``
    if needed.
    """
    raw = os.environ.get(MAX_PAGE_IMAGE_BYTES_ENV)
    if raw is None:
        return 1024 * 1024 * 1024
    try:
        parsed = int(raw)
    except ValueError as exc:
        raise ValueError(f"{MAX_PAGE_IMAGE_BYTES_ENV} must be an integer") from exc
    return max(1, parsed)


LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
"""The only hosts this CLI will talk to without TLS.

Every request on this surface carries a bearer credential (the 180-day
**device token**, or a **service credential** with wider scope) in a header.
Over cleartext HTTP those are readable by anything on the path, so the scheme
is checked once, where the client is built, rather than per call. Loopback is
the exception because nothing leaves the machine: it's what lets the
integration suite stand up a platform on `http://127.0.0.1:<port>`.
"""


def default_platform_url(environ: dict[str, str] | None = None) -> str:
    source = environ if environ is not None else os.environ
    return (source.get(PLATFORM_URL_ENV) or DEFAULT_PLATFORM_URL).rstrip("/")


def _is_loopback(host: str | None) -> bool:
    """Whether a hostname names this machine. `urlsplit` has already
    lowercased it and stripped the brackets off an IPv6 literal."""
    return bool(host) and host in LOOPBACK_HOSTS


def is_secure_platform_url(base_url: str) -> bool:
    """Whether a bearer credential may be sent to `base_url`."""
    parts = urlsplit(base_url)
    if parts.scheme == "https":
        return True
    return parts.scheme == "http" and _is_loopback(parts.hostname)


def require_secure_platform_url(base_url: str) -> str:
    """The trimmed URL, or `InsecurePlatformURL` if it would leak a credential.

    Enforced once, where a client is built, rather than at each call site that
    supplies a URL: `--api-url`, `$NOMIKOS_API_URL`, and the stored credential
    are three doors into the same room, and a check on one of them is a check
    on none of them.
    """
    trimmed = base_url.rstrip("/")
    if is_secure_platform_url(trimmed):
        return trimmed
    scheme = urlsplit(trimmed).scheme or "no"
    raise InsecurePlatformURL(
        f"Refusing to talk to {trimmed or 'an empty URL'} over {scheme} scheme. "
        "Every request carries this machine's device token, so the platform URL "
        "must be https (or http on localhost for a development platform)."
    )


def this_machine_name() -> str:
    """What the consent screen will call this computer.

    Self-reported and rendered as inert text by the `/pair` page: a convincing
    name is not evidence, and the **confirmation code** is what a researcher
    actually checks.
    """
    name = socket.gethostname().strip() or "unnamed machine"
    return name[:DEVICE_NAME_LIMIT]


def this_machine_platform() -> str:
    machine = platform_module.machine() or "unknown"
    return f"{platform_module.system().lower()}-{machine}"[:PLATFORM_LIMIT]


class ResponseBody(Protocol):
    """What `_read_bounded` needs: headers to pre-check, and chunked reads."""

    headers: Message
    """Present on both success responses and `HTTPError`s."""

    def read(self, amt: int = ...) -> bytes: ...


class HttpResponse(ResponseBody, Protocol):
    """What `open_request` yields: a body with a status, usable as a context
    manager. The `status` is typed `int` (not `int | None` as on
    `addinfourl`): this opener only speaks HTTP, where every answer carries
    one."""

    status: int

    def __enter__(self) -> HttpResponse: ...
    def __exit__(self, exc_type: object, exc: object, tb: object) -> None: ...


def _read_bounded(
    response: ResponseBody,
    limit: int,
    *,
    what: str = "the page image",
) -> bytes:
    """Read a response body, refusing to hold more than ``limit`` bytes.

    Streamed rather than ``response.read()`` so a response whose size was not
    announced (or was lied about) cannot force a single unbounded allocation.
    """
    content_length = response.headers.get("Content-Length")
    if content_length is not None:
        try:
            if int(content_length) > limit:
                raise PlatformError(f"{what} exceeded the size limit")
        except ValueError:
            pass

    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = response.read(1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        total += len(chunk)
        if total > limit:
            raise PlatformError(f"{what} exceeded the size limit")
        chunks.append(chunk)


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """A redirect handler that refuses instead of following.

    The stdlib's `redirect_request` copies every non-content header to the new
    URL, which on this surface means the device or service token. Refusing is
    the whole implementation: raising here aborts the request before a second
    one is made, so no credential ever leaves the first, https-checked host.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise PlatformError(
            f"Refusing to follow a {code} redirect to {newurl}. "
            "The platform must answer directly; following the redirect would "
            "resend this request's credential headers to a URL it chose."
        )


def _platform_opener() -> urllib.request.OpenerDirector:
    """An opener identical to the default except that it refuses redirects.

    `build_opener` leaves out a default handler when a passed handler
    subclasses it, so this keeps the stdlib's proxy, auth, and error handling
    exactly, with `_NoRedirectHandler` the only redirect handler installed.
    """
    return urllib.request.build_opener(_NoRedirectHandler)


_OPENER = _platform_opener()


def open_request(request: urllib.request.Request, timeout: float) -> HttpResponse:
    """Open one request through the no-redirect opener.

    A module-level function (rather than a direct `urlopen` call) so tests can
    swap the transport underneath the client: they replace `_OPENER` with a
    stub, never the network.
    """
    return _OPENER.open(request, timeout=timeout)


def require_fetchable(url: str, *, plaintext_loopback: bool) -> None:
    """Refuse a page image link that is not an HTTPS URL.

    Plain HTTP is allowed only when the client is itself pointed at a loopback
    platform (the integration suite, and nothing else). The page image carries
    no credential, so this isn't about eavesdropping: it's that a URL the
    platform chose must not be able to name a scheme that reads this machine
    instead of the network. Redirects are refused by the opener, so the scheme
    checked here is the scheme that is fetched.
    """
    parts = urlsplit(url)
    if parts.scheme == "https":
        return
    if parts.scheme == "http" and plaintext_loopback and _is_loopback(parts.hostname):
        return
    raise PlatformError(
        f"Refusing to fetch a page image over {parts.scheme or 'a URL with no'} scheme. "
        "A page image link must be https."
    )


__all__ = [
    "DEFAULT_PLATFORM_URL",
    "DEVICE_NAME_LIMIT",
    "IMAGE_TIMEOUT_SECONDS",
    "JSON_RESPONSE_LIMIT",
    "LOOPBACK_HOSTS",
    "MAX_PAGE_IMAGE_BYTES_ENV",
    "PLATFORM_LIMIT",
    "PLATFORM_URL_ENV",
    "REQUEST_TIMEOUT_SECONDS",
    "VERSION_LIMIT",
    "_OPENER",
    "default_platform_url",
    "is_secure_platform_url",
    "max_page_image_bytes",
    "open_request",
    "require_fetchable",
    "require_secure_platform_url",
    "this_machine_name",
    "this_machine_platform",
]
