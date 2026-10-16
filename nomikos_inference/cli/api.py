"""HTTP client for the platform's device protocols.

Pairing is RFC 8628's device authorization grant with the typable `user_code`
removed (`backend/ml/api/device_pairing.py`); the **claim** loop is the one
endpoint ADR 0003 costs, plus the platform's existing job callback
(`backend/jobs/api/device_claim.py`, `backend/jobs/api/internal_inference.py`).
This is a client for wire contracts that are already settled, not a design
surface, and the shapes below are the platform's DTOs read back.

`urllib.request` rather than an HTTP library, because `[project].dependencies`
is the closure that reaches a researcher's laptop, and neither protocol needs
streaming, connection reuse, or a negotiated auth scheme. One page per claim
(ADR 0002) keeps that true: no long-lived transfer to manage, just a request
per page.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from urllib.parse import urlsplit

from nomikos_inference.cli.dto import (
    AGENT_VERSION_REFUSED_STATUS,
    AgentFloor,
    Claim,
    ClaimedPage,
    DeviceIdentity,
    PairingPoll,
    StartedPairing,
    _agent_notice,
    _claimed_page,
    _decode,
    _parse_datetime,
    _refusal_floor,
    _unexpected,
    _version_refusal,
)
from nomikos_inference.cli.dto import (
    AGENT_VERSION_UNSUPPORTED as AGENT_VERSION_UNSUPPORTED,
)
from nomikos_inference.cli.dto import (
    STATUS_ACCESS_DENIED as STATUS_ACCESS_DENIED,
)
from nomikos_inference.cli.dto import (
    STATUS_APPROVED as STATUS_APPROVED,
)
from nomikos_inference.cli.dto import (
    STATUS_AUTHORIZATION_PENDING as STATUS_AUTHORIZATION_PENDING,
)
from nomikos_inference.cli.dto import (
    STATUS_EXPIRED as STATUS_EXPIRED,
)
from nomikos_inference.cli.dto import (
    STATUS_SLOW_DOWN as STATUS_SLOW_DOWN,
)
from nomikos_inference.cli.dto import (
    AgentNotice as AgentNotice,
)
from nomikos_inference.cli.errors import (
    AgentVersionRefused as AgentVersionRefused,
)
from nomikos_inference.cli.errors import (
    CredentialRefused,
    PageLeaseLost,
    PlatformError,
)
from nomikos_inference.cli.errors import (
    InsecurePlatformURL as InsecurePlatformURL,
)
from nomikos_inference.cli.transport import (
    DEFAULT_PLATFORM_URL as DEFAULT_PLATFORM_URL,
)
from nomikos_inference.cli.transport import (
    DEVICE_NAME_LIMIT as DEVICE_NAME_LIMIT,
)
from nomikos_inference.cli.transport import (
    IMAGE_TIMEOUT_SECONDS,
    JSON_RESPONSE_LIMIT,
    REQUEST_TIMEOUT_SECONDS,
    VERSION_LIMIT,
    _read_bounded,
    max_page_image_bytes,
    open_request,
    require_fetchable,
    require_secure_platform_url,
)
from nomikos_inference.cli.transport import (
    LOOPBACK_HOSTS as LOOPBACK_HOSTS,
)
from nomikos_inference.cli.transport import (
    MAX_PAGE_IMAGE_BYTES_ENV as MAX_PAGE_IMAGE_BYTES_ENV,
)
from nomikos_inference.cli.transport import (
    PLATFORM_LIMIT as PLATFORM_LIMIT,
)
from nomikos_inference.cli.transport import (
    PLATFORM_URL_ENV as PLATFORM_URL_ENV,
)
from nomikos_inference.cli.transport import (
    default_platform_url as default_platform_url,
)
from nomikos_inference.cli.transport import (
    is_secure_platform_url as is_secure_platform_url,
)
from nomikos_inference.cli.transport import (
    this_machine_name as this_machine_name,
)
from nomikos_inference.cli.transport import (
    this_machine_platform as this_machine_platform,
)
from nomikos_inference.cli.version import installed_version

DEVICE_TOKEN_HEADER = "X-Nomikos-Device-Token"  # noqa: S105 - a header name, not a token
AGENT_VERSION_HEADER = "X-Nomikos-Agent-Version"
"""Which build of the agent is calling. Checked by the **version floor** on the
**claim** path and on `GET /device/v1/agent/version`
(`backend/ml/api/agent_version.py`); defined here so the run loop and the
launch check report the same version."""
SERVICE_TOKEN_HEADER = "X-Nomikos-Service-Token"  # noqa: S105 - a header name, not a token
"""A hosted worker's **service credential**, in its own header because it
resolves to a different scope than the device token (ADR 0005, decision 1): a
device token claims `local` work on one account, this claims `cloud` work for
the platform. The two must never be interchangeable by accident."""

WORKER_NAME_HEADER = "X-Nomikos-Worker-Name"
"""Which hosted worker is calling. Names its device row; not a secret."""

CLAIM_PATH = "/device/v1/jobs/claim"
CALLBACK_PATH = "/internal/inference/job-complete"


class PlatformClient:
    """Every request the CLI makes: pair a machine, ask the floor, then claim,
    fetch, report."""

    def __init__(self, base_url: str, *, timeout: float = REQUEST_TIMEOUT_SECONDS) -> None:
        self.base_url = require_secure_platform_url(base_url)
        self.plaintext_loopback = urlsplit(self.base_url).scheme == "http"
        """This client is pointed at a platform on this machine. The only state
        in which it will fetch a page image over cleartext HTTP."""
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------
    def _request(
        self,
        method: str,
        path: str,
        *,
        body: dict | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> tuple[int, object]:
        """Send one request; return `(status, decoded body)`.

        Non-2xx statuses are returned rather than raised, since on this surface
        they are answers: a 401 from `/device/v1/self` is how a revoked device
        finds out. Only a failure to reach the platform at all is an exception.

        `timeout` overrides the client's default for the one call that needs
        it: a long poll is *expected* to hold the socket open, so the deadline
        that suits a JSON round trip would abort it every time.
        """
        deadline = self.timeout if timeout is None else timeout
        url = f"{self.base_url}{path}"
        payload = None if body is None else json.dumps(body).encode("utf-8")
        # S310 (here and on the `Request` below): `url` is `base_url` + a
        # literal path, and `base_url` passed `require_secure_platform_url`
        # in `__init__` - https, or http to loopback for the integration suite.
        request = urllib.request.Request(url, data=payload, method=method)  # noqa: S310
        request.add_header("Accept", "application/json")
        # Recorded on `helper_pairings.user_agent` for support correlation.
        request.add_header("User-Agent", f"nomikos-inference/{installed_version()}")
        if payload is not None:
            request.add_header("Content-Type", "application/json")
        for name, value in (headers or {}).items():
            request.add_header(name, value)

        try:
            with open_request(request, deadline) as response:
                return response.status, _decode(
                    _read_bounded(response, JSON_RESPONSE_LIMIT, what="the response")
                )
        except urllib.error.HTTPError as exc:
            return exc.code, _decode(_read_bounded(exc, JSON_RESPONSE_LIMIT, what="the response"))
        except urllib.error.URLError as exc:
            raise PlatformError(f"Cannot reach {self.base_url}: {exc.reason}") from exc
        except TimeoutError as exc:
            raise PlatformError(f"{self.base_url} did not answer within {deadline:g}s") from exc
        except OSError as exc:  # pragma: no cover - socket-level failures
            raise PlatformError(f"Cannot reach {self.base_url}: {exc}") from exc

    # ------------------------------------------------------------------
    # Protocol
    # ------------------------------------------------------------------
    def start_pairing(
        self,
        *,
        device_name: str,
        device_platform: str,
        agent_version: str,
        capabilities: dict | None = None,
    ) -> StartedPairing:
        status, body = self._request(
            "POST",
            "/device/v1/pairings",
            body={
                "device_name": device_name,
                "platform": device_platform,
                "helper_version": agent_version[:VERSION_LIMIT],
                "capabilities": capabilities or {},
            },
        )
        if status == 404:
            raise PlatformError(
                f"{self.base_url} is not serving device pairing. "
                "It is disabled by default in production until the consent page ships."
            )
        if status == 429:
            raise PlatformError(
                f"{self.base_url} is refusing new pairing requests right now. Try again shortly."
            )
        if status != 201 or not isinstance(body, dict):
            raise PlatformError(_unexpected(status, body, "starting the pairing"))
        try:
            return StartedPairing(
                pairing_id=body["pairing_id"],
                device_code=body["device_code"],
                verification_url=body["verification_url"],
                confirmation_code=body["confirmation_code"],
                expires_in=int(body["expires_in"]),
                interval_seconds=int(body["interval_seconds"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PlatformError(f"{self.base_url} returned an unusable pairing response") from exc

    def collect_token(self, *, pairing_id: str, device_code: str) -> PairingPoll:
        status, body = self._request(
            "POST",
            "/device/v1/pairings/token",
            body={"pairing_id": pairing_id, "device_code": device_code},
        )
        if status != 200 or not isinstance(body, dict):
            raise PlatformError(_unexpected(status, body, "waiting for approval"))
        try:
            return PairingPoll(
                status=str(body.get("status") or ""),
                interval_seconds=int(body.get("interval_seconds") or 0),
                device_id=body.get("device_id"),
                device_token=body.get("device_token"),
                token_expires_at=_parse_datetime(body.get("token_expires_at")),
                account_email=body.get("account_email"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PlatformError(f"{self.base_url} returned an unusable pairing response") from exc

    def read_self(self, *, device_token: str) -> DeviceIdentity | None:
        """Confirm a stored credential. `None` means the platform refused it.

        The platform answers every rejection (unknown, expired, revoked) with
        the same 401 and the same public message, so this can't say *why*. The
        caller pairs it with the stored expiry, the one thing it knows
        locally, to name the likely cause.
        """
        status, body = self._request(
            "GET", "/device/v1/self", headers={DEVICE_TOKEN_HEADER: device_token}
        )
        if status == 401:
            return None
        if status == 404:
            raise PlatformError(
                f"{self.base_url} is not serving the device layer, so this "
                "machine's credential cannot be checked."
            )
        if status != 200 or not isinstance(body, dict):
            raise PlatformError(_unexpected(status, body, "checking this machine's credential"))
        return DeviceIdentity(
            device_id=str(body.get("device_id") or ""),
            name=str(body.get("name") or ""),
            account_email=str(body.get("account_email") or ""),
            token_expires_at=_parse_datetime(body.get("token_expires_at")),
        )

    # ------------------------------------------------------------------
    # The claim loop
    # ------------------------------------------------------------------
    def claim_page(self, *, credential: dict[str, str], wait_seconds: int) -> Claim:
        """Take at most one page of work, or come back empty.

        The credential decides the **execution target** and the caller can't
        ask for a different one, so there's nothing to send but how long this
        agent is willing to wait. `wait_seconds` is clamped by the platform,
        not here: the ceiling is an operational dial that belongs to whoever
        is running it.

        The socket deadline is the *requested* wait plus the ordinary request
        budget, not the budget alone. A long poll that asked the platform to
        hold the connection for its own ceiling and then hung up first would
        time out every claim it made, and the flag that asked for it would
        read as broken rather than unsupported.
        """
        wait = max(0, int(wait_seconds))
        status, body = self._request(
            "POST",
            CLAIM_PATH,
            body={"wait_seconds": wait},
            headers=credential,
            timeout=wait + self.timeout,
        )
        if status == AGENT_VERSION_REFUSED_STATUS:
            raise _version_refusal(body, self.base_url)
        if status == 401:
            raise CredentialRefused(
                f"{self.base_url} does not accept this machine's credential. "
                "Run `nomikos pair` to authorise it again."
            )
        if status == 404:
            raise PlatformError(
                f"{self.base_url} is not serving the claim endpoint. "
                "The device layer is disabled by default in production."
            )
        if status != 200 or not isinstance(body, dict):
            raise PlatformError(_unexpected(status, body, "claiming a page"))

        try:
            return Claim(
                page=_claimed_page(body.get("page")),
                poll_after_seconds=float(body.get("poll_after_seconds") or 0.0),
                agent=_agent_notice(body.get("agent")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PlatformError(f"{self.base_url} returned an unusable claim response") from exc

    def fetch_page_image(self, url: str) -> bytes:
        """Download the one page image a claim points at.

        **No credential is attached, deliberately.** The signature in the URL
        is the authorization (ADR 0002); sending the device token here would
        imply the link needed it, and the link reaches exactly one object
        either way. That's also why this bypasses `_request`: that method adds
        headers this request has no business carrying, and the answer is
        bytes rather than JSON.

        The URL is server-controlled end to end, so its scheme is checked
        before it reaches the opener. `urlopen` is not an HTTP client: the
        default opener also services `file:`, which would turn "here is your
        page image" into a read of any path the researcher's account can open.
        """
        require_fetchable(url, plaintext_loopback=self.plaintext_loopback)
        # S310: `require_fetchable` above rejects every scheme but https, which
        # is what stops a server-chosen `file:` URL from reading this machine.
        # Redirects are refused by the opener, so the checked scheme is the one
        # that is fetched: a signed https link cannot downgrade to http mid-hop.
        request = urllib.request.Request(url, method="GET")  # noqa: S310
        request.add_header("User-Agent", f"nomikos-inference/{installed_version()}")
        try:
            with open_request(request, IMAGE_TIMEOUT_SECONDS) as response:
                if response.status != 200:
                    raise PlatformError(f"The page image could not be fetched ({response.status})")
                return _read_bounded(response, max_page_image_bytes())
        except urllib.error.HTTPError as exc:
            if exc.code == 403:
                # Forged, malformed, and expired are one status on purpose, so
                # the route is not an oracle for which object keys exist. Only
                # one of the three can plausibly happen to an honest agent.
                raise PlatformError(
                    "The link to this page's image was refused. It expires about a "
                    "minute after the claim, so it was most likely used too late."
                ) from exc
            raise PlatformError(f"The page image could not be fetched ({exc.code})") from exc
        except urllib.error.URLError as exc:
            raise PlatformError(f"The page image could not be fetched: {exc.reason}") from exc
        except TimeoutError as exc:
            raise PlatformError("The page image did not arrive in time") from exc
        except OSError as exc:  # pragma: no cover - socket-level failures
            raise PlatformError(f"The page image could not be fetched: {exc}") from exc

    def report_page(
        self,
        *,
        credential: dict[str, str],
        page: ClaimedPage,
        output: dict | None = None,
        error: str | None = None,
    ) -> None:
        """End the page, one way or the other.

        Not a new endpoint: this is the platform's existing
        `JobCallbackRequest` (ADR 0003), authorised by the same credential the
        page was claimed with and narrowed to the page this agent is actually
        holding. A researcher's laptop has no webhook secret and must not be
        given one.
        """
        payload: dict[str, object] = {
            "inference_job_id": page.inference_job_id,
            "product_job_id": page.product_job_id,
            "task": page.task,
            "status": "done" if error is None else "failed",
        }
        if error is None:
            payload["output"] = output
        else:
            payload["error"] = error

        status, body = self._request("POST", CALLBACK_PATH, body=payload, headers=credential)
        if status == 204:
            return
        if status == 403:
            expired = (
                " Its **lease** had already run out on this machine's clock."
                if (page.lease_expired())
                else ""
            )
            raise PageLeaseLost(
                "The platform says this machine is not holding that page. Its "
                "**lease** most likely expired and the page went back to the "
                f"queue.{expired}"
            )
        raise PlatformError(_unexpected(status, body, "reporting a page"))

    def read_agent_floor(self, *, agent_version: str) -> AgentFloor:
        """Ask what this version is allowed to do, without asking for work.

        Runs the same comparison the **claim** path runs, on an endpoint that
        touches no queue. That separation is the point: an agent learns it's
        below the floor at launch, when nothing is in flight and replacing its
        own code is safe, rather than while holding a page it's already been
        handed.

        No credential is sent or needed: the platform resolves the version
        before it looks at one. So this also answers for a machine that has
        never paired, which is exactly the machine most likely to be stale.
        """
        status, body = self._request(
            "GET", "/device/v1/agent/version", headers={AGENT_VERSION_HEADER: agent_version}
        )
        if status == 404:
            raise PlatformError(
                f"{self.base_url} is not serving the device layer, so it cannot "
                "say which agent version it requires."
            )
        if status == AGENT_VERSION_REFUSED_STATUS:
            return _refusal_floor(self.base_url, agent_version, body)
        if status != 200 or not isinstance(body, dict):
            raise PlatformError(_unexpected(status, body, "asking for the version floor"))
        try:
            return AgentFloor(
                agent_version=str(body["agent_version"]),
                minimum_version=str(body["minimum_version"]),
                latest_version=str(body["latest_version"]),
                package=str(body["package"]),
                refused=False,
                outdated=bool(body["outdated"]),
                message="",
                upgrade_command=str(body.get("upgrade_command") or ""),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PlatformError(
                f"{self.base_url} returned an unusable version-floor response"
            ) from exc
