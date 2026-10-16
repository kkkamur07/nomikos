"""The platform client sends credentials over TLS, never follows redirects, and
bounds what it reads.

Three properties follow from the threat model. The transport refuses every
redirect instead of following it: the stdlib's redirect handler copies bearer
headers to whatever `Location` names, so one 302 would deliver the device or
service token to another host. JSON answers are bounded at sixteen megabytes,
page images stream through their own limit. And a page image link that answers
without a 200 is refused rather than handed to the model as bytes.

These are unit tests over a stub opener: no sockets except the two loopback
redirect tests, which pin the actual no-redirect wiring end to end.
"""

from __future__ import annotations

import io
import json
import threading
import urllib.error
from datetime import UTC, datetime, timedelta
from email.message import Message
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from nomikos_inference.cli import transport as transport_module
from nomikos_inference.cli.api import (
    AGENT_VERSION_REFUSED_STATUS,
    CLAIM_PATH,
    AgentFloor,
    AgentVersionRefused,
    Claim,
    ClaimedPage,
    CredentialRefused,
    DeviceIdentity,
    PageLeaseLost,
    PlatformClient,
    PlatformError,
)
from nomikos_inference.cli.api import _agent_notice as _agent_notice
from nomikos_inference.cli.api import _claimed_page as _claimed_page
from nomikos_inference.cli.api import _decode as _decode
from nomikos_inference.cli.api import _unexpected as _unexpected
from nomikos_inference.cli.api import _version_refusal as _version_refusal
from nomikos_inference.cli.transport import _read_bounded as _read_bounded

BASE_URL = "https://platform.example.com"


class FakeResponse:
    """One stubbed HTTP answer: a status, a body, and headers."""

    def __init__(self, *, status: int = 200, body: bytes = b"{}", length: str | None = None):
        self.status = status
        self._body = io.BytesIO(body)
        self.headers = Message()
        if length is not None:
            self.headers["Content-Length"] = length

    def read(self, amt: int = -1) -> bytes:
        return self._body.read(amt)

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None


class FakeOpener:
    """A script of answers (or failures) that records what it was asked."""

    def __init__(self, script: list[object]):
        self._script = list(script)
        self.requests: list[object] = []
        self.timeouts: list[float] = []

    def open(self, request: object, timeout: float | None = None) -> FakeResponse:
        self.requests.append(request)
        self.timeouts.append(timeout if timeout is not None else -1.0)
        answer = self._script.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        assert isinstance(answer, FakeResponse)
        return answer


@pytest.fixture
def opener(monkeypatch: pytest.MonkeyPatch) -> FakeOpener:
    stub = FakeOpener([])
    monkeypatch.setattr(transport_module, "_OPENER", stub)
    return stub


def _json_bytes(payload: object) -> bytes:
    return json.dumps(payload).encode("utf-8")


def _http_error(code: int, payload: object) -> urllib.error.HTTPError:
    import urllib.error

    return urllib.error.HTTPError(
        "https://platform.example.com/x",
        code,
        "reason",
        Message(),
        io.BytesIO(_json_bytes(payload)),
    )


def _client() -> PlatformClient:
    return PlatformClient(BASE_URL)


def _page(*, lease_expires_at: datetime | None = None) -> ClaimedPage:
    return ClaimedPage(
        product_job_id="11111111-2222-3333-4444-555555555555",
        inference_job_id="66666666-7777-8888-9999-000000000000",
        lease_expires_at=lease_expires_at,
        task="segment",
        registry_model_id="blla-segment",
        registry_tag="stable",
        params={},
        page_image_url="https://storage.example.com/signed/page.png",
        page_image_expires_at=None,
    )


def _client() -> PlatformClient:
    return PlatformClient(BASE_URL)


def test_a_200_json_body_decodes(opener: FakeOpener) -> None:
    opener._script.append(FakeResponse(body=_json_bytes({"status": "authorization_pending"})))

    status, body = _client()._request("POST", "/device/v1/pairings/token")

    assert (status, body) == (200, {"status": "authorization_pending"})


def test_an_empty_body_decodes_to_none(opener: FakeOpener) -> None:
    opener._script.append(FakeResponse(body=b""))

    assert _client()._request("GET", "/device/v1/self") == (200, None)


def test_an_http_error_returns_its_status_and_body(opener: FakeOpener) -> None:
    opener._script.append(_http_error(401, {"error": {"message": "no"}}))

    status, body = _client()._request("GET", "/device/v1/self")

    assert status == 401
    assert body == {"error": {"message": "no"}}


def test_a_garbage_error_body_decodes_to_none(opener: FakeOpener) -> None:
    import urllib.error

    opener._script.append(
        urllib.error.HTTPError("https://x", 502, "bad gateway", Message(), io.BytesIO(b"<html"))
    )

    assert _client()._request("POST", CLAIM_PATH) == (502, None)


def test_an_unreachable_platform_is_a_platform_error(opener: FakeOpener) -> None:
    import urllib.error

    opener._script.append(urllib.error.URLError("connection refused"))

    with pytest.raises(PlatformError, match="Cannot reach"):
        _client()._request("GET", "/device/v1/self")


def test_a_silent_platform_is_a_platform_error(opener: FakeOpener) -> None:
    opener._script.append(TimeoutError("timed out"))

    with pytest.raises(PlatformError, match="did not answer within"):
        _client()._request("GET", "/device/v1/self")


def test_json_requests_carry_json_headers(opener: FakeOpener) -> None:
    opener._script.append(FakeResponse(body=b"{}"))
    client = _client()

    client._request("POST", "/device/v1/pairings/token", body={"a": 1})

    sent = opener.requests[0]
    assert sent.get_header("Content-type") == "application/json"
    assert sent.get_header("Accept") == "application/json"
    assert sent.get_header("User-agent").startswith("nomikos-inference/")


def test_error_bodies_are_bounded_too(opener: FakeOpener, monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.error

    import nomikos_inference.cli.api as api_module

    monkeypatch.setattr(api_module, "JSON_RESPONSE_LIMIT", 10)
    opener._script.append(
        urllib.error.HTTPError("https://x", 500, "boom", Message(), io.BytesIO(b"x" * 20))
    )

    with pytest.raises(PlatformError, match="the response exceeded the size limit"):
        _client()._request("GET", "/device/v1/self")


# ---------------------------------------------------------------------------
def test_a_json_body_past_the_bound_is_refused(
    opener: FakeOpener, monkeypatch: pytest.MonkeyPatch
) -> None:
    import nomikos_inference.cli.api as api_module

    monkeypatch.setattr(api_module, "JSON_RESPONSE_LIMIT", 10)
    opener._script.append(FakeResponse(body=b"x" * 20))

    with pytest.raises(PlatformError, match="the response exceeded the size limit"):
        _client()._request("GET", "/device/v1/self")


def test_a_lying_content_length_does_not_skip_the_bound(
    opener: FakeOpener, monkeypatch: pytest.MonkeyPatch
) -> None:
    import nomikos_inference.cli.api as api_module

    monkeypatch.setattr(api_module, "JSON_RESPONSE_LIMIT", 10)
    opener._script.append(FakeResponse(body=b"x" * 20, length="5"))

    with pytest.raises(PlatformError, match="the response exceeded the size limit"):
        _client()._request("GET", "/device/v1/self")


def test_an_announced_over_limit_body_is_refused_without_reading(
    opener: FakeOpener, monkeypatch: pytest.MonkeyPatch
) -> None:
    import nomikos_inference.cli.api as api_module

    monkeypatch.setattr(api_module, "JSON_RESPONSE_LIMIT", 10)
    opener._script.append(FakeResponse(body=b"x" * 20, length="20"))

    with pytest.raises(PlatformError, match="the response exceeded the size limit"):
        _client()._request("GET", "/device/v1/self")


def _dense_lines(n: int = 300, points: int = 256) -> list[dict[str, object]]:
    """A dense transcribe claim: every line carries a full polygon.

    Mirrors what the dispatcher puts in ``params.lines`` (line id, index, and
    up to 256 points), with pixel-like coordinates so the JSON size is honest.
    """
    return [
        {
            "line_id": f"00000000-0000-4000-8000-{i:012d}",
            "line_index": i,
            "points": [[100.5 + k * 0.75, 200.25 + k * 1.5] for k in range(points)],
        }
        for i in range(n)
    ]


def test_a_dense_transcribe_claim_is_accepted(opener: FakeOpener) -> None:
    from nomikos_inference.cli.transport import JSON_RESPONSE_LIMIT

    answer = _claim_answer()
    assert isinstance(answer["page"], dict)
    page = answer["page"]
    assert isinstance(page["request"], dict)
    page["request"]["params"] = {"lines": _dense_lines()}
    raw = _json_bytes(answer)
    # This exact body was refused by the old 1 MiB bound: the fix must take it.
    assert len(raw) > 1 << 20
    assert len(raw) < JSON_RESPONSE_LIMIT
    opener._script.append(FakeResponse(body=raw))

    claim = _client().claim_page(credential={"X": "y"}, wait_seconds=0)

    assert claim.page is not None
    assert len(claim.page.params["lines"]) == 300


def test_a_body_one_byte_over_the_bound_is_refused(opener: FakeOpener) -> None:
    from nomikos_inference.cli.transport import JSON_RESPONSE_LIMIT

    opener._script.append(FakeResponse(body=b"x" * (JSON_RESPONSE_LIMIT + 1)))

    with pytest.raises(PlatformError, match="the response exceeded the size limit"):
        _client()._request("GET", "/device/v1/self")


def test_an_announced_over_bound_body_is_refused_without_reading(
    opener: FakeOpener,
) -> None:
    from nomikos_inference.cli.transport import JSON_RESPONSE_LIMIT

    opener._script.append(FakeResponse(body=b"x", length=str(JSON_RESPONSE_LIMIT + 1)))

    with pytest.raises(PlatformError, match="the response exceeded the size limit"):
        _client()._request("GET", "/device/v1/self")


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------
def _pairing_answer() -> dict[str, object]:
    return {
        "pairing_id": "pair-1",
        "device_code": "code-1",
        "verification_url": "https://platform.example.com/pair",
        "confirmation_code": "ABCD-1234",
        "expires_in": 300,
        "interval_seconds": 5,
    }


def test_start_pairing_reads_the_answer(opener: FakeOpener) -> None:
    opener._script.append(FakeResponse(status=201, body=_json_bytes(_pairing_answer())))

    started = _client().start_pairing(
        device_name="laptop", device_platform="darwin-arm64", agent_version="0.4.0"
    )

    assert started.pairing_id == "pair-1"
    assert started.confirmation_code == "ABCD-1234"


def test_start_pairing_names_a_missing_endpoint(opener: FakeOpener) -> None:
    opener._script.append(FakeResponse(status=404, body=b"{}"))

    with pytest.raises(PlatformError, match="not serving device pairing"):
        _client().start_pairing(
            device_name="laptop", device_platform="darwin-arm64", agent_version="0.4.0"
        )


def test_start_pairing_names_rate_limiting(opener: FakeOpener) -> None:
    opener._script.append(FakeResponse(status=429, body=b"{}"))

    with pytest.raises(PlatformError, match="refusing new pairing requests"):
        _client().start_pairing(
            device_name="laptop", device_platform="darwin-arm64", agent_version="0.4.0"
        )


def test_start_pairing_rejects_an_unusable_answer(opener: FakeOpener) -> None:
    opener._script.append(FakeResponse(status=201, body=_json_bytes({"pairing_id": "only"})))

    with pytest.raises(PlatformError, match="unusable pairing response"):
        _client().start_pairing(
            device_name="laptop", device_platform="darwin-arm64", agent_version="0.4.0"
        )


def test_start_pairing_rejects_a_non_dict_answer(opener: FakeOpener) -> None:
    opener._script.append(FakeResponse(status=201, body=b"[1, 2]"))

    with pytest.raises(PlatformError, match="starting the pairing"):
        _client().start_pairing(
            device_name="laptop", device_platform="darwin-arm64", agent_version="0.4.0"
        )


def test_collect_token_reads_the_poll(opener: FakeOpener) -> None:
    opener._script.append(
        FakeResponse(body=_json_bytes({"status": "approved", "interval_seconds": 5}))
    )

    poll = _client().collect_token(pairing_id="pair-1", device_code="code-1")

    assert poll.status == "approved"


def test_collect_token_rejects_a_failure(opener: FakeOpener) -> None:
    opener._script.append(FakeResponse(status=500, body=b"{}"))

    with pytest.raises(PlatformError, match="waiting for approval"):
        _client().collect_token(pairing_id="pair-1", device_code="code-1")


def test_collect_token_rejects_an_unusable_poll(opener: FakeOpener) -> None:
    opener._script.append(
        FakeResponse(body=_json_bytes({"status": "approved", "interval_seconds": "soon"}))
    )

    with pytest.raises(PlatformError, match="unusable pairing response"):
        _client().collect_token(pairing_id="pair-1", device_code="code-1")


# ---------------------------------------------------------------------------
# read_self
# ---------------------------------------------------------------------------
def test_read_self_confirms_the_credential(opener: FakeOpener) -> None:
    opener._script.append(
        FakeResponse(
            body=_json_bytes(
                {"device_id": "d-1", "name": "laptop", "account_email": "r@example.com"}
            )
        )
    )

    identity = _client().read_self(device_token="secret")

    assert isinstance(identity, DeviceIdentity)
    assert identity.device_id == "d-1"


def test_read_self_returns_none_on_401(opener: FakeOpener) -> None:
    opener._script.append(FakeResponse(status=401, body=b"{}"))

    assert _client().read_self(device_token="secret") is None


def test_read_self_names_a_missing_device_layer(opener: FakeOpener) -> None:
    opener._script.append(FakeResponse(status=404, body=b"{}"))

    with pytest.raises(PlatformError, match="not serving the device layer"):
        _client().read_self(device_token="secret")


def test_read_self_rejects_a_broken_answer(opener: FakeOpener) -> None:
    opener._script.append(FakeResponse(status=500, body=b"{}"))

    with pytest.raises(PlatformError, match="checking this machine's credential"):
        _client().read_self(device_token="secret")


# ---------------------------------------------------------------------------
# claim_page
# ---------------------------------------------------------------------------
def _claim_answer() -> dict[str, object]:
    return {
        "page": {
            "product_job_id": "11111111-2222-3333-4444-555555555555",
            "inference_job_id": "66666666-7777-8888-9999-000000000000",
            "lease_expires_at": None,
            "request": {
                "task": "segment",
                "registry_model_id": "blla-segment",
                "registry_tag": "stable",
                "params": {},
            },
            "page_image_url": "https://storage.example.com/signed/page.png",
            "page_image_expires_at": None,
        },
        "poll_after_seconds": 1.5,
        "agent": {
            "agent_version": "0.4.0",
            "latest_version": "0.4.0",
            "outdated": False,
            "upgrade_command": "",
        },
    }


def test_claim_page_hands_over_work(opener: FakeOpener) -> None:
    opener._script.append(FakeResponse(body=_json_bytes(_claim_answer())))

    claim = _client().claim_page(credential={"X": "y"}, wait_seconds=25)

    assert isinstance(claim, Claim)
    assert claim.page is not None
    assert claim.page.registry_model_id == "blla-segment"
    assert claim.poll_after_seconds == 1.5


def test_claim_page_waits_the_requested_wait_plus_budget(opener: FakeOpener) -> None:
    opener._script.append(FakeResponse(body=_json_bytes({"poll_after_seconds": 0})))

    _client().claim_page(credential={"X": "y"}, wait_seconds=10)

    assert opener.timeouts[0] == pytest.approx(40.0)


def test_claim_page_raises_the_floor_refusal(opener: FakeOpener) -> None:
    opener._script.append(
        FakeResponse(
            status=AGENT_VERSION_REFUSED_STATUS,
            body=_json_bytes(
                {
                    "error": {
                        "code": "AGENT_VERSION_UNSUPPORTED",
                        "message": "too old",
                        "agent_version": "0.1.0",
                        "minimum_version": "0.4.0",
                        "latest_version": "0.5.0",
                        "package": "nomikos-inference",
                        "upgrade_command": "uv tool upgrade nomikos-inference",
                    }
                }
            ),
        )
    )

    with pytest.raises(AgentVersionRefused, match="too old") as exc_info:
        _client().claim_page(credential={"X": "y"}, wait_seconds=0)

    assert exc_info.value.minimum_version == "0.4.0"


def test_claim_page_refuses_a_dead_credential(opener: FakeOpener) -> None:
    opener._script.append(FakeResponse(status=401, body=b"{}"))

    with pytest.raises(CredentialRefused, match="nomikos pair"):
        _client().claim_page(credential={"X": "y"}, wait_seconds=0)


def test_claim_page_names_a_missing_claim_endpoint(opener: FakeOpener) -> None:
    opener._script.append(FakeResponse(status=404, body=b"{}"))

    with pytest.raises(PlatformError, match="not serving the claim endpoint"):
        _client().claim_page(credential={"X": "y"}, wait_seconds=0)


def test_claim_page_reports_a_broken_claim(opener: FakeOpener) -> None:
    opener._script.append(FakeResponse(status=500, body=b"{}"))

    with pytest.raises(PlatformError, match="claiming a page"):
        _client().claim_page(credential={"X": "y"}, wait_seconds=0)


def test_unparsable_datetimes_read_as_missing() -> None:
    from nomikos_inference.cli.api import _parse_datetime

    assert _parse_datetime("not-a-date") is None
    assert _parse_datetime("") is None
    assert _parse_datetime(None) is None
    assert _parse_datetime("2026-05-01T12:00:00+00:00") is not None


def test_claim_page_rejects_an_unusable_answer(opener: FakeOpener) -> None:
    answer = _claim_answer()
    page = dict(answer["page"]) if isinstance(answer["page"], dict) else {}
    page.pop("request", None)
    answer["page"] = page
    opener._script.append(FakeResponse(body=_json_bytes(answer)))

    with pytest.raises(PlatformError, match="unusable claim response"):
        _client().claim_page(credential={"X": "y"}, wait_seconds=0)


def test_an_empty_queue_is_a_claim_without_a_page(opener: FakeOpener) -> None:
    opener._script.append(FakeResponse(body=_json_bytes({"poll_after_seconds": 2.0})))

    claim = _client().claim_page(credential={"X": "y"}, wait_seconds=0)

    assert claim.page is None
    assert claim.agent is None


# ---------------------------------------------------------------------------
# read_agent_floor
# ---------------------------------------------------------------------------
def _floor_answer() -> dict[str, object]:
    return {
        "agent_version": "0.4.0",
        "minimum_version": "0.4.0",
        "latest_version": "0.5.0",
        "package": "nomikos-inference",
        "outdated": True,
    }


def test_read_agent_floor_reads_the_floor(opener: FakeOpener) -> None:
    opener._script.append(FakeResponse(body=_json_bytes(_floor_answer())))

    floor = _client().read_agent_floor(agent_version="0.4.0")

    assert isinstance(floor, AgentFloor)
    assert (floor.refused, floor.outdated) == (False, True)


def test_read_agent_floor_reads_a_refusal(opener: FakeOpener) -> None:
    opener._script.append(
        FakeResponse(
            status=AGENT_VERSION_REFUSED_STATUS,
            body=_json_bytes(
                {
                    "error": {
                        "code": "AGENT_VERSION_UNSUPPORTED",
                        "message": "too old",
                        "minimum_version": "0.4.0",
                        "latest_version": "0.5.0",
                        "package": "nomikos-inference",
                    }
                }
            ),
        )
    )

    floor = _client().read_agent_floor(agent_version="0.1.0")

    assert (floor.refused, floor.minimum_version) == (True, "0.4.0")


def test_read_agent_floor_rejects_a_refusal_without_the_contract(opener: FakeOpener) -> None:
    opener._script.append(
        FakeResponse(status=AGENT_VERSION_REFUSED_STATUS, body=_json_bytes({"error": {}}))
    )

    with pytest.raises(PlatformError, match="did not say what it requires"):
        _client().read_agent_floor(agent_version="0.1.0")


def test_read_agent_floor_rejects_a_refusal_with_a_wrong_code(opener: FakeOpener) -> None:
    opener._script.append(
        FakeResponse(
            status=AGENT_VERSION_REFUSED_STATUS,
            body=_json_bytes({"error": {"code": "OTHER"}}),
        )
    )

    with pytest.raises(PlatformError, match="did not say what it requires"):
        _client().read_agent_floor(agent_version="0.1.0")


def test_read_agent_floor_rejects_an_incomplete_refusal(opener: FakeOpener) -> None:
    opener._script.append(
        FakeResponse(
            status=AGENT_VERSION_REFUSED_STATUS,
            body=_json_bytes({"error": {"code": "AGENT_VERSION_UNSUPPORTED"}}),
        )
    )

    with pytest.raises(PlatformError, match="did not say what it requires"):
        _client().read_agent_floor(agent_version="0.1.0")


def test_read_agent_floor_names_a_missing_device_layer(opener: FakeOpener) -> None:
    opener._script.append(FakeResponse(status=404, body=b"{}"))

    with pytest.raises(PlatformError, match="cannot say which agent version"):
        _client().read_agent_floor(agent_version="0.4.0")


def test_read_agent_floor_rejects_a_broken_answer(opener: FakeOpener) -> None:
    opener._script.append(FakeResponse(status=200, body=b"[]"))

    with pytest.raises(PlatformError, match="asking for the version floor"):
        _client().read_agent_floor(agent_version="0.4.0")


def test_read_agent_floor_rejects_an_incomplete_floor(opener: FakeOpener) -> None:
    opener._script.append(FakeResponse(status=200, body=_json_bytes({"agent_version": "x"})))

    with pytest.raises(PlatformError, match="unusable version-floor response"):
        _client().read_agent_floor(agent_version="0.4.0")


# ---------------------------------------------------------------------------
# report_page
# ---------------------------------------------------------------------------
def test_report_page_accepts_a_204(opener: FakeOpener) -> None:
    opener._script.append(FakeResponse(status=204, body=b""))

    _client().report_page(credential={"X": "y"}, page=_page(), output={"kind": "segment"})


def test_report_page_names_a_lost_lease(opener: FakeOpener) -> None:
    opener._script.append(FakeResponse(status=403, body=b"{}"))
    page = _page(lease_expires_at=datetime.now(UTC) - timedelta(seconds=1))

    with pytest.raises(PageLeaseLost, match="lease.*had already run out"):
        _client().report_page(credential={"X": "y"}, page=page, error="broken")


def test_report_page_without_an_expired_lease_says_less(opener: FakeOpener) -> None:
    opener._script.append(FakeResponse(status=403, body=b"{}"))

    with pytest.raises(PageLeaseLost, match="not holding that page"):
        _client().report_page(credential={"X": "y"}, page=_page(), error="broken")


def test_report_page_rejects_a_broken_answer(opener: FakeOpener) -> None:
    opener._script.append(FakeResponse(status=500, body=b"{}"))

    with pytest.raises(PlatformError, match="reporting a page"):
        _client().report_page(credential={"X": "y"}, page=_page(), output={})


# ---------------------------------------------------------------------------
# fetch_page_image
# ---------------------------------------------------------------------------
def test_fetch_page_image_returns_bytes(opener: FakeOpener) -> None:
    opener._script.append(FakeResponse(status=200, body=b"PNG-bytes"))

    assert _client().fetch_page_image("https://storage.example.com/signed/page.png") == b"PNG-bytes"


def test_fetch_page_image_refuses_an_expired_link(opener: FakeOpener) -> None:
    opener._script.append(_http_error(403, {}))

    with pytest.raises(PlatformError, match="used too late"):
        _client().fetch_page_image("https://storage.example.com/signed/page.png")


def test_fetch_page_image_names_other_failures(opener: FakeOpener) -> None:
    opener._script.append(_http_error(404, {}))

    with pytest.raises(PlatformError, match=r"could not be fetched \(404\)"):
        _client().fetch_page_image("https://storage.example.com/signed/page.png")


def test_fetch_page_image_refuses_a_non_200_without_an_exception(opener: FakeOpener) -> None:
    opener._script.append(FakeResponse(status=500, body=b"oops"))

    with pytest.raises(PlatformError, match=r"could not be fetched \(500\)"):
        _client().fetch_page_image("https://storage.example.com/signed/page.png")


def test_fetch_page_image_reports_an_unreachable_store(opener: FakeOpener) -> None:
    import urllib.error

    opener._script.append(urllib.error.URLError("timed out"))

    with pytest.raises(PlatformError, match="could not be fetched"):
        _client().fetch_page_image("https://storage.example.com/signed/page.png")


def test_fetch_page_image_reports_a_timeout(opener: FakeOpener) -> None:
    opener._script.append(TimeoutError("slow"))

    with pytest.raises(PlatformError, match="did not arrive in time"):
        _client().fetch_page_image("https://storage.example.com/signed/page.png")


def test_fetch_page_image_enforces_its_size_limit(
    opener: FakeOpener, monkeypatch: pytest.MonkeyPatch
) -> None:
    import nomikos_inference.cli.api as api_module

    monkeypatch.setattr(api_module, "max_page_image_bytes", lambda: 4)
    opener._script.append(FakeResponse(status=200, body=b"too many bytes"))

    with pytest.raises(PlatformError, match="exceeded the size limit"):
        _client().fetch_page_image("https://storage.example.com/signed/page.png")


@pytest.mark.parametrize(
    "url",
    [
        "http://storage.example.com/signed/page.png",
        "file:///etc/passwd",
        "ftp://storage.example.com/page.png",
        "not-a-url",
        "",
    ],
)
def test_fetch_page_image_never_opens_a_non_https_link(url: str, opener: FakeOpener) -> None:
    with pytest.raises(PlatformError, match="must be https"):
        _client().fetch_page_image(url)

    assert opener.requests == []


def test_a_loopback_client_may_fetch_loopback_http(opener: FakeOpener) -> None:
    client = PlatformClient("http://127.0.0.1:8901")
    opener._script.append(FakeResponse(status=200, body=b"bytes"))

    assert client.fetch_page_image("http://127.0.0.1:8901/page.png") == b"bytes"


def test_a_loopback_client_may_not_fetch_remote_http(opener: FakeOpener) -> None:
    client = PlatformClient("http://127.0.0.1:8901")

    with pytest.raises(PlatformError, match="must be https"):
        client.fetch_page_image("http://storage.example.com/page.png")


def test_an_https_client_may_not_fetch_loopback_http(opener: FakeOpener) -> None:
    with pytest.raises(PlatformError, match="must be https"):
        _client().fetch_page_image("http://127.0.0.1:8901/page.png")


# ---------------------------------------------------------------------------
# Redirects: the Required finding, pinned end to end on loopback
# ---------------------------------------------------------------------------
def _serve(handler: type[BaseHTTPRequestHandler]) -> HTTPServer:
    server = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_a_redirect_never_carries_the_device_token_to_another_origin() -> None:
    seen: dict[str, object] = {"hits": 0}

    class Elsewhere(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            seen["hits"] = seen["hits"] + 1
            seen["token"] = self.headers.get("X-Nomikos-Device-Token")
            self.send_response(401)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *args: object) -> None:
            pass

    other = _serve(Elsewhere)

    class Platform(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{other.server_port}/steal")
            self.end_headers()

        def log_message(self, *args: object) -> None:
            pass

    platform = _serve(Platform)
    client = PlatformClient(f"http://127.0.0.1:{platform.server_port}")

    with pytest.raises(PlatformError, match="Refusing to follow"):
        client.read_self(device_token="secret-device-token")

    assert seen["hits"] == 0
    assert "token" not in seen


def test_a_redirect_on_a_claim_is_refused_too() -> None:
    class Platform(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(length)
            self.send_response(302)
            self.send_header("Location", "https://platform.example.com/elsewhere")
            self.end_headers()

        def log_message(self, *args: object) -> None:
            pass

    platform = _serve(Platform)
    client = PlatformClient(f"http://127.0.0.1:{platform.server_port}")

    with pytest.raises(PlatformError, match="Refusing to follow"):
        client.claim_page(credential={"X-Nomikos-Device-Token": "s"}, wait_seconds=0)


def test_a_redirect_on_a_page_image_is_refused() -> None:
    class Store(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(302)
            self.send_header("Location", "http://127.0.0.1:1/downgrade")
            self.end_headers()

        def log_message(self, *args: object) -> None:
            pass

    store = _serve(Store)
    client = PlatformClient(f"http://127.0.0.1:{store.server_port}")

    with pytest.raises(PlatformError, match="Refusing to follow"):
        client.fetch_page_image(f"http://127.0.0.1:{store.server_port}/page.png")


# ---------------------------------------------------------------------------
# Readers and helpers
# ---------------------------------------------------------------------------
def test_decode_handles_its_inputs() -> None:
    assert _decode(b"") is None
    assert _decode(b"\xff\xfe") is None
    assert _decode(b"{oops") is None
    assert _decode(_json_bytes({"a": 1})) == {"a": 1}


def test_unexpected_mentions_the_server_message() -> None:
    assert _unexpected(500, {"error": {"message": "broken"}}, "claiming a page") == (
        "The platform answered 500 while claiming a page: broken"
    )
    assert _unexpected(500, {}, "claiming a page") == (
        "The platform answered 500 while claiming a page"
    )
    assert _unexpected(500, [1], "claiming a page") == (
        "The platform answered 500 while claiming a page"
    )


def test_version_refusal_falls_back_to_defaults() -> None:
    refusal = _version_refusal({}, BASE_URL)

    assert refusal.minimum_version == "unknown"
    assert refusal.upgrade_command == "uv tool upgrade nomikos-inference"


def test_agent_notice_ignores_non_dicts() -> None:
    assert _agent_notice(None) is None
    assert _agent_notice([1]) is None
    assert _agent_notice({"outdated": True}) is not None


def test_claimed_page_rejects_bad_shapes() -> None:
    assert _claimed_page(None) is None
    with pytest.raises(TypeError):
        _claimed_page([1])
    with pytest.raises(TypeError):
        _claimed_page({"request": [1]})
    # Missing keys surface as `KeyError`, which the claim call turns into an
    # "unusable claim response" rather than a crash.
    with pytest.raises(KeyError):
        _claimed_page({"request": {}})


def test_claimed_page_defaults_missing_optionals() -> None:
    page = _claimed_page(
        {
            "product_job_id": "p",
            "inference_job_id": "i",
            "request": {"task": "segment", "registry_model_id": "m", "params": [1]},
            "page_image_url": "https://x/y.png",
        }
    )

    assert page is not None
    assert page.registry_tag == "stable"
    assert page.params == {}


def test_read_bounded_edge_cases() -> None:
    assert _read_bounded(FakeResponse(body=b""), 10) == b""
    assert _read_bounded(FakeResponse(body=b"12345", length="5"), 5) == b"12345"
    # A garbage Content-Length is ignored; the stream still bounds the read.
    assert _read_bounded(FakeResponse(body=b"12345", length="garbage"), 5) == b"12345"
    with pytest.raises(PlatformError):
        _read_bounded(FakeResponse(body=b"123456", length="garbage"), 5)


def test_claim_page_sends_the_credential(opener: FakeOpener) -> None:
    opener._script.append(FakeResponse(body=_json_bytes({"poll_after_seconds": 0})))

    _client().claim_page(credential={"X-Nomikos-Device-Token": "secret"}, wait_seconds=0)

    sent = opener.requests[0]
    assert sent.get_header("X-nomikos-device-token") == "secret"


# ---------------------------------------------------------------------------
# URL policy and limits
# ---------------------------------------------------------------------------
def test_default_platform_url_prefers_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    from nomikos_inference.cli.transport import DEFAULT_PLATFORM_URL, default_platform_url

    monkeypatch.delenv("NOMIKOS_API_URL", raising=False)
    assert default_platform_url() == DEFAULT_PLATFORM_URL
    assert default_platform_url({}) == DEFAULT_PLATFORM_URL

    monkeypatch.setenv("NOMIKOS_API_URL", "https://staging.example.com/")
    assert default_platform_url() == "https://staging.example.com"
    assert default_platform_url({"NOMIKOS_API_URL": "https://x.example.com/"}) == (
        "https://x.example.com"
    )


def test_max_page_image_bytes_has_a_generous_default(monkeypatch: pytest.MonkeyPatch) -> None:
    from nomikos_inference.cli.transport import max_page_image_bytes

    monkeypatch.delenv("NOMIKOS_MAX_PAGE_IMAGE_BYTES", raising=False)
    assert max_page_image_bytes() == 1024 * 1024 * 1024

    monkeypatch.setenv("NOMIKOS_MAX_PAGE_IMAGE_BYTES", "1024")
    assert max_page_image_bytes() == 1024

    monkeypatch.setenv("NOMIKOS_MAX_PAGE_IMAGE_BYTES", "0")
    assert max_page_image_bytes() == 1

    monkeypatch.setenv("NOMIKOS_MAX_PAGE_IMAGE_BYTES", "plenty")
    with pytest.raises(ValueError, match="must be an integer"):
        max_page_image_bytes()


def test_machine_names_are_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    import socket

    from nomikos_inference.cli.transport import (
        DEVICE_NAME_LIMIT,
        PLATFORM_LIMIT,
        this_machine_name,
        this_machine_platform,
    )

    monkeypatch.setattr(socket, "gethostname", lambda: "n" * 500)
    assert len(this_machine_name()) == DEVICE_NAME_LIMIT

    assert this_machine_platform()
    assert len(this_machine_platform()) <= PLATFORM_LIMIT


def test_an_empty_hostname_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    import socket

    from nomikos_inference.cli.transport import this_machine_name

    monkeypatch.setattr(socket, "gethostname", lambda: "   ")

    assert this_machine_name() == "unnamed machine"
