"""Pairing works against a scripted platform, without a browser or a server.

`_pair` orchestrates five steps (name, start, announce, wait, store) and
`_wait_for_approval` polls on the platform's cadence, not a constant. Both are
tested here with a stub client and a recording console: the real browser and
the real platform live in `tests/inference/integration/test_cli_pairing.py`,
which needs Postgres.
"""

from __future__ import annotations

import argparse
import contextlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from nomikos_inference.cli import pair as pair_module
from nomikos_inference.cli.api import PlatformError, StartedPairing
from nomikos_inference.cli.credentials import CredentialError, DeviceCredential


class FakeConsole:
    """Records printed lines; Rich renderables are rendered, not `repr`d.

    Tables carry the facts under test (device ids, expiry dates), and
    `str(table)` would hide them behind `<rich.table.Table object>`.
    """

    def __init__(self) -> None:
        self.lines: list[str] = []

    def print(self, *args: object, **kwargs: object) -> None:
        self.lines.append(" ".join(self._render(a) for a in args))

    @staticmethod
    def _render(value: object) -> str:
        if not hasattr(value, "__rich_console__"):
            return str(value)
        import io

        from rich.console import Console

        buffer = io.StringIO()
        Console(file=buffer, width=200, highlight=False).print(value, end="")
        return buffer.getvalue()

    def status(self, message: object) -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()

    def text(self) -> str:
        return "\n".join(self.lines)


class FakeClient:
    """A platform that answers pairing calls from a script."""

    base_url = "https://platform.example.com"

    def __init__(
        self,
        *,
        pairing: StartedPairing | None = None,
        polls: list[object] | None = None,
        identity: object = "ok",
    ):
        self._pairing = pairing
        self._polls = list(polls or [])
        self._identity = identity

    def start_pairing(self, **kwargs: object) -> StartedPairing:
        assert self._pairing is not None
        return self._pairing

    def collect_token(self, *, pairing_id: str, device_code: str) -> object:
        answer = self._polls.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    def read_self(self, *, device_token: str) -> object:
        if isinstance(self._identity, BaseException):
            raise self._identity
        return self._identity


def _started(*, expires_in: int = 300, interval_seconds: int = 1) -> StartedPairing:
    return StartedPairing(
        pairing_id="pair-1",
        device_code="code-1",
        verification_url="https://platform.example.com/pair",
        confirmation_code="ABCD-1234",
        expires_in=expires_in,
        interval_seconds=interval_seconds,
    )


def _poll(
    status: str,
    *,
    interval_seconds: int = 1,
    device_id: str | None = None,
    device_token: str | None = None,
    account_email: str | None = None,
    token_expires_at: datetime | None = None,
) -> object:
    from nomikos_inference.cli.dto import PairingPoll

    return PairingPoll(
        status=status,
        interval_seconds=interval_seconds,
        device_id=device_id,
        device_token=device_token,
        token_expires_at=token_expires_at,
        account_email=account_email,
    )


def _credential(
    *,
    platform_url: str = "https://platform.example.com",
    token_expires_at: datetime | None = None,
) -> DeviceCredential:
    now = datetime.now(UTC)
    return DeviceCredential(
        platform_url=platform_url,
        device_id="device-1",
        device_token="secret-token",
        account_email="researcher@example.com",
        device_name="laptop",
        token_expires_at=(
            token_expires_at if token_expires_at is not None else now + timedelta(days=180)
        ),
        paired_at=now,
    )


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pair_module.time, "sleep", lambda _seconds: None)


# ---------------------------------------------------------------------------
# _pair
# ---------------------------------------------------------------------------
def test_pair_stores_the_approved_credential(
    monkeypatch: pytest.MonkeyPatch, no_sleep: None
) -> None:
    from nomikos_inference.cli.api import STATUS_APPROVED

    saved: list[DeviceCredential] = []
    monkeypatch.setattr(pair_module, "installed_version", lambda: "0.4.0")
    monkeypatch.setattr(pair_module, "this_machine_platform", lambda: "darwin-arm64")
    monkeypatch.setattr(
        pair_module, "save_credential", lambda credential: saved.append(credential) or Path("/x")
    )
    monkeypatch.setattr(pair_module, "file_mode", lambda path: 0o600)
    client = FakeClient(
        pairing=_started(),
        polls=[
            _poll(
                STATUS_APPROVED,
                device_id="d-9",
                device_token="tok-9",
                account_email="r@example.com",
                token_expires_at=datetime.now(UTC) + timedelta(days=180),
            )
        ],
    )
    console, errors = FakeConsole(), FakeConsole()
    args = argparse.Namespace(name=" field laptop ", no_browser=True)

    assert pair_module._pair(console, errors, client, args) == 0

    assert saved[0].device_id == "d-9"
    assert saved[0].device_token == "tok-9"
    assert saved[0].device_name == "field laptop"
    assert saved[0].token_expires_at is not None
    assert "Paired." in console.text()
    assert "Token expires" in console.text()


def test_pair_stops_when_approval_is_denied(
    monkeypatch: pytest.MonkeyPatch, no_sleep: None
) -> None:
    from nomikos_inference.cli.api import STATUS_ACCESS_DENIED

    monkeypatch.setattr(pair_module, "installed_version", lambda: "0.4.0")
    monkeypatch.setattr(pair_module, "this_machine_platform", lambda: "darwin-arm64")
    client = FakeClient(pairing=_started(), polls=[_poll(STATUS_ACCESS_DENIED)])
    console, errors = FakeConsole(), FakeConsole()
    args = argparse.Namespace(name=None, no_browser=True)

    assert pair_module._pair(console, errors, client, args) == 1


def test_an_ssh_session_does_not_open_a_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    console = FakeConsole()

    monkeypatch.setenv("SSH_CONNECTION", "x")
    pair_module._maybe_open_browser(console, _started(), no_browser=False)

    assert "SSH session" in console.text()
    assert pair_module._format_timestamp(None) == "unknown"


def test_an_opened_browser_is_mentioned(monkeypatch: pytest.MonkeyPatch) -> None:
    console = FakeConsole()

    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.delenv("SSH_TTY", raising=False)
    monkeypatch.setattr(pair_module.webbrowser, "open", lambda url: True)

    pair_module._maybe_open_browser(console, _started(), no_browser=False)

    assert "Opening your browser" in console.text()


def test_a_browser_that_will_not_open_falls_back_to_the_link(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    console = FakeConsole()

    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.delenv("SSH_TTY", raising=False)
    monkeypatch.setattr(pair_module.webbrowser, "open", lambda url: False)

    pair_module._maybe_open_browser(console, _started(), no_browser=False)

    assert "Use the link above" in console.text()


def test_pair_without_a_token_is_a_failure(monkeypatch: pytest.MonkeyPatch, no_sleep: None) -> None:
    from nomikos_inference.cli.api import STATUS_APPROVED

    monkeypatch.setattr(pair_module, "installed_version", lambda: "0.4.0")
    monkeypatch.setattr(pair_module, "this_machine_platform", lambda: "darwin-arm64")
    client = FakeClient(pairing=_started(), polls=[_poll(STATUS_APPROVED)])
    console, errors = FakeConsole(), FakeConsole()
    args = argparse.Namespace(name=None, no_browser=True)

    assert pair_module._pair(console, errors, client, args) == 1
    assert "sent no device token" in errors.text()


# ---------------------------------------------------------------------------
# _wait_for_approval
# ---------------------------------------------------------------------------
def test_approval_after_pending_returns_the_poll(no_sleep: None) -> None:
    from nomikos_inference.cli.api import STATUS_APPROVED, STATUS_AUTHORIZATION_PENDING

    approved = _poll(STATUS_APPROVED, device_id="d-1", device_token="t-1")
    client = FakeClient(polls=[_poll(STATUS_AUTHORIZATION_PENDING), approved])
    console, errors = FakeConsole(), FakeConsole()

    assert (
        pair_module._wait_for_approval(console, errors, client, _started(expires_in=60)) is approved
    )


def test_a_denied_request_stops_the_wait(no_sleep: None) -> None:
    from nomikos_inference.cli.api import STATUS_ACCESS_DENIED

    client = FakeClient(polls=[_poll(STATUS_ACCESS_DENIED)])
    console, errors = FakeConsole(), FakeConsole()

    assert pair_module._wait_for_approval(console, errors, client, _started()) is None
    assert "was refused" in errors.text()


def test_an_expired_request_stops_the_wait(no_sleep: None) -> None:
    from nomikos_inference.cli.api import STATUS_EXPIRED

    client = FakeClient(polls=[_poll(STATUS_EXPIRED)])
    console, errors = FakeConsole(), FakeConsole()

    assert pair_module._wait_for_approval(console, errors, client, _started()) is None
    assert "expired before it was approved" in errors.text()


def test_an_unknown_state_stops_the_wait_and_is_printed_inert(no_sleep: None) -> None:
    client = FakeClient(polls=[_poll("[bold]weird")])
    console, errors = FakeConsole(), FakeConsole()

    assert pair_module._wait_for_approval(console, errors, client, _started()) is None
    # `repr` first, then markup-escaped: Rich renders this literally, with the
    # quotes the original message always had. `escape` only touches `[` (the
    # one character that opens a tag), so the closing bracket stays bare.
    assert "'\\[bold]weird'" in errors.text()
    from rich.markup import render

    assert str(render(errors.lines[0])) == ("The platform answered an unknown state: '[bold]weird'")


def test_slow_down_is_honoured(monkeypatch: pytest.MonkeyPatch, no_sleep: None) -> None:
    from nomikos_inference.cli.api import STATUS_APPROVED, STATUS_SLOW_DOWN

    sleeps: list[float] = []
    monkeypatch.setattr(pair_module.time, "sleep", lambda seconds: sleeps.append(seconds))
    approved = _poll(STATUS_APPROVED, device_id="d-1", device_token="t-1")
    client = FakeClient(polls=[_poll(STATUS_SLOW_DOWN, interval_seconds=9), approved])
    console, errors = FakeConsole(), FakeConsole()

    assert (
        pair_module._wait_for_approval(
            console, errors, client, _started(expires_in=600, interval_seconds=1)
        )
        is approved
    )
    assert sleeps[1] == 9


def test_giving_up_after_the_deadline(no_sleep: None) -> None:
    from nomikos_inference.cli.api import STATUS_AUTHORIZATION_PENDING

    client = FakeClient(polls=[_poll(STATUS_AUTHORIZATION_PENDING)] * 3)
    console, errors = FakeConsole(), FakeConsole()

    assert (
        pair_module._wait_for_approval(console, errors, client, _started(expires_in=-100)) is None
    )
    assert "Gave up waiting" in errors.text()


def test_a_dying_platform_aborts_the_wait(no_sleep: None) -> None:
    client = FakeClient(polls=[PlatformError("gateway down")])
    console, errors = FakeConsole(), FakeConsole()

    with pytest.raises(PlatformError, match="gateway down"):
        pair_module._wait_for_approval(console, errors, client, _started())


# ---------------------------------------------------------------------------
# _report_existing
# ---------------------------------------------------------------------------
def test_a_credential_for_another_platform_is_not_checked() -> None:
    credential = _credential(platform_url="https://other.example.com")
    console, errors = FakeConsole(), FakeConsole()

    code = pair_module._report_existing(
        console, errors, FakeClient(), credential, "https://platform.example.com"
    )

    assert code == 1
    assert "paired with" in errors.text()


def test_an_unreachable_platform_leaves_the_credential_alone() -> None:
    console, errors = FakeConsole(), FakeConsole()

    code = pair_module._report_existing(
        console,
        errors,
        FakeClient(identity=PlatformError("offline")),
        _credential(),
        "https://platform.example.com",
    )

    assert code == 1
    assert "left alone" in errors.text()


def test_an_expired_token_is_named_as_expired() -> None:
    credential = _credential(token_expires_at=datetime.now(UTC) - timedelta(days=1))
    console, errors = FakeConsole(), FakeConsole()

    code = pair_module._report_existing(
        console, errors, FakeClient(identity=None), credential, "https://platform.example.com"
    )

    assert code == 1
    assert "has expired" in errors.text()


def test_a_revoked_token_is_named_as_removed() -> None:
    console, errors = FakeConsole(), FakeConsole()

    code = pair_module._report_existing(
        console,
        errors,
        FakeClient(identity=None),
        _credential(),
        "https://platform.example.com",
    )

    assert code == 1
    assert "has been removed" in errors.text()


def test_a_live_credential_reports_the_pairing() -> None:
    from nomikos_inference.cli.api import DeviceIdentity

    identity = DeviceIdentity(
        device_id="d-1",
        name="laptop",
        account_email="r@example.com",
        token_expires_at=datetime.now(UTC) + timedelta(days=1),
    )
    console, errors = FakeConsole(), FakeConsole()

    code = pair_module._report_existing(
        console,
        errors,
        FakeClient(identity=identity),
        _credential(),
        "https://platform.example.com",
    )

    assert code == 0
    assert "already paired" in console.text()
    assert "Token expires" in console.text()


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
def test_run_refuses_an_insecure_url() -> None:
    args = argparse.Namespace(api_url="http://platform.example.com", force=False)

    assert pair_module.run(args) == 1


def test_run_reports_an_unreadable_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        pair_module, "load_credential", lambda: (_ for _ in ()).throw(CredentialError("bad"))
    )
    args = argparse.Namespace(api_url=None, force=False)

    assert pair_module.run(args) == 1


def test_run_checks_the_existing_pairing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pair_module, "load_credential", _credential)
    monkeypatch.setattr(pair_module, "_report_existing", lambda *a: 7)
    args = argparse.Namespace(api_url=None, force=False)

    assert pair_module.run(args) == 7


def test_run_pairs_when_forced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pair_module, "load_credential", _credential)
    monkeypatch.setattr(pair_module, "_pair", lambda *a: 8)
    args = argparse.Namespace(api_url=None, force=True)

    assert pair_module.run(args) == 8


def test_run_pairs_when_unpaired(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pair_module, "load_credential", lambda: None)
    monkeypatch.setattr(pair_module, "_pair", lambda *a: 0)
    args = argparse.Namespace(api_url=None, force=False)

    assert pair_module.run(args) == 0


def test_run_reports_a_platform_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail(*args: object) -> int:
        raise PlatformError("gateway down")

    monkeypatch.setattr(pair_module, "load_credential", lambda: None)
    monkeypatch.setattr(pair_module, "_pair", _fail)
    args = argparse.Namespace(api_url=None, force=False)

    assert pair_module.run(args) == 1
