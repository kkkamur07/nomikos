"""Claiming starts with an identity, and refuses to start without one.

`_resolve_identity` prefers a service credential from the environment (a host
given one is a hosted worker, whatever `device.json` says), and `run` turns
every setup failure into the same failed exit. Server strings printed on the
refusal paths are markup-escaped: `rich.markup.escape` only touches `[`, the
one character that opens a tag, so ordinary versions print byte-identical.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta

import pytest
from rich.markup import render

from nomikos_inference.cli import claim as claim_module
from nomikos_inference.cli import identity as identity_module
from nomikos_inference.cli import run as run_module
from nomikos_inference.cli.api import AgentVersionRefused, PlatformError
from nomikos_inference.cli.credentials import CredentialError, DeviceCredential
from nomikos_inference.cli.identity import AgentIdentity, RunSetupError


class FakeConsole:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def print(self, *args: object, **kwargs: object) -> None:
        self.lines.append(" ".join(str(a) for a in args))


def _device_credential() -> DeviceCredential:
    now = datetime.now(UTC)
    return DeviceCredential(
        platform_url="https://platform.example.com",
        device_id="device-1",
        device_token="secret-token",
        account_email="researcher@example.com",
        device_name="laptop",
        token_expires_at=now + timedelta(days=180),
        paired_at=now,
    )


# ---------------------------------------------------------------------------
# _resolve_identity
# ---------------------------------------------------------------------------
def test_a_service_token_makes_a_hosted_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(identity_module.SERVICE_TOKEN_ENV, "service-secret")
    monkeypatch.setenv(identity_module.WORKER_NAME_ENV, "worker-3")
    monkeypatch.setattr(identity_module, "installed_version", lambda: "0.4.0")

    identity = identity_module._resolve_identity("https://platform.example.com")

    assert identity.execution_target == "cloud"
    assert identity.default_wait_seconds == identity_module.WORKER_WAIT_SECONDS
    assert identity.credential["X-Nomikos-Service-Token"] == "service-secret"
    assert identity.credential["X-Nomikos-Worker-Name"] == "worker-3"


def test_a_service_token_without_a_name_uses_the_hostname(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(identity_module.SERVICE_TOKEN_ENV, "service-secret")
    monkeypatch.delenv(identity_module.WORKER_NAME_ENV, raising=False)
    monkeypatch.setattr(identity_module, "this_machine_name", lambda: "host-1")
    monkeypatch.setattr(identity_module, "installed_version", lambda: "0.4.0")

    identity = identity_module._resolve_identity("https://platform.example.com")

    assert identity.account == "the platform, as host-1"


def test_a_paired_machine_claims_local_work(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(identity_module.SERVICE_TOKEN_ENV, raising=False)
    monkeypatch.setattr(identity_module, "load_credential", lambda: _device_credential())
    monkeypatch.setattr(identity_module, "installed_version", lambda: "0.4.0")

    identity = identity_module._resolve_identity("https://platform.example.com")

    assert identity.execution_target == "local"
    assert identity.default_wait_seconds == identity_module.LAPTOP_WAIT_SECONDS
    assert identity.credential["X-Nomikos-Device-Token"] == "secret-token"


def test_an_unpaired_machine_cannot_claim(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(identity_module.SERVICE_TOKEN_ENV, raising=False)
    monkeypatch.setattr(identity_module, "load_credential", lambda: None)

    with pytest.raises(RunSetupError, match="not paired"):
        identity_module._resolve_identity("https://platform.example.com")


def test_a_credential_for_another_platform_cannot_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(identity_module.SERVICE_TOKEN_ENV, raising=False)
    monkeypatch.setattr(identity_module, "load_credential", lambda: _device_credential())

    with pytest.raises(RunSetupError, match="not https://other.example.com"):
        identity_module._resolve_identity("https://other.example.com")


def test_an_unreadable_credential_names_its_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _unreadable() -> DeviceCredential | None:
        raise CredentialError("not valid JSON: ...")

    monkeypatch.delenv(identity_module.SERVICE_TOKEN_ENV, raising=False)
    monkeypatch.setattr(identity_module, "load_credential", _unreadable)

    with pytest.raises(RunSetupError, match="nomikos pair"):
        identity_module._resolve_identity("https://platform.example.com")


def test_an_anonymous_credential_claims_for_this_account(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime.now(UTC)
    anonymous = DeviceCredential(
        platform_url="https://platform.example.com",
        device_id="device-1",
        device_token="secret-token",
        account_email="",
        device_name="laptop",
        token_expires_at=now + timedelta(days=180),
        paired_at=now,
    )
    monkeypatch.delenv(identity_module.SERVICE_TOKEN_ENV, raising=False)
    monkeypatch.setattr(identity_module, "load_credential", lambda: anonymous)
    monkeypatch.setattr(identity_module, "installed_version", lambda: "0.4.0")

    identity = identity_module._resolve_identity("https://platform.example.com")

    assert identity.account == "this account"


def _args(**overrides: object) -> argparse.Namespace:
    fields: dict[str, object] = {
        "api_url": None,
        "wait_seconds": None,
        "exit_when_empty": True,
    }
    fields.update(overrides)
    return argparse.Namespace(**fields)


def test_run_reports_a_setup_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _unpaired(base_url: str) -> AgentIdentity:
        raise RunSetupError("This machine is not paired")

    monkeypatch.setattr(run_module, "_resolve_identity", _unpaired)

    assert run_module.run(_args()) == 1
    assert "not paired" in capsys.readouterr().err


def test_run_claims_with_the_resolved_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def _fake_loop(
        console: object,
        errors: object,
        client: object,
        identity: AgentIdentity,
        *,
        wait_seconds: int,
        exit_when_empty: bool,
    ) -> int:
        seen["wait"] = wait_seconds
        seen["account"] = identity.account
        return 3

    monkeypatch.setattr(
        run_module,
        "_resolve_identity",
        lambda base_url: AgentIdentity(
            credential={},
            account="r@example.com",
            execution_target="local",
            default_wait_seconds=25,
        ),
    )
    monkeypatch.setattr(run_module, "_claim_loop", _fake_loop)

    assert run_module.run(_args()) == 3
    assert seen == {"wait": 25, "account": "r@example.com"}


def test_run_honours_an_explicit_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def _fake_loop(
        console: object,
        errors: object,
        client: object,
        identity: AgentIdentity,
        *,
        wait_seconds: int,
        exit_when_empty: bool,
    ) -> int:
        seen["wait"] = wait_seconds
        return 0

    monkeypatch.setattr(
        run_module,
        "_resolve_identity",
        lambda base_url: AgentIdentity(
            credential={},
            account="r@example.com",
            execution_target="local",
            default_wait_seconds=25,
        ),
    )
    monkeypatch.setattr(run_module, "_claim_loop", _fake_loop)

    assert run_module.run(_args(wait_seconds=-5)) == 0
    assert seen["wait"] == 0


def test_run_reports_a_floor_refusal_inert(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _refused(*args: object, **kwargs: object) -> int:
        raise AgentVersionRefused(
            message="too old",
            agent_version="0.1.0",
            minimum_version="0.4.0",
            latest_version="0.5.0",
            upgrade_command="[link=http://evil.example]x",
        )

    monkeypatch.setattr(
        run_module,
        "_resolve_identity",
        lambda base_url: AgentIdentity(
            credential={},
            account="r@example.com",
            execution_target="local",
            default_wait_seconds=0,
        ),
    )
    monkeypatch.setattr(run_module, "_claim_loop", _refused)

    assert run_module.run(_args()) == 1
    err = capsys.readouterr().err
    # Rendered literally (the URL visible) rather than as a live link.
    assert "http://evil.example" in err
    assert "Upgrade with" in err


def test_run_reports_a_dead_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    def _down(*args: object, **kwargs: object) -> int:
        raise PlatformError("gateway down")

    monkeypatch.setattr(
        run_module,
        "_resolve_identity",
        lambda base_url: AgentIdentity(
            credential={},
            account="r@example.com",
            execution_target="local",
            default_wait_seconds=0,
        ),
    )
    monkeypatch.setattr(run_module, "_claim_loop", _down)

    assert run_module.run(_args()) == 1


# ---------------------------------------------------------------------------
# Printing: server strings render inert
# ---------------------------------------------------------------------------
def test_a_hostile_upgrade_command_is_printed_literally() -> None:
    from nomikos_inference.cli.run_report import _report_refusal

    errors = FakeConsole()
    refusal = AgentVersionRefused(
        message="[link=http://evil.example]click[/link]",
        agent_version="[bold]0.1.0",
        minimum_version="0.4.0",
        latest_version="[i]0.5.0",
        upgrade_command="[link=http://evil.example]x",
    )

    _report_refusal(errors, refusal)

    upgrade_line = errors.lines[-1]
    assert "\\[link" in upgrade_line
    assert str(render(upgrade_line)) == (
        "Upgrade with [link=http://evil.example]x, then run this again."
    )
    assert str(render(errors.lines[0])) == "[link=http://evil.example]click[/link]"


def test_a_hostile_outdated_notice_is_printed_literally() -> None:
    from nomikos_inference.cli.api import AgentNotice
    from nomikos_inference.cli.run_report import _report_outdated

    console = FakeConsole()
    notice = AgentNotice(
        agent_version="[bold]0.4.0",
        latest_version="[i]0.5.0",
        outdated=True,
        upgrade_command="[link=http://evil.example]x",
    )

    _report_outdated(console, notice)

    assert "\\[bold" in console.lines[0]
    assert str(render(console.lines[0])).startswith("This agent is [bold]0.4.0;")


def test_stopped_summarises_and_interrupts() -> None:
    from nomikos_inference.cli.console import EXIT_INTERRUPTED
    from nomikos_inference.cli.run_report import _stopped

    console = FakeConsole()

    assert _stopped(console, 2, 1) == EXIT_INTERRUPTED
    assert "Stopped. 2 pages done, 1 failed." in console.lines[-1]


# ---------------------------------------------------------------------------
# The loop serves pages end to end
# ---------------------------------------------------------------------------
class _StubClient:
    """A platform that answers claims from a script and records reports."""

    def __init__(self, claims: list[object]):
        self._claims = list(claims)
        self.reported: list[dict[str, object]] = []

    def claim_page(self, *, credential: object, wait_seconds: int) -> object:
        answer = self._claims.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    def fetch_page_image(self, url: str) -> bytes:
        return b"page-bytes"

    def report_page(
        self, *, credential: object, page: object, output: object = None, error: object = None
    ) -> None:
        self.reported.append({"output": output, "error": error})


class _StubResult:
    def model_dump(self, mode: str = "json") -> dict[str, object]:
        return {"lines": []}


def _work_page() -> object:

    from nomikos_inference.cli.api import Claim, ClaimedPage

    return Claim(
        ClaimedPage(
            product_job_id="11111111-2222-3333-4444-555555555555",
            inference_job_id="66666666-7777-8888-9999-000000000000",
            lease_expires_at=None,
            task="segment",
            registry_model_id="blla-segment",
            registry_tag="stable",
            params={},
            page_image_url="https://storage.example.com/signed/page.png",
            page_image_expires_at=None,
        ),
        0.0,
        None,
    )


def _loop_identity() -> AgentIdentity:
    return AgentIdentity(
        credential={},
        account="r@example.com",
        execution_target="local",
        default_wait_seconds=0,
    )


def test_the_loop_serves_a_page_then_exits_when_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nomikos_inference.cli.api import Claim
    from nomikos_inference.cli.console import EXIT_OK

    monkeypatch.setattr(claim_module, "_execute", lambda page, image: _StubResult())
    monkeypatch.setattr(claim_module.time, "sleep", lambda _seconds: None)
    console = FakeConsole()
    client = _StubClient([_work_page(), Claim(None, 0.0, None)])

    code = claim_module._claim_loop(
        console, FakeConsole(), client, _loop_identity(), wait_seconds=0, exit_when_empty=True
    )

    assert code == EXIT_OK
    assert console.lines[-1] == "Nothing left in the queue. 1 page done, 0 failed."
    assert client.reported[0] == {
        "output": {"kind": "segment", "data": {"lines": []}},
        "error": None,
    }


def test_the_loop_files_a_failed_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nomikos_inference.cli.api import Claim
    from nomikos_inference.cli.console import EXIT_OK

    def _broken(page: object, image: bytes) -> object:
        raise ValueError("bad scan")

    monkeypatch.setattr(claim_module, "_execute", _broken)
    monkeypatch.setattr(claim_module.time, "sleep", lambda _seconds: None)
    console = FakeConsole()
    client = _StubClient([_work_page(), Claim(None, 0.0, None)])

    code = claim_module._claim_loop(
        console, FakeConsole(), client, _loop_identity(), wait_seconds=0, exit_when_empty=True
    )

    assert code == EXIT_OK
    assert console.lines[-1] == "Nothing left in the queue. 0 pages done, 1 failed."
    assert client.reported[0]["error"] == "bad scan"


def test_an_interrupt_between_claims_stops_after_the_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nomikos_inference.cli.console import EXIT_INTERRUPTED

    monkeypatch.setattr(claim_module, "_execute", lambda page, image: _StubResult())
    monkeypatch.setattr(claim_module.time, "sleep", lambda _seconds: None)
    console = FakeConsole()
    client = _StubClient([_work_page(), KeyboardInterrupt()])

    code = claim_module._claim_loop(
        console, FakeConsole(), client, _loop_identity(), wait_seconds=0, exit_when_empty=False
    )

    assert code == EXIT_INTERRUPTED
    assert console.lines[-1] == "Stopped. 1 page done, 0 failed."


def test_an_interrupt_while_backing_off_stops_at_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nomikos_inference.cli.api import Claim, PlatformError
    from nomikos_inference.cli.console import EXIT_INTERRUPTED

    calls: list[float] = []

    def _sleep_once(seconds: float) -> None:
        calls.append(seconds)
        if len(calls) == 1:
            raise KeyboardInterrupt

    monkeypatch.setattr(claim_module.time, "sleep", _sleep_once)
    console = FakeConsole()
    client = _StubClient([PlatformError("gateway down"), Claim(None, 0.0, None)])

    code = claim_module._claim_loop(
        console, FakeConsole(), client, _loop_identity(), wait_seconds=0, exit_when_empty=True
    )

    assert code == EXIT_INTERRUPTED
    assert console.lines[-1] == "Stopped. 0 pages done, 0 failed."


def test_waiting_for_work_announces_and_stops_on_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nomikos_inference.cli.api import Claim
    from nomikos_inference.cli.console import EXIT_INTERRUPTED

    calls = 0

    def _once(seconds: float) -> None:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise KeyboardInterrupt

    monkeypatch.setattr(claim_module.time, "sleep", _once)
    console = FakeConsole()
    client = _StubClient([Claim(None, 99.0, None), Claim(None, 99.0, None)])

    code = claim_module._claim_loop(
        console, FakeConsole(), client, _loop_identity(), wait_seconds=0, exit_when_empty=False
    )

    assert code == EXIT_INTERRUPTED
    assert "[dim]Waiting for work. Press Ctrl-C to stop.[/dim]" in console.lines
    assert console.lines[-1] == "Stopped. 0 pages done, 0 failed."


# ---------------------------------------------------------------------------
# Interrupts while reporting
# ---------------------------------------------------------------------------
def _report_page() -> object:
    from nomikos_inference.cli.api import ClaimedPage

    return ClaimedPage(
        product_job_id="11111111-2222-3333-4444-555555555555",
        inference_job_id="66666666-7777-8888-9999-000000000000",
        lease_expires_at=None,
        task="segment",
        registry_model_id="blla-segment",
        registry_tag="stable",
        params={},
        page_image_url="https://storage.example.com/signed/page.png",
        page_image_expires_at=None,
    )


def test_two_interrupts_while_reporting_abandon_the_page() -> None:

    calls: list[str] = []

    class _TwiceInterruptedClient(_StubClient):
        def report_page(
            self,
            *,
            credential: object,
            page: object,
            output: object = None,
            error: object = None,
        ) -> None:
            calls.append("report")
            raise KeyboardInterrupt

    console, errors = FakeConsole(), FakeConsole()
    client = _TwiceInterruptedClient([])

    claim_module._report(
        console,
        errors,
        client,
        _loop_identity(),
        _report_page(),
        output={"kind": "segment"},
        reason=None,
    )

    assert calls == ["report", "report"]
    assert "still reporting this page" in console.lines[0]
    assert "Abandoned while reporting." in errors.lines[-1]


def test_an_interrupt_while_backing_off_abandons_the_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nomikos_inference.cli.api import PlatformError

    class _FlakyClient(_StubClient):
        def report_page(
            self,
            *,
            credential: object,
            page: object,
            output: object = None,
            error: object = None,
        ) -> None:
            raise PlatformError("gateway down")

    def _never(seconds: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(claim_module.time, "sleep", _never)
    errors = FakeConsole()
    client = _FlakyClient([])

    claim_module._report(
        FakeConsole(),
        errors,
        client,
        _loop_identity(),
        _report_page(),
        output={"kind": "segment"},
        reason=None,
    )

    assert "Abandoned while reporting." in errors.lines[-1]


def test_an_interrupt_before_the_page_starts_is_still_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nomikos_inference.cli.console import EXIT_INTERRUPTED

    calls = 0

    def _monotonic() -> float:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise KeyboardInterrupt
        return 0.0

    monkeypatch.setattr(claim_module.time, "monotonic", _monotonic)
    monkeypatch.setattr(claim_module.time, "sleep", lambda _seconds: None)
    console = FakeConsole()
    client = _StubClient([_work_page()])

    code = claim_module._claim_loop(
        console, FakeConsole(), client, _loop_identity(), wait_seconds=0, exit_when_empty=True
    )

    assert code == EXIT_INTERRUPTED
    assert client.reported[0]["error"] == claim_module.INTERRUPTED_ERROR


def test_an_interrupt_while_the_model_runs_stops_the_loop_after_reporting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nomikos_inference.cli.api import Claim
    from nomikos_inference.cli.console import EXIT_INTERRUPTED

    def _interrupt(page: object, image: bytes) -> object:
        raise KeyboardInterrupt

    monkeypatch.setattr(claim_module, "_execute", _interrupt)
    monkeypatch.setattr(claim_module.time, "sleep", lambda _seconds: None)
    console = FakeConsole()
    client = _StubClient([_work_page(), Claim(None, 0.0, None)])

    code = claim_module._claim_loop(
        console, FakeConsole(), client, _loop_identity(), wait_seconds=0, exit_when_empty=True
    )

    assert code == EXIT_INTERRUPTED
    assert console.lines[-1] == "Stopped. 0 pages done, 1 failed."
    assert client.reported[0]["error"] == claim_module.INTERRUPTED_ERROR


def test_an_outdated_notice_prints_once_per_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nomikos_inference.cli.api import AgentNotice, Claim
    from nomikos_inference.cli.console import EXIT_OK

    monkeypatch.setattr(claim_module.time, "sleep", lambda _seconds: None)
    notice = AgentNotice(
        agent_version="0.4.0", latest_version="0.5.0", outdated=True, upgrade_command=""
    )
    console = FakeConsole()
    client = _StubClient([Claim(None, 0.0, notice), Claim(None, 0.0, notice)])
    code = claim_module._claim_loop(
        console, FakeConsole(), client, _loop_identity(), wait_seconds=0, exit_when_empty=True
    )

    assert code == EXIT_OK
    assert sum("is out" in line for line in console.lines) == 1


def test_short_byte_counts() -> None:
    from nomikos_inference.cli.run_report import _bytes

    assert _bytes(512) == "512 B"
    assert _bytes(2048) == "2.0 KB"
    assert _bytes(3 * 1024 * 1024) == "3.0 MB"


def test_a_reason_without_words_falls_back_to_the_class_name() -> None:
    assert claim_module._reason(ValueError()) == "ValueError"
    assert claim_module._reason(ValueError("x" * 200)) == "x" * 160
