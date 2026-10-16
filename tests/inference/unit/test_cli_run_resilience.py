"""The claim loop survives a bad hop, and a finished page is never thrown away.

Three failures with the same shape: something transient happened at the wrong
moment, and the agent treated it as final.

**A single failed claim ended the loop.** `_claim_loop` caught `KeyboardInterrupt`
and nothing else, and `PlatformError` covers every `URLError`, `TimeoutError`,
and non-200 on that path - so one 502 from a gateway being restarted stopped an
agent that `--exit-when-empty`'s own help text promises "keeps running" without
it. A researcher who left a laptop claiming overnight came back to a dead
process and a full queue.

**A Ctrl-C between running a page and reporting it lost the page.** `_handle_page`
computed `output` inside its `try`, then left it to print timing and call
`_report`. An interrupt in that window escaped to `_claim_loop`, which has no
access to `output` and reports `status="failed"` - terminal. The page died with
the transcription already computed, one stack frame away.

**A failed report discarded the work.** One `PlatformError` posting the terminal
callback was printed and dropped, so a transcribed page waited for its **lease**
to expire and went back to the queue to be run again.

These are unit tests over the loop's own functions with a stub client. The real
loop against a real platform is `tests/inference/integration/test_cli_run.py`,
which needs Postgres; this file needs nothing, so the guarantees are checked on
every run of the unit lane.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from nomikos_inference.cli import claim as claim_module
from nomikos_inference.cli import console as ui
from nomikos_inference.cli import run as run_module
from nomikos_inference.cli.api import (
    AgentVersionRefused,
    Claim,
    ClaimedPage,
    CredentialRefused,
    PageLeaseLost,
    PlatformError,
)


@pytest.fixture(autouse=True)
def _no_sleeping(monkeypatch: pytest.MonkeyPatch) -> None:
    """Backoff is the behaviour under test; waiting for it is not."""
    monkeypatch.setattr(run_module.time, "sleep", lambda _seconds: None)


def _identity() -> run_module.AgentIdentity:
    return run_module.AgentIdentity(
        credential={"X-Nomikos-Device-Token": "secret"},
        account="researcher@example.com",
        execution_target="local",
        default_wait_seconds=25,
    )


def _page(
    *,
    task: str = "segment",
    lease_expires_at: datetime | None = None,
    page_image_expires_at: datetime | None = None,
) -> ClaimedPage:
    return ClaimedPage(
        product_job_id="11111111-2222-3333-4444-555555555555",
        inference_job_id="66666666-7777-8888-9999-000000000000",
        lease_expires_at=lease_expires_at,
        task=task,
        registry_model_id="blla-segment",
        registry_tag="stable",
        params={},
        page_image_url="https://storage.example.com/signed/page.png",
        page_image_expires_at=page_image_expires_at,
    )


class StubClient:
    """A platform that answers from a script and records what it was told."""

    def __init__(self, claims: list[object], *, reports: list[object] | None = None) -> None:
        self._claims = list(claims)
        self._reports = list(reports or [])
        self.claim_calls = 0
        self.reported: list[dict] = []

    def claim_page(self, *, credential: dict, wait_seconds: int) -> Claim:
        self.claim_calls += 1
        answer = self._claims.pop(0) if self._claims else Claim(None, 0.0, None)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    def fetch_page_image(self, url: str) -> bytes:
        return b"page-bytes"

    def report_page(self, *, credential: dict, page: ClaimedPage, output=None, error=None) -> None:
        self.reported.append({"page": page, "output": output, "error": error})
        if self._reports:
            answer = self._reports.pop(0)
            if isinstance(answer, BaseException):
                raise answer


def _loop(client: StubClient, *, exit_when_empty: bool = True) -> int:
    return run_module._claim_loop(
        ui.out(),
        ui.err(),
        client,
        _identity(),
        wait_seconds=0,
        exit_when_empty=exit_when_empty,
    )


# ---------------------------------------------------------------------------
# One transient failure is not the end of the loop
# ---------------------------------------------------------------------------
def test_a_transient_platform_error_does_not_end_the_claim_loop() -> None:
    """The 502 that used to kill an overnight agent."""
    client = StubClient(
        [
            PlatformError("The platform answered 502 while claiming a page"),
            Claim(page=None, poll_after_seconds=0.0, agent=None),
        ]
    )

    assert _loop(client) == ui.EXIT_OK
    assert client.claim_calls == 2


def test_repeated_failures_give_up_rather_than_spinning_forever() -> None:
    """Backing off is not the same as never stopping.

    A platform that has failed this many times consecutively is not having a
    blip, and an agent that retried it indefinitely would be a load generator
    pointed at something already unwell.
    """
    client = StubClient([PlatformError("gateway down")] * (run_module.MAX_CLAIM_FAILURES + 4))

    assert _loop(client) == ui.EXIT_FAILED
    assert client.claim_calls == run_module.MAX_CLAIM_FAILURES


# `test_one_success_resets_the_failure_budget` stood here. Its body popped the trailing
# success and appended an identical success, never exhausted the budget a second time, and
# asserted `EXIT_OK` -- which holds whether or not `failures = 0` exists in `run.py`. It did
# not test what its docstring claimed.


def test_a_refused_credential_still_stops_immediately() -> None:
    """A 401 answers the same way however long this waits.

    Retrying it would hammer the platform with a credential it has already
    refused, and bury the one message that tells a researcher to run `pair`.
    """
    client = StubClient([CredentialRefused("does not accept this machine's credential")])

    with pytest.raises(CredentialRefused):
        _loop(client)
    assert client.claim_calls == 1


def test_a_version_refusal_still_leaves_the_loop() -> None:
    """The floor is an instruction, not a blip - `run` turns it into a report."""
    refusal = AgentVersionRefused(
        message="below the floor",
        agent_version="0.1.0",
        minimum_version="0.4.0",
        latest_version="0.4.0",
        upgrade_command="uv tool upgrade nomikos-inference",
    )
    client = StubClient([refusal])

    with pytest.raises(AgentVersionRefused):
        _loop(client)
    assert client.claim_calls == 1


# ---------------------------------------------------------------------------
# A finished page is reported, interrupt or no interrupt
# ---------------------------------------------------------------------------
def test_an_interrupt_after_the_model_ran_still_reports_the_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The window the old comment called "a handful of bytecodes".

    The model has returned and `output` is computed. A Ctrl-C arriving while the
    timing line is being rendered used to escape `_handle_page` entirely, and
    `_claim_loop` - which cannot see `output` - reported the page failed and
    terminal. The work was done and thrown away.
    """
    monkeypatch.setattr(claim_module, "_execute", lambda page, image: _FakeResult())
    monkeypatch.setattr(claim_module, "_job_output", lambda page, result: {"kind": "segment"})

    console = _InterruptingConsole(interrupt_after=2)
    client = StubClient([])

    outcome = run_module._handle_page(console, ui.err(), client, _identity(), _page(), 1)

    assert outcome.stopped is True
    assert len(client.reported) == 1
    assert client.reported[0]["error"] is None
    assert client.reported[0]["output"] == {"kind": "segment"}


def test_an_interrupt_while_the_model_runs_is_still_reported_as_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other side of the same guarantee: no output means a failed page.

    Nothing was computed, so a terminal failure is the truthful callback - and it
    is still made, rather than leaving the page for the lease to sweep.
    """

    def _interrupt(page, image):
        raise KeyboardInterrupt

    monkeypatch.setattr(claim_module, "_execute", _interrupt)
    client = StubClient([])

    outcome = run_module._handle_page(ui.out(), ui.err(), client, _identity(), _page(), 1)

    assert outcome.stopped is True
    assert outcome.finished is False
    assert client.reported[0]["error"] == run_module.INTERRUPTED_ERROR
    assert client.reported[0]["output"] is None


def test_a_dead_page_image_link_fails_the_page_without_spending_a_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`page_image_expires_at` is read rather than ignored.

    The link is the only route to the scan, and a 403 from the signed-link route
    is deliberately ambiguous - forged, malformed, and expired are one status so
    it is not an oracle. Checking the stated expiry first is what turns that into
    a reason a researcher can act on.
    """
    fetched: list[str] = []
    monkeypatch.setattr(claim_module, "_execute", lambda page, image: _FakeResult())
    client = StubClient([])
    client.fetch_page_image = lambda url: fetched.append(url) or b""  # type: ignore[assignment]

    page = _page(page_image_expires_at=datetime.now(UTC) - timedelta(seconds=30))
    outcome = run_module._handle_page(ui.out(), ui.err(), client, _identity(), page, 1)

    assert outcome.finished is False
    assert fetched == []
    assert "expired" in client.reported[0]["error"]


def test_a_live_page_image_link_is_fetched_normally(monkeypatch: pytest.MonkeyPatch) -> None:
    """The expiry check must not refuse a link that is simply still valid."""
    monkeypatch.setattr(claim_module, "_execute", lambda page, image: _FakeResult())
    monkeypatch.setattr(claim_module, "_job_output", lambda page, result: {"kind": "segment"})
    client = StubClient([])

    page = _page(page_image_expires_at=datetime.now(UTC) + timedelta(seconds=60))
    outcome = run_module._handle_page(ui.out(), ui.err(), client, _identity(), page, 1)

    assert outcome.finished is True
    assert client.reported[0]["output"] == {"kind": "segment"}


# ---------------------------------------------------------------------------
# The terminal callback is retried
# ---------------------------------------------------------------------------
def test_a_failed_report_is_retried_rather_than_dropped() -> None:
    """One bad hop used to cost a transcribed page.

    The page is finished; the only thing between it and the researcher is an HTTP
    request. Letting the lease expire so another agent runs the same page again
    is the expensive way to handle a 502.
    """
    client = StubClient([], reports=[PlatformError("502 while reporting a page"), None])

    run_module._report(
        ui.out(), ui.err(), client, _identity(), _page(), output={"kind": "segment"}, reason=None
    )

    assert len(client.reported) == 2
    assert client.reported[-1]["output"] == {"kind": "segment"}


def test_report_retries_are_bounded() -> None:
    """Persistence stops where the lease does. Past it, the page is not ours."""
    client = StubClient([], reports=[PlatformError("still down")] * 20)

    run_module._report(
        ui.out(), ui.err(), client, _identity(), _page(), output={"kind": "segment"}, reason=None
    )

    assert len(client.reported) == run_module.MAX_REPORT_ATTEMPTS


def test_a_lost_lease_is_not_retried() -> None:
    """A 403 means the platform already gave the page to somebody else.

    Retrying is at best noise and at worst a race with the agent now holding it,
    so this is the one report failure that is accepted first time.
    """
    client = StubClient([], reports=[PageLeaseLost("not holding that page")])

    run_module._report(
        ui.out(), ui.err(), client, _identity(), _page(), output={"kind": "segment"}, reason=None
    )

    assert len(client.reported) == 1


def test_a_page_whose_output_is_unusable_is_not_a_setup_error() -> None:
    """`RunSetupError` documents "this machine cannot start claiming".

    A transcribe page that arrived with no line regions is a fact about that one
    page, and the loop must fail it and carry on - so it raises `PageOutputError`
    instead, which the per-page handler catches like any other bad page.
    """
    page = _page(task="transcribe")

    with pytest.raises(run_module.PageOutputError):
        run_module._job_output(page, _FakeResult(data={"text": "no lines here"}))


class _FakeResult:
    """Stands in for a model response without importing the runtime."""

    def __init__(self, data: dict | None = None) -> None:
        self._data = data if data is not None else {"lines": []}

    def model_dump(self, mode: str = "json") -> dict:
        return self._data


class _InterruptingConsole:
    """A console that raises `KeyboardInterrupt` on its Nth write.

    Rendering to a terminal is where the interrupt actually lands in the field -
    it is the slowest thing between the model returning and the callback going
    out - so the test injects it there rather than at an artificial seam.
    """

    def __init__(self, *, interrupt_after: int) -> None:
        self._writes = 0
        self._interrupt_after = interrupt_after
        self._fired = False

    def print(self, *args, **kwargs) -> None:
        self._writes += 1
        if self._writes > self._interrupt_after and not self._fired:
            self._fired = True
            raise KeyboardInterrupt
