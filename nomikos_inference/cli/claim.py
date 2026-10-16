"""The claim loop: take a page, run it here, report it, repeat.

Split from `run.py` (the entry point) alongside `identity.py` (whose
credential claims) and `run_report.py` (what claiming prints), so this module
holds only work in flight: the loop itself, running one page, and posting the
terminal callback.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from rich.markup import escape

from nomikos_inference.cli import console as ui
from nomikos_inference.cli.api import (
    AgentVersionRefused,
    Claim,
    ClaimedPage,
    CredentialRefused,
    PageLeaseLost,
    PlatformClient,
    PlatformError,
)
from nomikos_inference.cli.identity import AgentIdentity
from nomikos_inference.cli.run_report import (
    _abandoned,
    _bytes,
    _reason,
    _report_outdated,
    _short,
    _stopped,
    _summarise,
)

INTERRUPTED_ERROR = "Stopped on the machine running it before the page finished"

CLAIM_RETRY_SECONDS = 2.0
CLAIM_RETRY_CAP_SECONDS = 60.0
MAX_CLAIM_FAILURES = 8
"""How many *consecutive* failed claims end the loop.

A gateway restarting, a deploy, wifi dropping in a lift: every one of those is
a `PlatformError` and none of them means "stop working". The loop backs off
instead, doubling to a cap so an agent left running overnight against a dead
platform isn't a load generator, and gives up only once the platform has
failed this many times in a row. One success resets it.
"""

REPORT_RETRY_SECONDS = 1.0
REPORT_RETRY_CAP_SECONDS = 8.0
MAX_REPORT_ATTEMPTS = 4
"""How many times the terminal callback is posted before the page is let go.

Shorter and more insistent than the claim backoff: the stakes aren't
symmetric, since a failed claim costs nothing but a failed report throws away
a page that's already been transcribed. Bounded anyway, since past the
**lease** the platform has re-queued the page and posting into that is not
persistence.
"""


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------
@dataclass
class _LoopProgress:
    """Where one `_claim_loop` run has got to. Mutable, threaded through the
    helpers below so the loop itself stays a straight sequence."""

    done: int = 0
    failed: int = 0
    claimed: int = 0
    told_outdated: bool = False
    waiting_announced: bool = False
    failures: int = 0
    retry_seconds: float = CLAIM_RETRY_SECONDS


class _LoopExit(Exception):
    """The loop is over: carry its exit code out through the helpers.

    Stopping (Ctrl-C, an empty queue with `--exit-when-empty`, too many failed
    claims) is decided deep inside one poll, but the code that returns it is
    the loop. An exception keeps both helpers flat instead of threading
    "stop or continue" through every level as a return value.
    """

    def __init__(self, code: int) -> None:
        super().__init__(code)
        self.code = code


def _claim_loop(
    console,
    errors,
    client: PlatformClient,
    identity: AgentIdentity,
    *,
    wait_seconds: int,
    exit_when_empty: bool,
) -> int:
    done = _LoopProgress()
    try:
        while True:
            page = _take_claim(
                console, errors, client, identity, wait_seconds, exit_when_empty, done
            )
            stopped = _serve_claim(console, errors, client, identity, done, page)
            if stopped is not None:
                return stopped
    except _LoopExit as end:
        return end.code


def _take_claim(
    console,
    errors,
    client: PlatformClient,
    identity: AgentIdentity,
    wait_seconds: int,
    exit_when_empty: bool,
    done: _LoopProgress,
) -> ClaimedPage:
    """Poll until the platform hands over a page. Raises `_LoopExit` when the
    loop ends before that: an interrupt, a refusal that retries cannot fix, an
    empty queue under `--exit-when-empty`, or too many failed claims."""
    while True:
        try:
            claim = client.claim_page(credential=identity.credential, wait_seconds=wait_seconds)
        except KeyboardInterrupt:
            # Nothing is in flight between claims, which is the whole reason
            # the loop is shaped this way: there's no page here to report.
            raise _LoopExit(_stopped(console, done.done, done.failed)) from None
        except (AgentVersionRefused, CredentialRefused):
            # Neither is a blip: the floor is an instruction, and a refused
            # credential answers the same way no matter how long this waits.
            # Both leave the loop rather than being retried into a spin.
            raise
        except PlatformError as exc:
            # Everything else here is transient by construction: a timeout, an
            # unreachable host, a gateway's 502. One bad response isn't an
            # empty queue.
            stopped = _note_claim_failure(console, errors, done, exc)
            if stopped is not None:
                raise _LoopExit(stopped) from None
            continue

        done.failures = 0
        done.retry_seconds = CLAIM_RETRY_SECONDS

        page = _settle_claim(console, done, claim, exit_when_empty)
        if page is None:
            continue

        done.waiting_announced = False
        return page


def _settle_claim(
    console, done: _LoopProgress, claim: Claim, exit_when_empty: bool
) -> ClaimedPage | None:
    """File one answered claim: note an outdated build once, wait out an empty
    queue, or hand over the page. `None` means keep polling; stopping raises
    `_LoopExit`."""
    if claim.agent is not None and claim.agent.outdated and not done.told_outdated:
        done.told_outdated = True
        _report_outdated(console, claim.agent)

    if claim.page is None:
        stopped = _await_work(console, done, claim, exit_when_empty)
        if stopped is not None:
            raise _LoopExit(stopped)
        return None

    return claim.page


def _serve_claim(
    console,
    errors,
    client: PlatformClient,
    identity: AgentIdentity,
    done: _LoopProgress,
    page: ClaimedPage,
) -> int | None:
    """Run and report one claimed page. The exit code when the loop stops
    after it, else `None`."""
    try:
        outcome = _handle_page(console, errors, client, identity, page, done.claimed + 1)
    except KeyboardInterrupt:
        # Reachable only before `_handle_page` has computed anything: it
        # takes responsibility for the page on entry and reports whatever
        # it has, so an interrupt that escapes it arrived with no output
        # to lose. Reporting failed here closes the one window in which a
        # page could be claimed and never ended.
        _report(
            console,
            errors,
            client,
            identity,
            page,
            output=None,
            reason=INTERRUPTED_ERROR,
        )
        return _stopped(console, done.done, done.failed + 1)

    if _tally(done, outcome):
        return _stopped(console, done.done, done.failed)
    return None


def _note_claim_failure(console, errors, done: _LoopProgress, exc: PlatformError) -> int | None:
    """Absorb one failed claim: back off, or give up after too many in a row."""
    done.failures += 1
    if done.failures >= MAX_CLAIM_FAILURES:
        errors.print(f"[red]{escape(str(exc))}[/red]")
        errors.print(
            f"[red]Giving up after {done.failures} claims in a row failed.[/red] "
            "Nothing is in flight; run this again when the platform is back."
        )
        _summarise(console, done.done, done.failed, "Stopped.")
        return ui.EXIT_FAILED
    errors.print(
        f"[yellow]{escape(str(exc))}[/yellow] Retrying in {done.retry_seconds:g}s "
        f"({done.failures}/{MAX_CLAIM_FAILURES})."
    )
    done.waiting_announced = False
    try:
        time.sleep(done.retry_seconds)
    except KeyboardInterrupt:
        return _stopped(console, done.done, done.failed)
    done.retry_seconds = min(done.retry_seconds * 2, CLAIM_RETRY_CAP_SECONDS)
    return None


def _await_work(console, done: _LoopProgress, claim: Claim, exit_when_empty: bool) -> int | None:
    """Absorb one empty queue: stop for `--exit-when-empty`, else wait."""
    if exit_when_empty:
        _summarise(console, done.done, done.failed, "Nothing left in the queue.")
        return ui.EXIT_OK
    if not done.waiting_announced:
        done.waiting_announced = True
        console.print("[dim]Waiting for work. Press Ctrl-C to stop.[/dim]")
    try:
        time.sleep(max(claim.poll_after_seconds, 0.5))
    except KeyboardInterrupt:
        return _stopped(console, done.done, done.failed)
    return None


def _tally(done: _LoopProgress, outcome: PageOutcome) -> bool:
    """File one finished page, and say whether the loop stops after it."""
    done.claimed += 1
    if outcome.finished:
        done.done += 1
    else:
        done.failed += 1
    return outcome.stopped


class PageOutputError(RuntimeError):
    """One page's result cannot be made into a callback the platform will take.

    A property of that page, not of this machine, so it fails the page and
    the loop moves on: the opposite of what `RunSetupError` means.
    """


@dataclass(frozen=True)
class PageOutcome:
    """What became of one page, and whether the loop should keep going."""

    finished: bool
    stopped: bool
    """A Ctrl-C arrived while this page was in flight. The page has already
    been reported by the time this is read; stopping is all that's left."""


def _handle_page(
    console,
    errors,
    client: PlatformClient,
    identity: AgentIdentity,
    page: ClaimedPage,
    index: int,
) -> PageOutcome:
    """Run one page and report it, whatever happens to it.

    This deliberately never raises `KeyboardInterrupt`. Reporting the page is
    the *last* thing that must survive an interrupt: an exception that skips
    past the callback would leave a researcher watching a page nobody is
    running any more, which is the one bug this whole design exists to
    prevent.

    The reporting call is inside that protection rather than after it, because
    the gap between the model returning and the callback going out (a timing
    calculation and two console writes, with rich rendering to a terminal) is
    long enough to catch a Ctrl-C. An interrupt landing there would escape to
    `_claim_loop`, which can't see `output` and would report the page *failed*
    with the transcription already computed and about to be discarded.
    """
    output: dict | None = None
    reason: str | None = None
    interrupted = False
    started = time.monotonic()

    try:
        console.print(
            f"[bold][{index}][/bold] {page.task} [cyan]{page.registry_model_id}[/cyan] "
            f"job {_short(page.product_job_id)}"
        )
        if page.image_link_expired():
            raise PageOutputError(
                "the link to this page's image expired before it could be fetched"
            )
        image = client.fetch_page_image(page.page_image_url)
        console.print(f"    fetched {_bytes(len(image))}")
        output = _job_output(page, _execute(page, image))
    except KeyboardInterrupt:
        interrupted = True
        reason = INTERRUPTED_ERROR
    except Exception as exc:  # noqa: BLE001 - one bad page is not a bad loop
        reason = _reason(exc)

    # Past this point the page's fate is decided and only saying so is left.
    # An interrupt here is recorded and swallowed rather than obeyed: `_report`
    # has its own two-Ctrl-C bargain with the researcher, and reaching it is
    # the whole point.
    for first_pass in (True, False):
        try:
            if first_pass:
                elapsed = time.monotonic() - started
                if reason is None:
                    console.print(f"    ran in {elapsed:.1f}s")
                else:
                    console.print(f"    [red]failed after {elapsed:.1f}s:[/red] {reason}")
            _report(console, errors, client, identity, page, output=output, reason=reason)
            break
        except KeyboardInterrupt:
            interrupted = True

    return PageOutcome(finished=reason is None, stopped=interrupted)


def _report(
    console,
    errors,
    client: PlatformClient,
    identity: AgentIdentity,
    page: ClaimedPage,
    *,
    output: dict | None,
    reason: str | None,
) -> None:
    """Post the terminal callback, absorbing one Ctrl-C and retrying a bad hop.

    The first extra interrupt is swallowed and explained, because a
    researcher pressing Ctrl-C twice in a second means "stop", not "leave
    that page half-reported". The second is honoured: at that point the
    **lease** is the right mechanism, and it re-queues the page rather than
    failing it.

    A failed post is retried, backing off, up to `MAX_REPORT_ATTEMPTS`,
    because the page is already *finished* at this point: the only thing
    between a researcher and their transcription is one HTTP request, and a
    502 from a gateway shouldn't throw the result away while waiting for the
    lease to expire.

    `PageLeaseLost` is the exception to that: a 403 means the platform has
    already given the page to someone else, and there's nothing left here to
    deliver it to.
    """
    attempts = 0
    retry_seconds = REPORT_RETRY_SECONDS
    interrupted_once = False

    while True:
        try:
            client.report_page(
                credential=identity.credential, page=page, output=output, error=reason
            )
        except KeyboardInterrupt:
            if _note_interrupt(console, errors, interrupted_once):
                return
            interrupted_once = True
            continue
        except PlatformError as exc:
            if isinstance(exc, PageLeaseLost):
                # Terminal for this agent, not a bad hop: the 403 means the
                # platform has already given the page to someone else, so
                # there is nothing left here to deliver it to.
                errors.print(f"    [red]{exc}[/red]")
                return
            attempts += 1
            if not _retry_report_post(errors, exc, attempts, retry_seconds):
                return
            retry_seconds = min(retry_seconds * 2, REPORT_RETRY_CAP_SECONDS)
            continue
        console.print(f"    reported {'done' if reason is None else 'failed'}")
        return


def _note_report_failure(
    errors, exc: PlatformError, attempts: int, retry_seconds: float
) -> float | None:
    """Record one failed report post. The next backoff, or `None` to give up
    and leave the page to its lease."""
    if attempts >= MAX_REPORT_ATTEMPTS:
        errors.print(f"    [red]{escape(str(exc))}[/red]")
        errors.print(
            f"    [red]This page could not be reported after {attempts} "
            "attempts.[/red] Its lease will expire and the platform will "
            "hand it to another agent."
        )
        return None
    errors.print(
        f"    [yellow]{escape(str(exc))}[/yellow] Retrying in {retry_seconds:g}s "
        f"({attempts}/{MAX_REPORT_ATTEMPTS})."
    )
    return retry_seconds


def _retry_report_post(errors, exc: PlatformError, attempts: int, retry_seconds: float) -> bool:
    """Print one failed post and sleep the backoff. False when the page is
    let go: either reporting gave up, or a Ctrl-C abandoned it."""
    retry = _note_report_failure(errors, exc, attempts, retry_seconds)
    if retry is None:
        return False
    return _sleep_or_abandon(errors, retry)


def _note_interrupt(console, errors, interrupted_once: bool) -> bool:
    """Absorb one Ctrl-C while reporting. True when it was the second one and
    the page is abandoned to its lease."""
    if not interrupted_once:
        console.print("    [dim]still reporting this page - press Ctrl-C again to abandon it[/dim]")
        return False
    _abandoned(errors)
    return True


def _sleep_or_abandon(errors, retry_seconds: float) -> bool:
    """Sleep between report retries. A Ctrl-C abandons the page to its lease."""
    try:
        time.sleep(retry_seconds)
    except KeyboardInterrupt:
        _abandoned(errors)
        return False
    return True


# ---------------------------------------------------------------------------
# Running the page
# ---------------------------------------------------------------------------
def _execute(page: ClaimedPage, image_bytes: bytes):
    """Call the same `run_model` the platform's own worker calls.

    Imported here rather than at module scope: this is the only line in the
    CLI that needs Torch, and `nomikos version` must not pay for it.
    """
    from nomikos_inference.contracts.common import InferenceTask
    from nomikos_inference.jobs.runner import run_model

    return run_model(
        task=InferenceTask(page.task),
        registry_model_id=page.registry_model_id,
        registry_tag=page.registry_tag,
        image_bytes=image_bytes,
        params=page.params,
    )


def _job_output(page: ClaimedPage, result) -> dict:
    """Wrap the model's answer in the callback's discriminated union.

    `kind` has to equal the task or the platform rejects the callback, so
    this also catches the one shape mismatch that can happen honestly: a
    transcribe page whose line regions didn't survive, coming back as a
    single-line response where a batch was required.
    """
    data = result.model_dump(mode="json")
    if page.task == "transcribe" and "lines" not in data:
        raise PageOutputError("this page carried no line regions to transcribe")
    return {"kind": page.task, "data": data}
