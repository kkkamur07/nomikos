"""What `nomikos run` prints about versions, and how it ends a session.

Server-supplied strings are rendered inert here, never as live Rich markup: a
value containing `[link=...]` or a style tag would otherwise disguise the
command a researcher is told to run. `rich.markup.escape` leaves ordinary text
byte-identical, so the printed contract only changes for inputs that were
trying to be markup.
"""

from __future__ import annotations

from rich.markup import escape

from nomikos_inference.cli import console as ui
from nomikos_inference.cli.api import AgentNotice, AgentVersionRefused

ABANDONED_LINE = (
    "    [yellow]Abandoned while reporting.[/yellow] The platform releases "
    "this page when its lease expires, and another agent may take it."
)
"""Printed when a second Ctrl-C abandons a page mid-report. One definition for
the two places that print it, so they cannot drift apart."""


def _stopped(console, done: int, failed: int) -> int:
    """The shared Ctrl-C exit: say what finished, and stop with 130."""
    console.print()
    _summarise(console, done, failed, "Stopped.")
    return ui.EXIT_INTERRUPTED


def _abandoned(errors) -> None:
    """Say the page was left for its lease to re-queue."""
    errors.print(ABANDONED_LINE)


def _report_refusal(errors, refusal: AgentVersionRefused) -> None:
    """A 426 is not a blip to back off from - it is an instruction."""
    errors.print(f"[red]{escape(str(refusal))}[/red]")
    errors.print()
    table = ui.detail_table()
    table.add_row("This agent", ui.value(refusal.agent_version or "not stated"))
    table.add_row("Platform needs", ui.value(f"{refusal.minimum_version} or newer"))
    table.add_row("Newest release", ui.value(refusal.latest_version))
    errors.print(table)
    errors.print()
    errors.print(
        f"Upgrade with [bold]{escape(refusal.upgrade_command)}[/bold], then run this again."
    )


def _report_outdated(console, notice: AgentNotice) -> None:
    """Printed once per session, not once per claim: the notice rides every
    claim response, and repeating it every few seconds would bury the work."""
    console.print(
        f"[yellow]This agent is {escape(notice.agent_version)}; "
        f"{escape(notice.latest_version)} is out.[/yellow] "
        "It is still being served. "
        f"Upgrade with [bold]{escape(notice.upgrade_command)}[/bold]."
    )
    console.print()


def _summarise(console, done: int, failed: int, headline: str) -> None:
    console.print()
    console.print(f"{headline} {_count(done, 'page')} done, {failed} failed.")


def _count(value: int, noun: str) -> str:
    return f"{value} {noun}" if value == 1 else f"{value} {noun}s"


def _bytes(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def _short(identifier: str) -> str:
    return identifier.split("-", 1)[0]


MAX_REASON_CHARS = 160
"""The platform prefixes the reason, redacts URLs and paths out of it, and
truncates at 200. Staying well inside that keeps a reason readable instead of
ending in a placeholder."""


def _reason(error: BaseException) -> str:
    """Why this page failed, in a form the platform will still store as words.

    The platform redacts URLs, paths, and anything token-shaped out of a
    callback error before storing it, so a reason built from a traceback's
    worth of file paths arrives as a row of placeholders. This collapses
    whitespace, truncates, and falls back to the exception's class name when
    it has nothing to say.
    """
    text = " ".join(str(error).split())
    if not text:
        text = error.__class__.__name__
    return text[:MAX_REASON_CHARS]
