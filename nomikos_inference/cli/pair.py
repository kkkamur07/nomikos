"""`nomikos pair` - authorise this machine against a researcher's account.

The order of what this prints is the design, not the presentation:

1. **The pairing URL, first and always.** ADR 0002 demotes `webbrowser.open()`
   from *the only affordance the process has* to a convenience, since opening
   a browser is actively wrong over SSH. The printed URL is what makes the
   flow work everywhere; the browser is layered on top and may fail silently
   without costing anything.
2. **The confirmation code, before the wait begins.** ADR 0001 decision 13
   derived a code for the consent screen, noting that nothing displayed it on
   the client, which left the screen asking for consent with nothing
   checkable on it (every other field there is supplied by whoever started
   the pairing). Printing it is the whole mitigation: a researcher sent a
   phishing link sees a code their own terminal never showed.

Two states end this command without a new device row. A machine that's
already paired says so, since silently starting a second pairing would leave
an orphan device on the account with no explanation. A machine whose
credential the platform no longer accepts reports that and exits non-zero,
since revocation is a decision someone made in a browser (ADR 0001, decision
11), and re-pairing over it without saying so would undo that decision
quietly.
"""

from __future__ import annotations

import argparse
import os
import time
import webbrowser
from datetime import UTC, datetime

from rich.markup import escape

from nomikos_inference.cli import console as ui
from nomikos_inference.cli.api import (
    DEVICE_NAME_LIMIT,
    STATUS_ACCESS_DENIED,
    STATUS_APPROVED,
    STATUS_AUTHORIZATION_PENDING,
    STATUS_EXPIRED,
    STATUS_SLOW_DOWN,
    PlatformClient,
    PlatformError,
    StartedPairing,
    default_platform_url,
    this_machine_name,
    this_machine_platform,
)
from nomikos_inference.cli.credentials import (
    CredentialError,
    DeviceCredential,
    credential_path,
    file_mode,
    load_credential,
    save_credential,
)
from nomikos_inference.cli.version import installed_version

# What this machine tells the platform it can do. The **execution target** a
# job runs on is decided at submission from **host eligibility** and
# **capacity**, not from this - it's recorded for support, and for the day a
# second runtime exists to distinguish.
CAPABILITIES = {"runtime": "torch"}

_MINIMUM_POLL_SECONDS = 1
_POLL_MARGIN_SECONDS = 5
"""Kept polling this far past the advertised expiry, so the platform is what
declares a pairing dead, not the client's own clock."""


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--api-url",
        metavar="URL",
        help="Platform to pair against. Defaults to $NOMIKOS_API_URL, then the hosted API.",
    )
    parser.add_argument(
        "--name",
        metavar="NAME",
        help="What to call this machine on the account. Defaults to its hostname.",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Only print the URL. Implied when this looks like an SSH session.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Pair again even though this machine already holds a credential. "
            "The existing device stays on the account until it is removed there."
        ),
    )


def run(args: argparse.Namespace) -> int:
    console = ui.out()
    errors = ui.err()

    base_url = (args.api_url or default_platform_url()).rstrip("/")
    try:
        client = PlatformClient(base_url)
    except PlatformError as exc:
        errors.print(f"[red]{exc}[/red]")
        return ui.EXIT_FAILED

    try:
        existing = load_credential()
    except CredentialError as exc:
        errors.print(f"[red]{exc}[/red]")
        errors.print(f"Delete {credential_path()} and run `nomikos pair` again to replace it.")
        return ui.EXIT_FAILED

    if existing is not None and not args.force:
        return _report_existing(console, errors, client, existing, base_url)

    try:
        return _pair(console, errors, client, args)
    except PlatformError as exc:
        errors.print(f"[red]{exc}[/red]")
        return ui.EXIT_FAILED


# ---------------------------------------------------------------------------
# Already paired, or paired and no longer accepted
# ---------------------------------------------------------------------------
def _report_existing(
    console,
    errors,
    client: PlatformClient,
    credential: DeviceCredential,
    base_url: str,
) -> int:
    if credential.platform_url != base_url:
        errors.print(
            f"[red]This machine is paired with {credential.platform_url}, not {base_url}.[/red]"
        )
        errors.print(
            "One credential is stored per machine. Run `nomikos pair --force` to replace it."
        )
        return ui.EXIT_FAILED

    try:
        identity = client.read_self(device_token=credential.device_token)
    except PlatformError as exc:
        # Unreachable is not the same as refused, and must not read as one: a
        # researcher on a train has not been revoked.
        errors.print(f"[red]{exc}[/red]")
        errors.print("This machine's credential could not be checked, so it was left alone.")
        return ui.EXIT_FAILED

    if identity is None:
        return _report_rejected(errors, credential)

    console.print("[green]This machine is already paired.[/green]")
    console.print()
    table = ui.detail_table()
    table.add_row("Account", ui.value(identity.account_email))
    table.add_row("Device", ui.value(identity.name))
    table.add_row("Device id", ui.value(identity.device_id))
    table.add_row("Platform", ui.value(base_url))
    table.add_row("Credential", ui.value(_credential_summary()))
    if identity.token_expires_at is not None:
        table.add_row("Token expires", ui.value(_format_timestamp(identity.token_expires_at)))
    console.print(table)
    console.print()
    console.print("Pass [bold]--force[/bold] to pair this machine a second time.")
    return ui.EXIT_OK


def _report_rejected(errors, credential: DeviceCredential) -> int:
    """The platform refused the stored credential. Say which cause is likely.

    Every rejection (unknown, expired, revoked) comes back as the same 401
    with the same public message, so the platform can't be asked which one
    it was. The stored expiry is the one fact available locally, and it
    separates the two cases a researcher would act on differently.
    """
    if credential.is_expired():
        errors.print("[red]This machine's device token has expired.[/red]")
        errors.print(
            f"It ran out on {_format_timestamp(credential.token_expires_at)}. "
            "Run `nomikos pair --force` to pair again."
        )
        return ui.EXIT_FAILED

    errors.print("[red]This machine's access has been removed.[/red]")
    errors.print(
        f"{credential.platform_url} no longer accepts its device token, and the token "
        "has not expired - so it was revoked from the account."
    )
    errors.print()
    errors.print(
        "Nothing here can undo that: revocation is a decision made on the account, "
        "and only pairing again reverses it. Run [bold]nomikos pair --force[/bold] "
        "if that is what you want."
    )
    return ui.EXIT_FAILED


# ---------------------------------------------------------------------------
# The pairing itself
# ---------------------------------------------------------------------------
def _pair(console, errors, client: PlatformClient, args: argparse.Namespace) -> int:
    # Truncated after the choice, not inside `this_machine_name()`: the
    # platform caps the field on the way in, and a `--name` over the cap is
    # the same 422 the cap exists to keep a long hostname from producing.
    # Either source can be too long, so the limit belongs where the name is
    # settled.
    device_name = ((args.name or this_machine_name()).strip() or this_machine_name())[
        :DEVICE_NAME_LIMIT
    ]
    started = client.start_pairing(
        device_name=device_name,
        device_platform=this_machine_platform(),
        agent_version=installed_version(),
        capabilities=CAPABILITIES,
    )

    _announce(console, started, device_name, client.base_url)
    _maybe_open_browser(console, started, no_browser=args.no_browser)

    poll = _wait_for_approval(console, errors, client, started)
    if poll is None:
        return ui.EXIT_FAILED

    if not poll.device_token or not poll.device_id:
        errors.print("[red]The platform approved the pairing but sent no device token.[/red]")
        return ui.EXIT_FAILED

    credential = DeviceCredential(
        platform_url=client.base_url,
        device_id=str(poll.device_id),
        device_token=poll.device_token,
        account_email=poll.account_email or "",
        device_name=device_name,
        token_expires_at=poll.token_expires_at,
        paired_at=datetime.now(UTC),
    )
    path = save_credential(credential)

    console.print()
    console.print("[green]Paired.[/green]")
    console.print()
    table = ui.detail_table()
    table.add_row("Account", ui.value(credential.account_email or "unknown"))
    table.add_row("Device", ui.value(device_name))
    table.add_row("Device id", ui.value(credential.device_id))
    table.add_row("Credential", ui.value(f"{path} ({_mode_summary(file_mode(path))})"))
    if credential.token_expires_at is not None:
        table.add_row("Token expires", ui.value(_format_timestamp(credential.token_expires_at)))
    console.print(table)
    console.print()
    console.print(
        "Run [bold]nomikos run[/bold] to start taking work, or remove this machine "
        "from your account settings to revoke it."
    )
    return ui.EXIT_OK


def _announce(console, started: StartedPairing, device_name: str, base_url: str) -> None:
    """Print the URL, then the code. Nothing may come between them and the wait."""
    console.print(f"Pairing [bold]{device_name}[/bold] with {base_url}")
    console.print()
    console.print("Open this link and approve the request:")
    console.print()
    ui.unbroken(console, f"  [bold cyan]{started.verification_url}[/bold cyan]")
    console.print()
    console.print("Confirmation code:")
    console.print()
    console.print(f"  [bold]{started.confirmation_code}[/bold]")
    console.print()
    console.print(
        "The page must show this exact code. If it shows a different one, close it "
        "and approve nothing: the request came from somewhere else, not from here."
    )
    console.print()


def _maybe_open_browser(console, started: StartedPairing, *, no_browser: bool) -> None:
    """A convenience on top of the printed URL, never a replacement for it."""
    if no_browser:
        return
    if _looks_like_ssh():
        # Over SSH a browser either doesn't open or opens on the wrong machine.
        console.print("[dim]Not opening a browser: this looks like an SSH session.[/dim]")
        return
    try:
        opened = webbrowser.open(started.verification_url)
    except Exception:  # pragma: no cover - platform browser launchers vary wildly
        opened = False
    if opened:
        console.print("[dim]Opening your browser...[/dim]")
    else:
        console.print("[dim]No browser could be opened. Use the link above.[/dim]")


def _looks_like_ssh() -> bool:
    return bool(os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY"))


def _wait_for_approval(console, errors, client: PlatformClient, started: StartedPairing):
    """Poll until the browser decides, the request dies, or its lifetime runs out.

    The cadence comes from the platform on every response, not from a
    constant here: `DEVICE_PAIRING_POLL_INTERVAL_SECONDS` is an operational
    dial, and `slow_down` returns a doubled interval that this must honour or
    the pairing row starts burning attempts.
    """
    interval = max(started.interval_seconds, _MINIMUM_POLL_SECONDS)
    deadline = time.monotonic() + started.expires_in + _POLL_MARGIN_SECONDS

    with console.status("Waiting for approval in the browser..."):
        while True:
            time.sleep(interval)
            poll = client.collect_token(
                pairing_id=started.pairing_id, device_code=started.device_code
            )
            interval = max(poll.interval_seconds or interval, _MINIMUM_POLL_SECONDS)

            if poll.status == STATUS_APPROVED:
                return poll
            if poll.status == STATUS_ACCESS_DENIED:
                errors.print("[red]The request was refused.[/red]")
                errors.print(
                    "Either it was denied in the browser, or this pairing code has "
                    "already been used. Run `nomikos pair` to start a new one."
                )
                return None
            if poll.status == STATUS_EXPIRED:
                errors.print("[red]The pairing request expired before it was approved.[/red]")
                errors.print("Run `nomikos pair` to start a new one.")
                return None
            if poll.status not in (STATUS_AUTHORIZATION_PENDING, STATUS_SLOW_DOWN):
                errors.print(
                    f"[red]The platform answered an unknown state: {escape(repr(poll.status))}[/red]"
                )
                return None
            if time.monotonic() >= deadline:
                errors.print("[red]Gave up waiting for approval.[/red]")
                errors.print("Run `nomikos pair` to start a new one.")
                return None


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def _format_timestamp(moment: datetime | None) -> str:
    if moment is None:
        return "unknown"
    return moment.astimezone().strftime("%Y-%m-%d %H:%M %Z")


def _mode_summary(mode: int) -> str:
    return f"{mode:04o}, owner-only" if mode == 0o600 else f"{mode:04o}"


def _credential_summary() -> str:
    path = credential_path()
    try:
        return f"{path} ({_mode_summary(file_mode(path))})"
    except OSError:  # pragma: no cover - the file was just read successfully
        return str(path)
