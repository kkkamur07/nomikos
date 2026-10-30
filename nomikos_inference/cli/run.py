"""`nomikos run` - the **claim** loop: take a page, run it here, report it, repeat.

The loop itself lives in `claim.py`; this module is the entry point (argument
parsing, setup, and the launch announcement). See that module for why the loop
is shaped the way it is.
"""

from __future__ import annotations

import argparse
import time as time  # noqa: PLC0414 - re-exported: tests patch the loop's clock through here

from nomikos_inference.cli import console as ui
from nomikos_inference.cli.api import (
    AgentVersionRefused,
    PlatformClient,
    PlatformError,
    default_platform_url,
)

# Re-exported so existing importers keep their paths: the loop and its tuning
# live in `claim.py`, the identity in `identity.py`, the printing in
# `run_report.py`.
from nomikos_inference.cli.claim import CLAIM_RETRY_CAP_SECONDS as CLAIM_RETRY_CAP_SECONDS
from nomikos_inference.cli.claim import CLAIM_RETRY_SECONDS as CLAIM_RETRY_SECONDS
from nomikos_inference.cli.claim import INTERRUPTED_ERROR as INTERRUPTED_ERROR
from nomikos_inference.cli.claim import MAX_CLAIM_FAILURES as MAX_CLAIM_FAILURES
from nomikos_inference.cli.claim import MAX_REPORT_ATTEMPTS as MAX_REPORT_ATTEMPTS
from nomikos_inference.cli.claim import REPORT_RETRY_CAP_SECONDS as REPORT_RETRY_CAP_SECONDS
from nomikos_inference.cli.claim import REPORT_RETRY_SECONDS as REPORT_RETRY_SECONDS
from nomikos_inference.cli.claim import PageOutcome as PageOutcome
from nomikos_inference.cli.claim import PageOutputError as PageOutputError
from nomikos_inference.cli.claim import _claim_loop as _claim_loop
from nomikos_inference.cli.claim import _execute as _execute
from nomikos_inference.cli.claim import _handle_page as _handle_page
from nomikos_inference.cli.claim import _job_output as _job_output
from nomikos_inference.cli.claim import _report as _report
from nomikos_inference.cli.identity import LAPTOP_WAIT_SECONDS, WORKER_WAIT_SECONDS
from nomikos_inference.cli.identity import SERVICE_TOKEN_ENV as SERVICE_TOKEN_ENV
from nomikos_inference.cli.identity import WORKER_NAME_ENV as WORKER_NAME_ENV
from nomikos_inference.cli.identity import AgentIdentity as AgentIdentity
from nomikos_inference.cli.identity import RunSetupError as RunSetupError
from nomikos_inference.cli.identity import _resolve_identity as _resolve_identity
from nomikos_inference.cli.run_report import MAX_REASON_CHARS as MAX_REASON_CHARS
from nomikos_inference.cli.run_report import _reason as _reason
from nomikos_inference.cli.run_report import _report_outdated as _report_outdated
from nomikos_inference.cli.run_report import _report_refusal as _report_refusal
from nomikos_inference.cli.run_report import _summarise as _summarise
from nomikos_inference.cli.version import installed_version


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--api-url",
        metavar="URL",
        help="Platform to claim from. Defaults to $NOMIKOS_API_URL, then the hosted API.",
    )
    parser.add_argument(
        "--exit-when-empty",
        action="store_true",
        help=(
            "Stop once the queue has nothing left, instead of waiting for more. "
            "The scripted half of this command; without it, it keeps running."
        ),
    )
    parser.add_argument(
        "--wait-seconds",
        type=int,
        metavar="SECONDS",
        help=(
            "How long one claim waits for work before coming back empty. "
            f"Defaults to {LAPTOP_WAIT_SECONDS} for a paired machine and "
            f"{WORKER_WAIT_SECONDS} for a hosted worker. Clamped by the platform."
        ),
    )


def run(args: argparse.Namespace) -> int:
    console = ui.out()
    errors = ui.err()

    base_url = (args.api_url or default_platform_url()).rstrip("/")
    try:
        client = PlatformClient(base_url)
        identity = _resolve_identity(base_url)
    except (RunSetupError, PlatformError) as exc:
        errors.print(f"[red]{exc}[/red]")
        return ui.EXIT_FAILED

    wait_seconds = (
        identity.default_wait_seconds if args.wait_seconds is None else max(0, args.wait_seconds)
    )
    _announce(console, identity, base_url)

    try:
        return _claim_loop(
            console,
            errors,
            client,
            identity,
            wait_seconds=wait_seconds,
            exit_when_empty=args.exit_when_empty,
        )
    except AgentVersionRefused as exc:
        _report_refusal(errors, exc)
        return ui.EXIT_FAILED
    except PlatformError as exc:
        errors.print(f"[red]{exc}[/red]")
        return ui.EXIT_FAILED


def _announce(console, identity: AgentIdentity, base_url: str) -> None:
    console.print(f"[bold]nomikos run[/bold] {installed_version()}")
    console.print()
    table = ui.detail_table()
    table.add_row("Platform", ui.value(base_url))
    table.add_row("Claiming", ui.value(f"{identity.execution_target} work for {identity.account}"))
    console.print(table)
    console.print()
