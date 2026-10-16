"""`nomikos pair` and `nomikos version`, run as the installed console script.

Everything here is live. The CLI is the real `nomikos` executable from a real
wheel in its own virtual environment; the platform is a real uvicorn process
serving the real `create_app()`; the database is real Postgres, migrated by
alembic. Nothing is patched, faked, or substituted - the approval that unblocks
each pairing is an HTTP request from a second process, exactly as a browser
would make it.

That matters more here than it usually would. Every claim this issue makes is
about behaviour *between* processes: that the URL is printed before a browser is
launched, that the code the terminal shows is the code the consent screen shows,
that the credential lands on disk readable by nobody else. A test that imported
`nomikos_inference.cli.pair` and called it could not observe any of them.

The scaffolding all three CLI integration modules share - Postgres, alembic,
uvicorn, the wheel build, the hand-rolled HTTP client - is in
`tests/inference/integration/conftest.py`. This module keeps its own database
(`kalamos_056_cli`) rather than the one `tests/nomikos/integration/conftest.py`
truncates between tests, so a server held open across it cannot have the ground
moved under it.
"""

from __future__ import annotations

import json
import stat
import subprocess
import time
import uuid
from pathlib import Path

import pytest

from tests.inference.integration.conftest import (
    APP_ORIGIN,
    PAIRING_TIMEOUT_SECONDS,
    await_line,
    build_and_install_cli,
    cli_environment,
    decide_pairing,
    http_request,
    migrate_database,
    register_account,
    serve_platform,
)

pytestmark = pytest.mark.integration

DATABASE = "kalamos_056_cli"

# The real cadence would make this module spend five seconds per poll waiting for
# an approval that has already happened. The cadence itself is the platform's
# and is covered by its own tests.
POLL_INTERVAL_SECONDS = "1"

#: A developer's own SSH session, browser, or API URL must not decide what these
#: tests observe. `PYTHONPATH` is deliberately *not* dropped: this module's wheel
#: is installed with `--no-deps`, and what it can import is `test_published_package.py`'s
#: question rather than this one's.
INHERITED_TO_DROP = ("SSH_CONNECTION", "SSH_TTY", "BROWSER", "NOMIKOS_API_URL")


# ---------------------------------------------------------------------------
# The platform: real app, real Postgres, real HTTP
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def migrated_database() -> str:
    return migrate_database(DATABASE)


@pytest.fixture(scope="session")
def platform_url(migrated_database: str, tmp_path_factory: pytest.TempPathFactory) -> str:
    """A uvicorn process serving `create_app()`, torn down at the end."""
    log_path = tmp_path_factory.mktemp("platform") / "server.log"
    yield from serve_platform(
        migrated_database,
        log_path,
        DEVICE_PAIRING_POLL_INTERVAL_SECONDS=POLL_INTERVAL_SECONDS,
        # Every test in this module pairs from 127.0.0.1, so they all charge one
        # per-client bucket. At the default of 10 the module exhausts it partway
        # through and the remaining tests fail on a 429 that has nothing to do
        # with what they assert.
        DEVICE_PAIRING_RATE_LIMIT_REQUESTS="1000",
    )


# ---------------------------------------------------------------------------
# The CLI: a real wheel, a real console script
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def installed_cli(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """Build the wheel and install the console script into an empty environment.

    `--no-deps` on purpose. Downloading the published closure would turn a CLI
    test into a multi-gigabyte one; the pairing path imports nothing from it.
    `rich` is installed explicitly because the CLI does import that, and an entry
    point that cannot start proves nothing. That the closure resolves at all is
    `test_published_package.py`'s question, not this one's.
    """
    return build_and_install_cli(
        tmp_path_factory, install_sets=(["--no-deps", "{wheel}"], ["rich"])
    )


def _wheel_version(wheel: Path) -> str:
    # `nomikos_inference-0.3.1-py3-none-any.whl`
    return wheel.name.split("-")[1]


@pytest.fixture
def home(tmp_path: Path) -> Path:
    """Where this test's CLI keeps its credential. Never the real one."""
    return tmp_path / "nomikos-home"


def _cli_environment(home: Path, *, extra: dict[str, str] | None = None) -> dict[str, str]:
    return cli_environment(home, drop=INHERITED_TO_DROP, extra=extra)


# ---------------------------------------------------------------------------
# Acting as the researcher's browser
# ---------------------------------------------------------------------------
def _get(url: str, headers: dict[str, str] | None = None) -> tuple[int, object]:
    return http_request("GET", url, None, headers, timeout=15)


def _delete(url: str, headers: dict[str, str]) -> int:
    return http_request("DELETE", url, None, headers, timeout=15)[0]


@pytest.fixture
def researcher(platform_url: str) -> dict[str, str]:
    email, headers = register_account(platform_url, "cli")
    return {"email": email, "headers": headers}


# ---------------------------------------------------------------------------
# Driving `nomikos pair` while approving it from outside
# ---------------------------------------------------------------------------
class PairRun:
    """One `nomikos pair` invocation and everything observable about it."""

    def __init__(
        self,
        returncode: int,
        stdout: str,
        stderr: str,
        verification_url: str | None,
        consent_screen: dict | None,
    ):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.verification_url = verification_url
        self.consent_screen = consent_screen
        """What `/pair` was served for this request, captured before approving.

        It has to be read then: approval consumes the pairing row, and looking
        it up afterwards is a 404 - single-use is the point of ADR 0001's
        decision 7.
        """

    @property
    def output(self) -> str:
        return f"{self.stdout}\n{self.stderr}"

    @property
    def confirmation_code(self) -> str | None:
        """The code as the terminal showed it, read back off the terminal."""
        lines = [line.strip() for line in self.stdout.splitlines()]
        for index, line in enumerate(lines):
            if line.startswith("Confirmation code"):
                for candidate in lines[index + 1 :]:
                    if candidate:
                        return candidate
        return None

    @property
    def verification_token(self) -> str:
        assert self.verification_url, "no pairing URL was printed"
        return self.verification_url.split("#", 1)[1]


def _run_pair(
    installed_cli: dict[str, object],
    platform_url: str,
    home: Path,
    *,
    arguments: tuple[str, ...] = (),
    environment: dict[str, str] | None = None,
    approve_as: dict[str, str] | None = None,
    deny_as: dict[str, str] | None = None,
) -> PairRun:
    """Run the real console script, and answer its consent request out of band.

    Output goes to files rather than pipes so this can read what the CLI has
    printed *while it is still running* without any chance of deadlocking on a
    full pipe buffer - the pairing URL only becomes readable partway through.
    """
    stdout_path = home.parent / f"pair-stdout-{uuid.uuid4().hex[:6]}.txt"
    stderr_path = home.parent / f"pair-stderr-{uuid.uuid4().hex[:6]}.txt"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)

    with stdout_path.open("w") as out_file, stderr_path.open("w") as err_file:
        process = subprocess.Popen(
            [
                str(installed_cli["executable"]),
                "pair",
                "--api-url",
                platform_url,
                *arguments,
            ],
            env=_cli_environment(home, extra=environment),
            cwd=str(installed_cli["workspace"]),
            stdout=out_file,
            stderr=err_file,
            text=True,
        )
        verification_url = None
        consent_screen = None
        try:
            if approve_as is not None or deny_as is not None:
                verification_url = await_line(process, stdout_path, APP_ORIGIN)
                if verification_url is not None:
                    consent_screen = decide_pairing(
                        platform_url,
                        verification_url,
                        headers=(approve_as or deny_as)["headers"],
                        approve=approve_as is not None,
                    )
            process.wait(timeout=PAIRING_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:  # pragma: no cover - a hung CLI is the failure
            process.kill()
            process.wait()
            raise AssertionError(
                f"`nomikos pair` did not finish.\n{stdout_path.read_text()}\n"
                f"{stderr_path.read_text()}"
            ) from None

    return PairRun(
        returncode=process.returncode,
        stdout=stdout_path.read_text(),
        stderr=stderr_path.read_text(),
        verification_url=verification_url,
        consent_screen=consent_screen,
    )


# ---------------------------------------------------------------------------
# The pairing URL, and when the browser is allowed to be involved
# ---------------------------------------------------------------------------
def test_the_pairing_url_is_printed_before_any_browser_is_opened(
    installed_cli, platform_url, home, researcher, tmp_path
) -> None:
    """ADR 0002: the browser is a convenience layered on a printed URL.

    Proved by making `$BROWSER` a script that writes a marker onto the CLI's own
    stdout. Both writes land in one file in the order they happened, so the
    assertion is about real sequencing between two processes rather than about
    which function this module believes is called first.
    """
    stub = tmp_path / "browser-stub.sh"
    stub.write_text('#!/bin/sh\necho "BROWSER-OPENED $1"\n')
    stub.chmod(0o755)

    run = _run_pair(
        installed_cli,
        platform_url,
        home,
        environment={"BROWSER": str(stub)},
        approve_as=researcher,
    )
    assert run.returncode == 0, run.output

    lines = run.stdout.splitlines()
    url_line = next(index for index, line in enumerate(lines) if APP_ORIGIN in line)
    browser_line = next(index for index, line in enumerate(lines) if "BROWSER-OPENED" in line)

    assert url_line < browser_line, run.stdout
    # And what it was handed is the same URL the researcher was shown.
    assert lines[url_line].strip() in lines[browser_line]


def test_pairing_completes_over_ssh_with_no_browser_available(
    installed_cli, platform_url, home, researcher, tmp_path
) -> None:
    """`webbrowser.open()` over SSH opens nothing, or opens it on the wrong machine.

    The stub would record either. It records nothing, and the pairing still
    completes on the printed URL alone.
    """
    stub = tmp_path / "browser-stub.sh"
    record = tmp_path / "opened.txt"
    stub.write_text(f'#!/bin/sh\necho "$1" >> {record}\n')
    stub.chmod(0o755)

    run = _run_pair(
        installed_cli,
        platform_url,
        home,
        environment={
            "BROWSER": str(stub),
            "SSH_CONNECTION": "10.0.0.2 51000 10.0.0.9 22",
            "SSH_TTY": "/dev/pts/3",
        },
        approve_as=researcher,
    )

    assert run.returncode == 0, run.output
    assert not record.exists(), f"a browser was launched over SSH: {record.read_text()}"
    assert "SSH session" in run.stdout
    assert APP_ORIGIN in run.stdout
    assert "Paired." in run.stdout


def test_the_confirmation_code_in_the_terminal_matches_the_consent_screen(
    installed_cli, platform_url, home, researcher
) -> None:
    """ADR 0001 decision 13, finally made real on the client side.

    The consent screen's every other field is supplied by whoever started the
    pairing. This code is not, and comparing the two is the only check a
    researcher can make - so the two strings have to be the same string.
    """
    run = _run_pair(installed_cli, platform_url, home, approve_as=researcher)
    assert run.returncode == 0, run.output

    printed = run.confirmation_code
    assert printed, run.stdout
    assert run.consent_screen is not None

    assert printed == run.consent_screen["confirmation_code"]
    # It is shown before the wait begins, not after approval - a code that
    # arrives once the researcher has already clicked checks nothing.
    assert run.stdout.index(printed) < run.stdout.index("Paired.")


# ---------------------------------------------------------------------------
# The credential on disk
# ---------------------------------------------------------------------------
def test_the_credential_is_written_with_owner_only_permissions(
    installed_cli, platform_url, home, researcher
) -> None:
    """ADR 0001: `0600` in a `0700` directory, so another account cannot claim as them."""
    run = _run_pair(installed_cli, platform_url, home, approve_as=researcher)
    assert run.returncode == 0, run.output

    credential = home / "device.json"
    assert credential.is_file(), run.output
    assert stat.S_IMODE(credential.stat().st_mode) == 0o600
    assert stat.S_IMODE(home.stat().st_mode) == 0o700

    stored = json.loads(credential.read_text())
    assert stored["platform_url"] == platform_url
    assert stored["account_email"] == researcher["email"]
    assert stored["device_token"].startswith("nmd1.")
    # The credential is the only place the token exists. It is never printed.
    assert stored["device_token"] not in run.output


# `test_the_stored_token_actually_authenticates_this_machine` stood here and redeemed the
# stored token against `/device/v1/self`. Mint-redeem-then-authenticate is
# `tests/nomikos/integration/test_device_pairing.py::test_mint_redeem_then_authenticate`,
# and that the token in `device.json` is the real one is implied by every `pair`-then-`run`
# test in `test_cli_run.py`, which claims real pages with it.


# ---------------------------------------------------------------------------
# The two states that end without a new device row
# ---------------------------------------------------------------------------
def test_pairing_an_already_paired_machine_reports_it_and_creates_no_second_device(
    installed_cli, platform_url, home, researcher
) -> None:
    first = _run_pair(installed_cli, platform_url, home, approve_as=researcher)
    assert first.returncode == 0, first.output

    # No approval offered: if this started a pairing it would sit there polling.
    second = _run_pair(installed_cli, platform_url, home)

    assert second.returncode == 0, second.output
    assert "already paired" in second.stdout.lower()
    assert APP_ORIGIN not in second.output, "a second pairing request was started"

    status, devices = _get(f"{platform_url}/devices", researcher["headers"])
    assert status == 200
    assert len(devices) == 1, devices


def test_a_revoked_device_reports_the_revocation_and_exits_non_zero(
    installed_cli, platform_url, home, researcher
) -> None:
    """Revocation is a decision made on the account (ADR 0001, decision 11).

    Re-pairing over it silently would undo it without telling anyone, and
    polling for a consent nobody is going to give would hang. It does neither.
    """
    paired = _run_pair(installed_cli, platform_url, home, approve_as=researcher)
    assert paired.returncode == 0, paired.output
    device_id = json.loads((home / "device.json").read_text())["device_id"]

    assert _delete(f"{platform_url}/devices/{device_id}", researcher["headers"]) == 204

    started = time.monotonic()
    revoked = _run_pair(installed_cli, platform_url, home)
    elapsed = time.monotonic() - started

    assert revoked.returncode != 0, revoked.output
    assert "removed" in revoked.stderr.lower()
    assert "--force" in revoked.stderr
    # It reported and stopped. Nothing was started that could be waited on.
    assert APP_ORIGIN not in revoked.output
    assert elapsed < 20, f"it spun for {elapsed:.1f}s instead of reporting"


def test_a_denied_pairing_exits_non_zero_and_writes_nothing(
    installed_cli, platform_url, home, researcher
) -> None:
    run = _run_pair(installed_cli, platform_url, home, deny_as=researcher)

    assert run.returncode != 0, run.output
    assert "refused" in run.stderr.lower()
    assert not (home / "device.json").exists()


# ---------------------------------------------------------------------------
# `nomikos version`
# ---------------------------------------------------------------------------
def test_the_version_subcommand_reports_the_installed_package_version(installed_cli, home) -> None:
    """Read from installed metadata, so it is the version that is really on disk."""
    completed = subprocess.run(
        [str(installed_cli["executable"]), "version"],
        env=_cli_environment(home),
        cwd=str(installed_cli["workspace"]),
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    expected = _wheel_version(installed_cli["wheel"])
    assert f"nomikos-inference {expected}" in completed.stdout
    # Not a source checkout: the whole point of reading metadata.
    assert "0+unknown" not in completed.stdout


# Two more `version` tests stood here. One asserted the help text names
# `X-Nomikos-Agent-Version`; the header actually sent is `AGENT_VERSION_HEADER` in
# `nomikos_inference/cli/api.py`, which that test never touched. The other reported whether this
# machine is paired, and its one load-bearing line - a version report is never a way to
# read the credential out - is already asserted against `run.output` by
# `test_the_credential_is_written_with_owner_only_permissions` above.
