"""`nomikos upgrade` - the launch check - against a real index and a real platform.

Everything here is live, and the package index is the part that had to be. The
agent is a real `nomikos` console script installed from a real wheel; the index
it upgrades from is a PEP 503 tree of real wheels served over real HTTP by
`python -m http.server`; the installer is the real `pip` or the real `uv`,
resolving over the network to that index; and the platform is a real uvicorn
process serving the real `create_app()` against real Postgres. The re-exec is a
real `execve`, observed by reading the version off the process that came out the
other side.

That matters because every claim this issue makes is about what happens *between*
processes. "Upgrades, re-execs, and then claims" is three processes and two
network services; a test that patched `subprocess.run` would be asserting that
this module knows what it wrote, and would have proved nothing about whether a
researcher's laptop comes back up on the new build.

Two things about the wheels are deliberately unlike production, and neither is a
substitution for anything under test. They are built from the repository's own
`nomikos_inference/` tree with a generated `pyproject.toml`, because two versions of one
package cannot both be the checked-in version - the *code* in them is this
repository's. And they declare no dependencies, so the resolver never leaves the
local index; whether the published closure resolves is
`test_published_package.py`'s question, and answering it here would download the
whole thing twice per test.

The scaffolding all three CLI integration modules share - Postgres, alembic,
uvicorn, the hand-rolled HTTP client - is in
`tests/inference/integration/conftest.py`. This module keeps its own database
(`kalamos_058_upgrade`) for the same reason `test_cli_pairing.py` does: servers
are held open across it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.fixtures.paths import REPO_ROOT
from tests.inference.integration.conftest import (
    free_port,
    http_request,
    migrate_database,
    require_uv,
    start_platform,
    stop_platform,
    wait_for_http,
)

pytestmark = pytest.mark.integration

DATABASE = "kalamos_058_upgrade"

CLI_TIMEOUT_SECONDS = 300.0

PACKAGE = "nomikos-inference"

# The two builds on the index. `0.9.0` beats `0.1.0` by arithmetic and not by
# alphabet, which is the ordering the platform's floor is careful about.
AGENT_OLD = "0.1.0"
AGENT_NEW = "0.9.0"
# Above everything the index has, so a floor set here is one no upgrade can
# reach - which is how the failure path is provoked without breaking the index.
BEYOND_INDEX = "1.5.0"

_PYPROJECT = """\
[project]
name = "nomikos-inference"
version = "{version}"
description = "Agent build {version}, for the self-upgrade tests"
requires-python = ">=3.11,<3.13"
dependencies = []

[project.scripts]
nomikos = "nomikos_inference.cli:main"

[build-system]
requires = ["hatchling>=1.27"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["nomikos_inference"]
"""


# ---------------------------------------------------------------------------
# A real package index: two wheels, PEP 503 layout, served over HTTP
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def local_index(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Build both agent builds and serve them as a simple index.

    Session-scoped because it is two wheel builds and one HTTP server, and every
    test in the module resolves against the same index.
    """
    uv = require_uv()
    workspace = tmp_path_factory.mktemp("index")
    source = workspace / "package"
    shutil.copytree(
        REPO_ROOT / "nomikos_inference",
        source / "nomikos_inference",
        # The loopback HTTP surfaces are excluded from the published wheel too,
        # and nothing in the CLI imports them.
        ignore=shutil.ignore_patterns("api", "helper", "__pycache__", "*.pyc"),
    )

    root = workspace / "index"
    project = root / "simple" / PACKAGE
    project.mkdir(parents=True)

    for version in (AGENT_OLD, AGENT_NEW):
        (source / "pyproject.toml").write_text(_PYPROJECT.format(version=version))
        built = subprocess.run(
            [uv, "build", "--wheel", "-o", str(project), str(source)],
            capture_output=True,
            text=True,
        )
        assert built.returncode == 0, built.stderr

    wheels = sorted(project.glob("*.whl"))
    assert len(wheels) == 2, wheels
    links = "\n".join(f'<a href="{wheel.name}">{wheel.name}</a><br/>' for wheel in wheels)
    (project / "index.html").write_text(f"<!DOCTYPE html><html><body>{links}</body></html>\n")
    (root / "simple" / "index.html").write_text(
        f'<!DOCTYPE html><html><body><a href="{PACKAGE}/">{PACKAGE}</a></body></html>\n'
    )

    port = free_port()
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "http.server",
            str(port),
            "--bind",
            "127.0.0.1",
            "--directory",
            str(root),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    index_url = f"http://127.0.0.1:{port}/simple/"
    try:
        wait_for_http(server, f"{index_url}{PACKAGE}/", what="the local package index")
        yield index_url
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover
            server.kill()


# ---------------------------------------------------------------------------
# The platform: real app, real Postgres, one server per version policy
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def migrated_database() -> str:
    return migrate_database(DATABASE)


@dataclass
class Platform:
    """One running platform, and where its server log went.

    `log_path` used to be read back: two tests counted requests to the version
    floor off the uvicorn access log to pin a call count. They were implementation
    detail and are gone, and the servers dropped back to `--log-level warning`
    with them. The path is kept because a failure in this module is usually
    something the platform said, and knowing where it said it is worth one field.
    """

    base_url: str
    log_path: Path


@pytest.fixture(scope="session")
def platform_at(migrated_database: str, tmp_path_factory: pytest.TempPathFactory):
    """Start (and reuse) a platform with a given version floor and latest release.

    One server per policy rather than one server reconfigured, because
    `DeviceSettings` is cached for the life of the process - which is the point
    of `@settings_cache`, and not something a test should be reaching around.
    """
    servers: dict[tuple[str, str], tuple[subprocess.Popen, Platform]] = {}
    logs = tmp_path_factory.mktemp("platforms")

    def start(*, minimum: str, latest: str) -> Platform:
        key = (minimum, latest)
        if key in servers:
            return servers[key][1]

        log_path = logs / f"server-{minimum}-{latest}.log"
        server, base_url = start_platform(
            migrated_database,
            log_path,
            what=f"the platform (floor {minimum})",
            INFERENCE_AGENT_MIN_VERSION=minimum,
            INFERENCE_AGENT_LATEST_VERSION=latest,
        )
        platform = Platform(base_url=base_url, log_path=log_path)
        servers[key] = (server, platform)
        return platform

    try:
        yield start
    finally:
        for server, _ in servers.values():
            stop_platform(server)


# ---------------------------------------------------------------------------
# The agent: a real console script in its own environment
# ---------------------------------------------------------------------------
@dataclass
class Agent:
    """One installed agent, and how to run it."""

    executable: Path
    python: Path
    home: Path
    workspace: Path
    environment: dict[str, str]

    def run(
        self, *arguments: str, extra: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess:
        environment = dict(self.environment)
        environment.update(extra or {})
        return subprocess.run(
            [str(self.executable), *arguments],
            env=environment,
            cwd=str(self.workspace),
            capture_output=True,
            text=True,
            timeout=CLI_TIMEOUT_SECONDS,
        )

    def installed_version(self) -> str:
        """What is on disk now, read the way the floor reads it."""
        completed = subprocess.run(
            [
                str(self.python),
                "-c",
                "from importlib.metadata import version; print(version('nomikos-inference'))",
            ],
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        return completed.stdout.strip()


@pytest.fixture
def agent_at(local_index: str, tmp_path: Path):
    """Install one agent build into an empty environment, and hand it back.

    `with_pip` picks which installer the agent will find. Both branches are real
    install paths: `uv tool install` leaves no pip in the environment at all, and
    `pip install` leaves one. The CLI has to bring itself forward either way.
    """
    uv = require_uv()
    counter = [0]

    def install(version: str, *, with_pip: bool = False) -> Agent:
        counter[0] += 1
        workspace = tmp_path / f"agent-{counter[0]}"
        venv = workspace / "venv"
        home = workspace / "nomikos-home"
        workspace.mkdir(parents=True)

        create = [
            uv,
            "venv",
            str(venv),
            "--python",
            f"{sys.version_info.major}.{sys.version_info.minor}",
        ]
        if with_pip:
            create.append("--seed")
        subprocess.run(create, check=True, capture_output=True, text=True)

        scripts = venv / ("Scripts" if os.name == "nt" else "bin")
        python = scripts / "python"
        installed = subprocess.run(
            [
                uv,
                "pip",
                "install",
                "--python",
                str(python),
                "--index-url",
                local_index,
                f"{PACKAGE}=={version}",
            ],
            capture_output=True,
            text=True,
        )
        assert installed.returncode == 0, installed.stderr
        # `rich` is a real dependency of the real package; the test wheels drop
        # their dependency list so the upgrade resolves entirely against the
        # local index, so it is installed here instead.
        subprocess.run(
            [uv, "pip", "install", "--python", str(python), "rich"],
            check=True,
            capture_output=True,
            text=True,
        )

        environment = dict(os.environ)
        for inherited in ("NOMIKOS_API_URL", "NOMIKOS_UPGRADED_FROM", "VIRTUAL_ENV"):
            environment.pop(inherited, None)
        environment.update(
            {
                "NOMIKOS_HOME": str(home),
                "PYTHONUNBUFFERED": "1",
                # The index the *installer* reaches, exactly as a researcher's
                # own configuration would reach it. The CLI has no index flag:
                # it upgrades with whatever installer already owns the
                # environment, from wherever that installer is pointed.
                "UV_INDEX_URL": local_index,
                "PIP_INDEX_URL": local_index,
                "PIP_TRUSTED_HOST": "127.0.0.1",
                "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                # Rich wraps at the console width, and with no terminal that is
                # 80 columns - narrow enough to break a sentence this module
                # asserts on across two lines. A width, not a mock: the CLI
                # would read the same value from a real terminal.
                "COLUMNS": "200",
            }
        )

        executable = scripts / ("nomikos.exe" if os.name == "nt" else "nomikos")
        assert executable.is_file(), "the wheel did not install a `nomikos` console script"
        return Agent(
            executable=executable,
            python=python,
            home=home,
            workspace=workspace,
            environment=environment,
        )

    return install


# ---------------------------------------------------------------------------
# A real device credential, so "and then claims" can mean a real claim
# ---------------------------------------------------------------------------
def _post(url: str, body: dict, headers: dict[str, str] | None = None) -> tuple[int, dict]:
    return http_request("POST", url, body, headers)


def pair_a_device(platform: Platform, *, agent_version: str) -> str:
    """Run the pairing protocol over HTTP and return the **device token**.

    The CLI's own `pair` is exercised in `test_cli_pairing.py`; what this module
    needs is a credential that really authenticates, so that the claim it makes
    after an upgrade is a real claim rather than an unauthenticated probe.
    """
    suffix = uuid.uuid4().hex[:8]
    status, account = _post(
        f"{platform.base_url}/auth/register",
        {
            "email": f"upgrade-{suffix}@test.kalamos",
            "username": f"upgrade_{suffix}",
            "password": "test-pass-123",
        },
    )
    assert status == 201, account
    headers = {"Authorization": f"Bearer {account['access_token']}"}

    status, started = _post(
        f"{platform.base_url}/device/v1/pairings",
        {
            "device_name": f"agent-{suffix}",
            "platform": "test-arm64",
            "helper_version": agent_version,
            "capabilities": {"runtime": "torch"},
        },
    )
    assert status == 201, started

    token = started["verification_url"].split("#", 1)[1]
    status, looked_up = _post(
        f"{platform.base_url}/devices/pairings/lookup", {"verification_token": token}, headers
    )
    assert status == 200, looked_up
    status, approved = _post(
        f"{platform.base_url}/devices/pairings/{looked_up['pairing_id']}/approve",
        {"verification_token": token},
        headers,
    )
    assert status in (200, 204), approved

    status, collected = _post(
        f"{platform.base_url}/device/v1/pairings/token",
        {"pairing_id": started["pairing_id"], "device_code": started["device_code"]},
    )
    assert status == 200 and collected.get("device_token"), collected
    return collected["device_token"]


def claim(platform: Platform, *, device_token: str, agent_version: str) -> tuple[int, dict]:
    """One real **claim**, stating a version. The queue is empty, so it is a 200
    with no page - which is the normal state of a healthy platform and is still
    a claim the platform agreed to serve."""
    return _post(
        f"{platform.base_url}/device/v1/jobs/claim",
        {"wait_seconds": 0},
        {
            "X-Nomikos-Device-Token": device_token,
            "X-Nomikos-Agent-Version": agent_version,
        },
    )


# ---------------------------------------------------------------------------
# Below the floor: upgrade, re-exec, then claim
# ---------------------------------------------------------------------------
def test_an_agent_below_the_floor_upgrades_re_execs_and_then_claims(
    agent_at, platform_at, local_index
) -> None:
    """The whole point of the issue, end to end across three processes.

    Before: the platform refuses this build a claim, with 426. The CLI upgrades
    itself from a real index, `execve`s into what it installed, and the version
    that comes out the other side is the new one. After: the same credential
    claiming with the new version is served.
    """
    platform = platform_at(minimum="0.5.0", latest=AGENT_NEW)
    agent = agent_at(AGENT_OLD)
    device_token = pair_a_device(platform, agent_version=AGENT_OLD)

    refused, body = claim(platform, device_token=device_token, agent_version=AGENT_OLD)
    assert refused == 426, body
    assert body["error"]["code"] == "AGENT_VERSION_UNSUPPORTED"

    completed = agent.run("upgrade", "--api-url", platform.base_url)

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Upgrading" in completed.stdout, completed.stdout
    assert "Restarting" in completed.stdout, completed.stdout
    assert agent.installed_version() == AGENT_NEW

    served, body = claim(platform, device_token=device_token, agent_version=AGENT_NEW)
    assert served == 200, body
    assert body["agent"]["agent_version"] == AGENT_NEW
    assert body["agent"]["outdated"] is False


# `test_the_upgraded_process_is_the_one_that_carries_on` stood here and ran `--version` as a
# *separate* invocation after an upgrade. That proves what is on disk, which
# `test_an_agent_below_the_floor_upgrades_re_execs_and_then_claims` above already asserts
# with `agent.installed_version() == AGENT_NEW` - and a second process says nothing about
# whether the first one `execve`d or forked.


# `test_an_agent_at_the_current_version_prints_nothing` stood here, asserting stdout and
# stderr are both the empty string for a current build. That is a claim about silence, and
# it cost a wheel install and a platform to make it.


# ---------------------------------------------------------------------------
# Failure is loud, fatal, and claims nothing
# ---------------------------------------------------------------------------
def test_a_failed_upgrade_exits_non_zero_and_claims_nothing(agent_at, platform_at) -> None:
    """A floor no build on the index can reach: the installer really fails.

    The failure mode being engineered against is a researcher watching a
    terminal that looks busy while nothing is being transcribed - so an agent
    that cannot make itself claimable stops, says why, and exits non-zero.
    """
    platform = platform_at(minimum=BEYOND_INDEX, latest=BEYOND_INDEX)
    agent = agent_at(AGENT_OLD)
    device_token = pair_a_device(platform, agent_version=AGENT_OLD)

    completed = agent.run("upgrade", "--api-url", platform.base_url)

    assert completed.returncode != 0, completed.stdout
    assert "failed" in completed.stderr.lower(), completed.stderr
    assert "Nothing was claimed" in completed.stderr, completed.stderr
    # It names the recovery, and it is still the build it started as.
    assert PACKAGE in completed.stderr
    assert agent.installed_version() == AGENT_OLD

    refused, body = claim(platform, device_token=device_token, agent_version=AGENT_OLD)
    assert refused == 426, body


def test_a_source_checkout_is_told_to_install_rather_than_upgraded(platform_at, tmp_path) -> None:
    """There is no distribution here to replace.

    A source checkout reports `0+unknown`, which the floor refuses on the same
    terms as a version that is too old. Installing a wheel over the checkout
    would leave two copies of the CLI on the path, so this stops and says what
    would actually fix it.
    """
    platform = platform_at(minimum="0.5.0", latest=AGENT_NEW)
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": str(REPO_ROOT),
            "NOMIKOS_HOME": str(tmp_path / "home"),
            "PYTHONUNBUFFERED": "1",
        }
    )
    environment.pop("NOMIKOS_API_URL", None)
    environment.pop("NOMIKOS_UPGRADED_FROM", None)

    completed = subprocess.run(
        [sys.executable, "-m", "nomikos_inference.cli", "upgrade", "--api-url", platform.base_url],
        cwd=str(tmp_path),
        env=environment,
        capture_output=True,
        text=True,
        timeout=CLI_TIMEOUT_SECONDS,
    )

    assert completed.returncode != 0, completed.stdout
    assert "source checkout" in completed.stderr, completed.stderr
    assert "install" in completed.stderr.lower()


# ---------------------------------------------------------------------------
# Both installers, because both are real install paths
# ---------------------------------------------------------------------------
def test_an_environment_with_pip_upgrades_with_pip(agent_at, platform_at) -> None:
    """`pip install nomikos-inference` leaves a pip behind, and it is the
    installer that owns that environment."""
    platform = platform_at(minimum="0.5.0", latest=AGENT_NEW)
    agent = agent_at(AGENT_OLD, with_pip=True)

    completed = agent.run("upgrade", "--api-url", platform.base_url)

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "-m pip install" in completed.stdout, completed.stdout
    assert agent.installed_version() == AGENT_NEW


# `test_an_environment_with_no_pip_upgrades_with_uv` stood here. Every other test in this
# file installs with `agent_at(...)` and no `with_pip`, i.e. the uv branch, and asserts the
# upgrade landed - so the uv path is the one this module exercises by default. The pip
# variant above is the branch nothing else covers, and it stays.


# `test_the_floor_is_asked_once_per_launch` and `test_commands_that_do_not_claim_never_ask_
# the_floor` stood here. Both counted requests off the uvicorn access log to pin a call
# count, which is an implementation detail rather than a behaviour a researcher can
# observe; the second did not test what its docstring claimed, exercising `version` and
# `--help` but never `pair`. Cutting them retired `Platform.requests_to`, the `FLOOR_PATH`
# constant, and the `--log-level info` this module's servers only needed so that log would
# be written.


# `test_asking_for_the_floor_takes_nothing_from_the_queue` and
# `test_an_agent_that_states_no_version_is_refused_by_the_floor_endpoint` stood here. Both
# were raw `urllib` calls against a platform endpoint with no CLI involved at all, and the
# second was a verbatim duplicate of `tests/nomikos/integration/test_agent_version_floor.py::
# test_an_agent_that_does_not_say_what_it_is_is_refused`. That module owns this surface and
# covers it in fifteen tests.
