"""`nomikos run`, the **claim** loop, driven as the installed console script.

Everything here is live, and on this issue that is not a stylistic preference.
Every claim the run loop makes is about behaviour *between* processes: that a
page is claimed over HTTP and reported back over HTTP, that exactly one is in
flight at a time, that a Ctrl-C delivered to a real process ends the page it was
holding. None of that is observable from inside one interpreter, and a test that
imported `nomikos_inference.cli.run` and called it would be asserting about its own
mocks.

So: the CLI is the real `nomikos` executable from a real wheel with its real
dependency closure installed, the platform is a real uvicorn process serving the
real `create_app()`, the database is real Postgres migrated by alembic, and the
models are the real **Hub artifact**s - ONNX graphs under ADR 0006, resolved
through the **Hub cache** exactly as a researcher's laptop resolves them.

Most of what is here is a platform invariant - credentials, signals, exit codes -
rather than anything about a model, and those tests run against a **Registry**
whose weights are not on this machine: a page travels the whole loop and ends
`failed` instead of `done`, which costs nothing and keeps the invariant in the
pull-request lane. The three `ml`-marked tests are the ones where the weights
are doing real work: two compare what the model produced, and
`test_only_one_page_is_ever_in_flight` needs a page to take long enough that
"one row says `waiting`" still means "the agent holds one page" - see its
docstring.

The scaffolding all three CLI integration modules share - Postgres, alembic,
uvicorn, the wheel build, the hand-rolled HTTP client - is in
`tests/inference/integration/conftest.py`. This module keeps its own database
rather than the one `tests/nomikos/integration/conftest.py` truncates between
tests, so a server held open across it cannot have the ground moved under it.
"""

from __future__ import annotations

import json
import signal
import subprocess
import time
import urllib.request
import uuid
from pathlib import Path

import pytest

from tests.fixtures.paths import SEGMENT_PAGE
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

DATABASE = "kalamos_057_run"

#: A page load plus a real model run, from a cold process. Generous on purpose:
#: an assertion about how long the runtime takes is an assertion about the machine.
RUN_TIMEOUT_SECONDS = 600.0

SERVICE_TOKEN = "test-inference-worker-service-token-not-for-production"

#: Above the wheel's version but not a floor, so every claim carries an
#: **outdated** notice - a different state from being refused.
NEWER_THAN_THIS_AGENT = "9.9.9"

#: The suite's own repository-relative settings must not reach the installed
#: package: `INFERENCE_REGISTRY_PATH` in particular would have it read the
#: **Registry** out of the checkout instead of its own bundled copy.
INHERITED_TO_DROP = (
    "PYTHONPATH",
    "INFERENCE_REGISTRY_PATH",
    "SSH_CONNECTION",
    "SSH_TTY",
    "BROWSER",
    "NOMIKOS_API_URL",
    "NOMIKOS_SERVICE_TOKEN",
    "NOMIKOS_WORKER_NAME",
)


# ---------------------------------------------------------------------------
# The platform: real app, real Postgres, real HTTP
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def media_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("media")


@pytest.fixture(scope="session")
def migrated_database() -> str:
    return migrate_database(DATABASE)


@pytest.fixture(scope="session")
def platform_url(migrated_database, media_root, tmp_path_factory):
    """The platform everything here runs against.

    `INFERENCE_AGENT_LATEST_VERSION` is set above the wheel's version so every
    claim carries an **outdated** notice. That is the served state, not the
    refused one, and having it on by default proves the loop keeps working while
    it is being told to upgrade.
    """
    log_path = tmp_path_factory.mktemp("platform") / "server.log"
    yield from serve_platform(
        migrated_database,
        log_path,
        DEVICE_PAIRING_POLL_INTERVAL_SECONDS="1",
        # Pinned, never inherited. Settings read an ambient dotenv, and a
        # developer's `.env` pointing at a live Supabase project would send this
        # suite's page images somewhere real.
        STORAGE_BACKEND="local",
        MEDIA_ROOT=str(media_root),
        INFERENCE_WORKER_SERVICE_TOKEN=SERVICE_TOKEN,
        INFERENCE_AGENT_LATEST_VERSION=NEWER_THAN_THIS_AGENT,
    )


# ---------------------------------------------------------------------------
# The CLI: a real wheel, its real closure, a real console script
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def installed_cli(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """Build the wheel and install it, closure and all, into an empty environment.

    Unlike `test_cli_pairing.py` this cannot use `--no-deps`: the whole point of
    `run` is that it executes a model, so the run loop's dependency closure is
    part of what is under test. Nothing here is satisfied from the repository
    tree. ADR 0006 needs no install flag for it - `onnxruntime` publishes one CPU
    wheel per platform, so there is no accelerator variant to exclude.
    """
    return build_and_install_cli(tmp_path_factory, install_sets=(["{wheel}"],))


def _cli_environment(home: Path, *, extra: dict[str, str] | None = None) -> dict[str, str]:
    return cli_environment(home, drop=INHERITED_TO_DROP, extra=extra)


# ---------------------------------------------------------------------------
# Acting as the researcher's browser and as the researcher
# ---------------------------------------------------------------------------
def _post(url: str, body: dict | None = None, headers: dict | None = None) -> tuple[int, object]:
    return http_request("POST", url, body, headers)


def _get(url: str, headers: dict | None = None) -> tuple[int, object]:
    return http_request("GET", url, None, headers)


def _put(url: str, body: dict, headers: dict) -> tuple[int, object]:
    return http_request("PUT", url, body, headers)


def _upload_part(base: str, document_id: str, headers: dict[str, str], image: bytes) -> str:
    """Multipart upload, hand-built. The stdlib has no client for it, and adding
    an HTTP library to reach one endpoint is not worth it."""
    boundary = f"----nomikos{uuid.uuid4().hex}"
    body = b"".join(
        [
            f"--{boundary}\r\n".encode(),
            b'Content-Disposition: form-data; name="file"; filename="page.jpeg"\r\n',
            b"Content-Type: image/jpeg\r\n\r\n",
            image,
            f"\r\n--{boundary}--\r\n".encode(),
        ]
    )
    request = urllib.request.Request(
        f"{base}/{document_id}/parts", data=body, method="POST", headers=dict(headers)
    )
    request.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())["id"]


class Researcher:
    """One account, one project, one paired machine, ready to be given work."""

    def __init__(self, platform_url: str, headers: dict[str, str], email: str, home: Path):
        self.platform_url = platform_url
        self.headers = headers
        self.email = email
        self.home = home
        slug = f"run-loop-{uuid.uuid4().hex[:8]}"
        status, project = _post(
            f"{platform_url}/projects", {"name": "Run loop", "slug": slug}, headers
        )
        assert status == 201, project
        self.project_id = project["id"]

    @property
    def documents_url(self) -> str:
        return f"{self.platform_url}/projects/{self.project_id}/documents"

    def new_page(self, image: bytes = b"") -> tuple[str, str]:
        status, document = _post(self.documents_url, {"name": "Claimable page"}, self.headers)
        assert status == 201, document
        part_id = _upload_part(
            self.documents_url, document["id"], self.headers, image or SEGMENT_PAGE.read_bytes()
        )
        return document["id"], part_id

    def submit_segment(self, ids: tuple[str, str]) -> str:
        document_id, part_id = ids
        status, body = _post(
            f"{self.documents_url}/{document_id}/parts/{part_id}/segment", None, self.headers
        )
        assert status == 202, body
        return body["job_id"]

    def prefer_local(self) -> None:
        """Without this the job is routed to `cloud` and no device token may
        claim it - the **execution target** is fixed at submission."""
        status, body = _put(
            f"{self.platform_url}/account/execution-target",
            {"prefer_local_inference": True},
            self.headers,
        )
        assert status == 200, body

    def job(self, job_id: str) -> dict:
        status, body = _get(f"{self.platform_url}/jobs/{job_id}", self.headers)
        assert status == 200, body
        return body

    def await_job(self, job_id: str, *, status: str, timeout: float = 30.0) -> dict:
        deadline = time.monotonic() + timeout
        seen = None
        while time.monotonic() < deadline:
            seen = self.job(job_id)
            if seen["status"] == status:
                return seen
            time.sleep(0.2)
        raise AssertionError(f"job {job_id} is {seen and seen['status']!r}, not {status!r}")

    def lines(self, ids: tuple[str, str]) -> list[dict]:
        document_id, part_id = ids
        status, body = _get(
            f"{self.documents_url}/{document_id}/parts/{part_id}/lines", self.headers
        )
        assert status == 200, body
        return body


def _register(platform_url: str, home: Path) -> Researcher:
    email, headers = register_account(platform_url, "run")
    return Researcher(platform_url, headers, email, home)


# ---------------------------------------------------------------------------
# Pairing this machine, through the real `nomikos pair`
# ---------------------------------------------------------------------------
def _pair(installed_cli: dict, platform_url: str, researcher: Researcher) -> str:
    """Authorise `researcher.home` by running the console script and approving it.

    Through the CLI rather than by writing `device.json`, because what `run`
    reads has to be what `pair` writes - that seam is the product, and a
    hand-built credential file would not test it.
    """
    stdout_path = researcher.home.parent / f"pair-{uuid.uuid4().hex[:6]}.txt"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)

    with stdout_path.open("w") as out_file:
        process = subprocess.Popen(
            [str(installed_cli["executable"]), "pair", "--api-url", platform_url, "--no-browser"],
            env=_cli_environment(researcher.home),
            cwd=str(installed_cli["workspace"]),
            stdout=out_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            url = await_line(process, stdout_path, APP_ORIGIN)
            assert url is not None, stdout_path.read_text()
            decide_pairing(platform_url, url, headers=researcher.headers)
            process.wait(timeout=PAIRING_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:  # pragma: no cover - a hung CLI is the failure
            process.kill()
            raise AssertionError(f"`nomikos pair` hung:\n{stdout_path.read_text()}") from None

    assert process.returncode == 0, stdout_path.read_text()
    return json.loads((researcher.home / "device.json").read_text())["device_id"]


def _announce_capacity(platform_url: str, home: Path) -> None:
    """Report **capacity** the way the agent does: by asking for work.

    Submission refuses to create a `local` page when no device for that host was
    seen recently, so this has to happen before anything is queued.
    """
    token = json.loads((home / "device.json").read_text())["device_token"]
    status, body = _post(
        f"{platform_url}/device/v1/jobs/claim",
        {"wait_seconds": 0},
        {"X-Nomikos-Device-Token": token, "X-Nomikos-Agent-Version": "1.0.0"},
    )
    assert status == 200, body


@pytest.fixture
def agent(installed_cli, platform_url, tmp_path) -> Researcher:
    """A registered researcher with this machine paired and ready to take work."""
    home = tmp_path / "nomikos-home"
    researcher = _register(platform_url, home)
    _pair(installed_cli, platform_url, researcher)
    _announce_capacity(platform_url, home)
    researcher.prefer_local()
    return researcher


# ---------------------------------------------------------------------------
# Driving `nomikos run`
# ---------------------------------------------------------------------------
class RunProcess:
    """One `nomikos run` invocation, readable while it is still running."""

    def __init__(self, process: subprocess.Popen, stdout: Path, stderr: Path):
        self.process = process
        self.stdout_path = stdout
        self.stderr_path = stderr

    @property
    def stdout(self) -> str:
        return self.stdout_path.read_text()

    @property
    def stderr(self) -> str:
        return self.stderr_path.read_text()

    @property
    def output(self) -> str:
        return f"{self.stdout}\n{self.stderr}"

    @property
    def returncode(self) -> int | None:
        return self.process.returncode

    def await_output(self, needle: str, *, timeout: float = RUN_TIMEOUT_SECONDS) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if needle in self.stdout:
                return True
            if self.process.poll() is not None:
                return needle in self.stdout
            time.sleep(0.1)
        return False

    def wait(self, timeout: float = RUN_TIMEOUT_SECONDS) -> int:
        try:
            return self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:  # pragma: no cover - a hung loop is the failure
            self.process.kill()
            self.process.wait()
            raise AssertionError(f"`nomikos run` did not finish:\n{self.output}") from None


def _start_run(
    installed_cli: dict,
    platform_url: str,
    home: Path,
    *,
    arguments: tuple[str, ...] = (),
    environment: dict[str, str] | None = None,
) -> RunProcess:
    """Start the console script with its output on files rather than pipes.

    Files because this has to read what the CLI has printed *while it is still
    running* - to know when a page has been claimed, so a signal can be sent at a
    meaningful moment - and because a loop that runs several pages prints more
    than a pipe nobody is draining will hold.
    """
    marker = uuid.uuid4().hex[:6]
    stdout_path = home.parent / f"run-stdout-{marker}.txt"
    stderr_path = home.parent / f"run-stderr-{marker}.txt"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.write_text("")
    stderr_path.write_text("")

    with stdout_path.open("w") as out_file, stderr_path.open("w") as err_file:
        process = subprocess.Popen(
            [str(installed_cli["executable"]), "run", "--api-url", platform_url, *arguments],
            env=_cli_environment(home, extra=environment),
            cwd=str(installed_cli["workspace"]),
            stdout=out_file,
            stderr=err_file,
            text=True,
        )
    return RunProcess(process, stdout_path, stderr_path)


def _run(
    installed_cli: dict,
    platform_url: str,
    home: Path,
    *,
    arguments: tuple[str, ...] = (),
    environment: dict[str, str] | None = None,
) -> RunProcess:
    started = _start_run(
        installed_cli, platform_url, home, arguments=arguments, environment=environment
    )
    started.wait()
    return started


def _broken_registry(path: Path) -> Path:
    """A **Registry** whose segmenter points at a weight file this machine does
    not have - the ordinary way a page fails to run on one particular laptop."""
    path.write_text(
        "models:\n"
        "  blla-segment:\n"
        "    task: segment\n"
        "    architecture: blla\n"
        "    device: cpu\n"
        "    host_eligibility: local\n"
        "    versions:\n"
        "      stable:\n"
        "        weights_source: file://not-installed-on-this-machine/blla.safetensors\n"
    )
    return path


# ---------------------------------------------------------------------------
# Claim, fetch, run, callback - end to end, on real weights
# ---------------------------------------------------------------------------
@pytest.mark.ml
def test_a_page_is_claimed_fetched_run_and_reported_end_to_end(
    installed_cli, platform_url, agent
) -> None:
    """The four steps ADR 0003 is built around, across one database and one hop.

    Real **Hub artifact**s: the segmenter here is the PyTorch checkpoint resolved
    through the **Hub cache**, the same one a researcher's laptop resolves.
    """
    ids = agent.new_page()
    job_id = agent.submit_segment(ids)

    run = _run(
        installed_cli,
        platform_url,
        agent.home,
        arguments=("--exit-when-empty", "--wait-seconds", "0"),
    )

    assert run.returncode == 0, run.output
    # Per-page progress, in the order the page went through the loop.
    assert "[1] segment" in run.stdout, run.stdout
    assert "fetched" in run.stdout
    assert "ran in" in run.stdout
    assert "reported done" in run.stdout

    finished = agent.await_job(job_id, status="done")
    assert finished["error"] is None
    assert finished["execution_target"] == "local"
    # The work actually landed: the platform holds lines it did not have before.
    assert len(agent.lines(ids)) > 1


@pytest.mark.ml
def test_the_cli_produces_what_the_same_code_produces_in_process(
    installed_cli, platform_url, agent
) -> None:
    """Local and cloud are the same program, so the CLI must add nothing.

    Compared against `run_model` called here, on the *stored* page bytes rather
    than the uploaded file - the platform normalises uploads to WebP, so the
    bytes the agent fetches are not the bytes that were uploaded, and comparing
    against the wrong ones would be comparing two different pages.
    """
    from nomikos_inference.contracts.common import InferenceTask
    from nomikos_inference.jobs.runner import run_model

    ids = agent.new_page()
    job_id = agent.submit_segment(ids)

    run = _run(
        installed_cli,
        platform_url,
        agent.home,
        arguments=("--exit-when-empty", "--wait-seconds", "0"),
    )
    assert run.returncode == 0, run.output
    agent.await_job(job_id, status="done")

    stored = agent.lines(ids)
    in_process = run_model(
        task=InferenceTask.segment,
        registry_model_id="blla-segment",
        registry_tag="stable",
        image_bytes=_stored_page_bytes(agent, ids),
    )

    # Two empty lists are equal, and would prove nothing about either path.
    assert len(in_process.lines) > 1, "the model found nothing, so there is no output to compare"
    assert len(stored) == len(in_process.lines)
    assert [line["points"] for line in stored] == [line.points for line in in_process.lines]


def _stored_page_bytes(agent: Researcher, ids: tuple[str, str]) -> bytes:
    """The page image exactly as the platform holds it.

    Asked for at full width, which is the one request that returns the stored
    object untouched - the same bytes the **signed page image link** serves the
    agent. A thumbnail would be a re-encode, and comparing a model's output on
    two different encodings of a page proves nothing.
    """
    request = urllib.request.Request(f"{agent.platform_url}/media/parts/{ids[1]}", method="GET")
    for name, value in agent.headers.items():
        request.add_header(name, value)
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


@pytest.mark.ml
def test_only_one_page_is_ever_in_flight(installed_cli, platform_url, agent) -> None:
    """A batch is N claims, not one claim of N pages (ADR 0002).

    Watched from outside while the loop runs: a page the agent holds is
    `waiting`, so the count of `waiting` jobs is the number in flight. If the
    loop ever claimed ahead, this would see two.

    Keeps its `ml` marker, and the real model run is load-bearing for the
    *observation* rather than for the claim. This test infers "the agent holds
    one page" from "one row says `waiting`", and those two are the same statement
    only while a page takes appreciably longer to run than the platform takes to
    move a row between states. Re-pointed at a **Registry** with no reachable
    weights - so that a page travels the whole loop and fails in milliseconds -
    it went red on `assert in_flight <= 1` roughly one run in three: the previous
    page's terminal callback and the next page's claim land close enough together
    that a sampler catches both rows saying `waiting`, which is the loop behaving
    correctly and the proxy failing to say so.

    That is a false failure on the invariant, which is the worst kind to put in
    the pull-request lane, and the only ways out weaken the test: relax the
    bound, or stop asserting a page was ever observed in flight at all. The
    honest fix is a different observation - the platform stamps `started_at` when
    a page is claimed and `completed_at` when its callback lands, so asserting
    those intervals do not overlap would settle the same question with no
    sampling and no weights. That is a rewrite rather than a re-marking, so it is
    left as one.
    """
    first = agent.submit_segment(agent.new_page())
    second = agent.submit_segment(agent.new_page())

    run = _start_run(
        installed_cli,
        platform_url,
        agent.home,
        arguments=("--exit-when-empty", "--wait-seconds", "0"),
    )
    try:
        highest = 0
        while run.process.poll() is None:
            in_flight = sum(
                1 for job_id in (first, second) if agent.job(job_id)["status"] == "waiting"
            )
            highest = max(highest, in_flight)
            assert in_flight <= 1, f"{in_flight} pages were in flight at once"
            time.sleep(0.2)
    finally:
        run.wait()

    assert run.returncode == 0, run.output
    assert highest == 1, "no page was ever observed in flight, so nothing was proved"
    assert agent.await_job(first, status="done")
    assert agent.await_job(second, status="done")
    assert "[2] segment" in run.stdout, run.stdout


# ---------------------------------------------------------------------------
# Every page ends, including the ones that do not finish
# ---------------------------------------------------------------------------
def test_a_page_that_cannot_run_here_is_reported_failed_and_the_loop_continues(
    installed_cli, platform_url, agent, tmp_path
) -> None:
    """A researcher is never left waiting on a page that already died.

    The failure is real rather than injected: this machine is given a
    **Registry** whose segmenter points at a weight file it does not have, which
    is exactly what a half-provisioned laptop looks like.
    """
    first = agent.submit_segment(agent.new_page())
    second = agent.submit_segment(agent.new_page())

    run = _run(
        installed_cli,
        platform_url,
        agent.home,
        arguments=("--exit-when-empty", "--wait-seconds", "0"),
        environment={"INFERENCE_REGISTRY_PATH": str(_broken_registry(tmp_path / "registry.yaml"))},
    )

    # The loop did not stop at the first failure, and did not exit non-zero for
    # it either: a page failing here is an outcome, not a reason to give up.
    assert run.returncode == 0, run.output
    assert "[1] segment" in run.stdout
    assert "[2] segment" in run.stdout, "the loop stopped after the first failure"
    assert run.stdout.count("reported failed") == 2, run.stdout

    for job_id in (first, second):
        failed = agent.await_job(job_id, status="failed")
        assert failed["error"], "a failed page must carry its reason"
        assert "BLLA model not found" in failed["error"], failed["error"]


def test_ctrl_c_reports_the_page_in_flight_before_exiting(
    installed_cli, platform_url, agent
) -> None:
    """A considerate shutdown leaves nothing stuck.

    The signal goes to a real process at a real moment - once it has said it is
    holding a page - and the page has to be terminal on the platform afterwards.
    Only a *crash* is allowed to leave a page for the **lease**.
    """
    job_id = agent.submit_segment(agent.new_page())

    run = _start_run(installed_cli, platform_url, agent.home, arguments=("--wait-seconds", "0"))
    try:
        assert run.await_output("[1] segment", timeout=120), run.output
        run.process.send_signal(signal.SIGINT)
        run.wait(timeout=120)
    finally:
        if run.process.poll() is None:  # pragma: no cover - only on an unresponsive CLI
            run.process.kill()

    assert run.returncode == 130, run.output  # 128 + SIGINT, by the shell convention
    assert "reported failed" in run.stdout, run.stdout
    assert "Stopped." in run.stdout

    ended = agent.await_job(job_id, status="failed")
    assert ended["error"], "the interrupted page was reported without a reason"


# `test_a_killed_process_leaves_a_page_the_lease_later_releases` stood here. The lease
# semantics it asserted - an expired lease re-pends a page rather than failing it, and
# hands it to a different agent - are `tests/nomikos/integration/test_device_lease.py::
# test_an_expired_lease_is_re_pended_and_never_failed` and
# `::test_an_expired_lease_returns_the_page_to_a_different_agent`, tested against the
# platform that implements them. The CLI increment was "SIGKILL cannot be caught", which
# is an OS guarantee. It cost a second Postgres database, a second alembic run, a third
# uvicorn process and a hard `time.sleep(32)`; all four went with it, along with the
# `short_lease_platform_url` fixture and the `LEASE_*` constants that existed only for it.


# ---------------------------------------------------------------------------
# An empty queue: the two behaviours one flag chooses between
# ---------------------------------------------------------------------------
def test_an_empty_queue_ends_the_run_when_asked_to(installed_cli, platform_url, agent) -> None:
    """The scripted half: nothing queued, so there is nothing to wait for."""
    started = time.monotonic()
    run = _run(
        installed_cli,
        platform_url,
        agent.home,
        arguments=("--exit-when-empty", "--wait-seconds", "0"),
    )
    elapsed = time.monotonic() - started

    assert run.returncode == 0, run.output
    assert "Nothing left in the queue." in run.stdout
    assert elapsed < 60, f"it waited {elapsed:.1f}s for a queue it was told to give up on"
    # Being **outdated** is a notice delivered with the work, not a refusal: it
    # is said once and the loop carries on.
    assert NEWER_THAN_THIS_AGENT in run.stdout, run.stdout


# `test_an_empty_queue_is_waited_on_by_default` stood here. Its SIGINT half - exit 130 and
# "Stopped." from a real signal - is `test_ctrl_c_reports_the_page_in_flight_before_exiting`
# above, which additionally pins that the page in flight is reported. Its "does not exit on
# empty" half is the negative of `test_an_empty_queue_ends_the_run_when_asked_to`, which
# asserts the flag is what makes the difference.


# ---------------------------------------------------------------------------
# The two credentials, and the version floor in front of both
# ---------------------------------------------------------------------------
def test_a_hosted_worker_runs_the_same_loop_with_a_service_credential(
    installed_cli, platform_url, tmp_path
) -> None:
    """Local and cloud differ by credential and uptime, not by code path (ADR 0003).

    No `device.json` exists in this home directory at all: the only thing making
    this process an **inference agent** is the **service credential** in its
    environment, and the only thing it changes is that the work it takes is
    `cloud` work.

    Deliberately *not* `ml`-marked. Every hop this test is about is an
    authentication hop - the claim, the **signed page image link**, and the
    terminal callback all carry the service credential and none of them branch on
    whether the page ran - so the **Registry** here has no reachable weights and
    the page ends `failed`. That the callback carries *output* the platform
    accepts is `test_a_page_is_claimed_fetched_run_and_reported_end_to_end`,
    which does need real weights and keeps its marker. Splitting them this way
    puts the credential path in the pull-request lane, which is where a broken
    service token should be caught.
    """
    home = tmp_path / "worker-home"
    home.mkdir(parents=True)
    researcher = _register(platform_url, home)
    worker = {
        "NOMIKOS_SERVICE_TOKEN": SERVICE_TOKEN,
        "NOMIKOS_WORKER_NAME": "test-worker",
        "INFERENCE_REGISTRY_PATH": str(_broken_registry(tmp_path / "registry.yaml")),
    }

    # A hosted worker registers itself by working: this first claim provisions
    # the `cloud` device row that reports cloud **capacity**, without which
    # submission would announce no host rather than create the page.
    status, body = _post(
        f"{platform_url}/device/v1/jobs/claim",
        {"wait_seconds": 0},
        {
            "X-Nomikos-Service-Token": SERVICE_TOKEN,
            "X-Nomikos-Worker-Name": "test-worker",
            "X-Nomikos-Agent-Version": "1.0.0",
        },
    )
    assert status == 200, body

    job_id = researcher.submit_segment(researcher.new_page())
    assert researcher.job(job_id)["execution_target"] == "cloud"
    assert not (home / "device.json").exists()

    run = _run(
        installed_cli,
        platform_url,
        home,
        arguments=("--exit-when-empty", "--wait-seconds", "0"),
        environment=worker,
    )

    assert run.returncode == 0, run.output
    # It took `cloud` work, and it took it with nothing but the service credential.
    assert "cloud work" in run.stdout, run.stdout
    # The terminal callback went out over that credential and the platform took it.
    assert "reported failed" in run.stdout, run.output
    assert researcher.await_job(job_id, status="failed")["error"], "the page ended with no reason"


# `test_an_agent_below_the_version_floor_is_refused_and_told_what_to_do` stood here, with a
# whole extra uvicorn process (`refusing_platform_url`) behind it. What the loop does with
# a 426 is `tests/inference/unit/test_cli_run_resilience.py::
# test_a_version_refusal_still_leaves_the_loop`, and the message and exit code are
# `test_cli_self_upgrade.py::test_a_failed_upgrade_exits_non_zero_and_claims_nothing`.
