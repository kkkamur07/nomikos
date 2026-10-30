"""Whose credential a claim runs as: a hosted worker's, or this machine's.

Moved out of the claim loop so `claim.py` holds only work in flight. The
**execution target** is fixed by the credential (ADR 0005, decision 1); this
module only resolves which credential that is.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from nomikos_inference.cli.api import (
    AGENT_VERSION_HEADER,
    DEVICE_TOKEN_HEADER,
    SERVICE_TOKEN_HEADER,
    WORKER_NAME_HEADER,
    this_machine_name,
)
from nomikos_inference.cli.credentials import (
    CredentialError,
    credential_path,
    load_credential,
)
from nomikos_inference.cli.version import installed_version

SERVICE_TOKEN_ENV = "NOMIKOS_SERVICE_TOKEN"  # noqa: S105 - an env var name, not a token
"""A hosted worker's **service credential**, read from the environment and
never from a flag. A token passed on the command line ends up in `ps` output
and shell history, and this one isn't bounded to a single account the way a
device token is."""

WORKER_NAME_ENV = "NOMIKOS_WORKER_NAME"

LAPTOP_WAIT_SECONDS = 25
"""A researcher's page should start within a second of being submitted, so a
laptop long-polls. The platform clamps this to its own ceiling."""

WORKER_WAIT_SECONDS = 0
"""A hosted worker short-polls instead: it's never idle for long, doesn't need
the latency, and every second it waits is a request-handler slot held on a
serverless host (ADR 0003)."""


class RunSetupError(RuntimeError):
    """This machine cannot start claiming, and says why."""


@dataclass(frozen=True)
class AgentIdentity:
    """Which credential this process claims with, and what that credential means.

    The **execution target** is not negotiable and is not sent: the
    credential fixes it (ADR 0005, decision 1). This records it only so the
    loop can say out loud whose work it is about to take.
    """

    credential: dict[str, str]
    account: str
    execution_target: str
    default_wait_seconds: int


def _resolve_identity(base_url: str) -> AgentIdentity:
    """A **service credential** if one is in the environment, else this machine's.

    Service-first, matching the order the platform resolves them in
    (`resolve_inference_agent`): a host given a service token is a hosted
    worker, and a stray `device.json` in its home directory must not quietly
    turn it into somebody's laptop.
    """
    version = installed_version()
    service_token = (os.environ.get(SERVICE_TOKEN_ENV) or "").strip()
    if service_token:
        worker_name = (os.environ.get(WORKER_NAME_ENV) or "").strip() or this_machine_name()
        return AgentIdentity(
            credential={
                SERVICE_TOKEN_HEADER: service_token,
                WORKER_NAME_HEADER: worker_name,
                AGENT_VERSION_HEADER: version,
            },
            account=f"the platform, as {worker_name}",
            execution_target="cloud",
            default_wait_seconds=WORKER_WAIT_SECONDS,
        )

    try:
        credential = load_credential()
    except CredentialError as exc:
        raise RunSetupError(
            f"{exc}\nDelete {credential_path()} and run `nomikos pair` again to replace it."
        ) from exc
    if credential is None:
        raise RunSetupError(
            "This machine is not paired, so it has nothing to claim work with.\n"
            "Run `nomikos pair` to authorise it against your account."
        )
    if credential.platform_url != base_url:
        # The credential only means something against the platform that
        # minted it; presenting it elsewhere is a 401 with no explanation.
        raise RunSetupError(
            f"This machine is paired with {credential.platform_url}, not {base_url}.\n"
            "Run `nomikos pair --force` to pair it with this platform instead."
        )

    return AgentIdentity(
        credential={
            DEVICE_TOKEN_HEADER: credential.device_token,
            AGENT_VERSION_HEADER: version,
        },
        account=credential.account_email or "this account",
        execution_target="local",
        default_wait_seconds=LAPTOP_WAIT_SECONDS,
    )
