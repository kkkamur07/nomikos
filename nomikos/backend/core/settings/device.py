"""Helper device pairing and device-token settings.

Every value here is an operational dial that can be turned without shipping a
new helper build - the helper reads its poll cadence and lifetimes from the
platform rather than compiling them in.
"""

import logging

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings

from backend.core.settings._cache import settings_cache
from backend.core.settings._env import env_settings_config
from backend.core.settings.secrets import is_placeholder_secret

logger = logging.getLogger(__name__)


class DeviceSettings(BaseSettings):
    model_config = env_settings_config()

    environment: str = Field(default="development", alias="ENVIRONMENT")
    device_pairing_enabled: bool | None = Field(
        default=None,
        alias="DEVICE_PAIRING_ENABLED",
        description=(
            "Kill switch for the whole device layer. Unset means off in production "
            "and on everywhere else; see pairing_enabled()."
        ),
    )

    device_token_hmac_secret: str | None = Field(
        default=None,
        alias="DEVICE_TOKEN_HMAC_SECRET",
        description=(
            "Keys every device credential digest. Required in production once "
            "DEVICE_PAIRING_ENABLED is true; falls back to JWT_SECRET elsewhere."
        ),
    )
    device_token_ttl_days: int = Field(default=180, alias="DEVICE_TOKEN_TTL_DAYS", ge=1, le=730)
    device_token_renew_overlap_hours: int = Field(
        default=24, alias="DEVICE_TOKEN_RENEW_OVERLAP_HOURS", ge=0, le=168
    )
    device_max_per_user: int = Field(default=10, alias="DEVICE_MAX_PER_USER", ge=1, le=100)
    device_online_window_seconds: int = Field(
        default=120, alias="DEVICE_ONLINE_WINDOW_SECONDS", ge=10
    )
    device_idle_window_seconds: int = Field(default=900, alias="DEVICE_IDLE_WINDOW_SECONDS", ge=10)

    # ------------------------------------------------------------------
    # Claim protocol (ADR 0003). One page per claim, no heartbeat.
    # ------------------------------------------------------------------
    device_lease_seconds: int = Field(
        default=600,
        alias="DEVICE_LEASE_SECONDS",
        ge=30,
        le=3600,
        description=(
            "How long one claimed page stays with the agent that took it. Work is "
            "seconds-to-minutes, so this covers it with margin and no heartbeat "
            "endpoint is needed; a slept laptop loses one page, not a document. "
            "Deliberately shorter than JOB_WORKER_RUNNING_TIMEOUT_SECONDS (1800): "
            "that one is sized for a server that does not sleep, and a closed lid "
            "must not hold a page for half an hour. When it expires the page goes "
            "back to pending, not to failed - see release_expired_device_leases."
        ),
    )
    device_page_image_url_ttl_seconds: int = Field(
        default=60,
        alias="DEVICE_PAGE_IMAGE_URL_TTL_SECONDS",
        ge=10,
        le=600,
        description=(
            "How long the signed page image link in a claim response stays usable. "
            "Deliberately independent of DEVICE_LEASE_SECONDS: the agent fetches the "
            "image once, immediately after claiming, so this only has to cover "
            "downloading a large scan on a bad connection - not the whole run. Tying it "
            "to the lease would leave a bearer token in a URL alive for ten minutes to "
            "buy nothing."
        ),
    )
    device_claim_max_wait_seconds: int = Field(
        default=25,
        alias="DEVICE_CLAIM_MAX_WAIT_SECONDS",
        ge=0,
        le=120,
        description=(
            "Ceiling on how long one claim request waits for work. A laptop long-polls "
            "up to this; a hosted worker sends 0 and short-polls, because it is never "
            "idle for long and does not need the latency."
        ),
    )
    device_claim_poll_interval_seconds: float = Field(
        default=1.0,
        alias="DEVICE_CLAIM_POLL_INTERVAL_SECONDS",
        gt=0,
        le=30,
        description="Server-side re-check cadence inside one long poll. No connection is held between checks.",
    )
    device_claim_idle_poll_seconds: float = Field(
        default=5.0,
        alias="DEVICE_CLAIM_IDLE_POLL_SECONDS",
        gt=0,
        le=300,
        description="What an empty claim response tells the agent to wait before asking again.",
    )
    # ------------------------------------------------------------------
    # The agent version floor (ADR 0002). Three dials, all turnable without a
    # release - which is the entire reason the signal comes from here rather
    # than from PyPI. Raising the floor stops a known-bad agent from claiming
    # in the time it takes to change an environment variable, with nothing to
    # install and nobody to wait for.
    # ------------------------------------------------------------------
    inference_agent_min_version: str = Field(
        default="0.1.0",
        alias="INFERENCE_AGENT_MIN_VERSION",
        description=(
            "Oldest agent allowed to claim. Anything below it is refused with 426 and "
            "must upgrade; a missing or unparseable version is refused too, rather "
            "than assumed current."
        ),
    )
    inference_agent_latest_version: str | None = Field(
        default=None,
        alias="INFERENCE_AGENT_LATEST_VERSION",
        description=(
            "Newest published agent. An agent at or above the floor but below this is "
            "served normally and told it is outdated - deliberately a different state "
            "from being refused. Unset means the floor is the newest thing we know "
            "about, so nobody is told they are behind; see agent_latest_version()."
        ),
    )
    inference_agent_package: str = Field(
        default="nomikos-inference",
        alias="INFERENCE_AGENT_PACKAGE",
        min_length=1,
        max_length=64,
        description=(
            "Distribution name the refusal tells the agent to upgrade. A package name "
            "rather than a command string: the platform names what to install, it does "
            "not hand a remote agent a line to execute."
        ),
    )

    inference_worker_service_token: str | None = Field(
        default=None,
        alias="INFERENCE_WORKER_SERVICE_TOKEN",
        description=(
            "The hosted worker's credential for the claim endpoint. A service credential "
            "rather than a device token (ADR 0003): it claims cloud work, and cloud work "
            "belongs to the platform rather than to one researcher. Unset means no hosted "
            "worker can claim."
        ),
    )

    device_pairing_ttl_seconds: int = Field(
        default=300,
        alias="DEVICE_PAIRING_TTL_SECONDS",
        ge=60,
        le=3600,
        description="How long one consent link stays live. This is the phishing window.",
    )
    device_pairing_max_lifetime_seconds: int = Field(
        default=900,
        alias="DEVICE_PAIRING_MAX_LIFETIME_SECONDS",
        ge=300,
        le=86_400,
        description=(
            "Hard cap on how far polling can extend a pairing request. The poller is "
            "whoever created the request, so this - not the TTL - is the real ceiling "
            "on how long a transferable consent link survives."
        ),
    )
    device_pairing_max_live_total: int = Field(
        default=10_000,
        alias="DEVICE_PAIRING_MAX_LIVE_TOTAL",
        ge=1,
        description=(
            "Platform-wide ceiling on live pairing requests. A table-growth backstop, "
            "not an abuse control - see DevicePairingService.start_pairing."
        ),
    )
    device_pairing_retention_seconds: int = Field(
        default=86_400,
        alias="DEVICE_PAIRING_RETENTION_SECONDS",
        ge=60,
        description="How long a dead pairing row is kept before the sweep deletes it.",
    )
    device_pairing_poll_interval_seconds: int = Field(
        default=5, alias="DEVICE_PAIRING_POLL_INTERVAL_SECONDS", ge=1, le=60
    )
    device_pairing_max_poll_interval_seconds: int = Field(
        default=30, alias="DEVICE_PAIRING_MAX_POLL_INTERVAL_SECONDS", ge=1, le=300
    )
    device_pairing_max_attempts: int = Field(
        default=5,
        alias="DEVICE_PAIRING_MAX_ATTEMPTS",
        ge=1,
        le=20,
        description="Wrong device_code presentations before the pairing row is burned.",
    )
    device_pairing_app_origin: str | None = Field(
        default=None,
        alias="DEVICE_PAIRING_APP_ORIGIN",
        description="Origin of the SPA that renders /pair; defaults to the first CORS origin.",
    )

    @model_validator(mode="after")
    def _validate_secret(self) -> "DeviceSettings":
        if self.device_token_hmac_secret is None:
            return self
        normalized = self.device_token_hmac_secret.strip()
        if len(normalized) < 32 or is_placeholder_secret(normalized):
            raise ValueError(
                "DEVICE_TOKEN_HMAC_SECRET must be at least 32 non-placeholder characters"
            )
        return self

    @model_validator(mode="after")
    def _validate_service_token(self) -> "DeviceSettings":
        """A weak service credential claims *every* account's cloud work.

        A device token is bounded by one ``helper_devices.user_id``; this one is
        not bounded by anything, because cloud work is platform work. So it is
        held to the same floor as the HMAC key rather than to none.
        """
        if self.inference_worker_service_token is None:
            return self
        normalized = self.inference_worker_service_token.strip()
        if len(normalized) < 32 or is_placeholder_secret(normalized):
            raise ValueError(
                "INFERENCE_WORKER_SERVICE_TOKEN must be at least 32 non-placeholder characters"
            )
        return self

    @model_validator(mode="after")
    def _validate_agent_version_policy(self) -> "DeviceSettings":
        """A floor nobody can parse is a floor that refuses every agent.

        Both dials are compared against every claim, so an unparseable one would
        fail the whole claim path at request time - as a 500, on the endpoint all
        inference runs through. Refusing to start is the cheaper failure, and it
        happens on the deploy that set the bad value rather than on the first
        agent that polls afterwards.

        ``latest`` below ``minimum`` is also refused: it would describe a
        platform that refuses an agent for being below the floor and calls the
        same version ahead of the newest release.
        """
        from backend.ml.domain.agent_version import AgentVersion, MalformedAgentVersionError

        try:
            minimum = AgentVersion.parse(self.inference_agent_min_version)
            latest = AgentVersion.parse(self.agent_latest_version())
        except MalformedAgentVersionError as exc:
            raise ValueError(
                "INFERENCE_AGENT_MIN_VERSION and INFERENCE_AGENT_LATEST_VERSION must be "
                f"MAJOR.MINOR.PATCH versions: {exc}"
            ) from exc
        if latest < minimum:
            raise ValueError(
                "INFERENCE_AGENT_LATEST_VERSION must not be below INFERENCE_AGENT_MIN_VERSION"
            )
        return self

    @model_validator(mode="after")
    def _validate_poll_interval(self) -> "DeviceSettings":
        if (
            self.device_pairing_max_poll_interval_seconds
            < self.device_pairing_poll_interval_seconds
        ):
            raise ValueError(
                "DEVICE_PAIRING_MAX_POLL_INTERVAL_SECONDS must not be below "
                "DEVICE_PAIRING_POLL_INTERVAL_SECONDS"
            )
        if self.device_pairing_max_lifetime_seconds < self.device_pairing_ttl_seconds:
            raise ValueError(
                "DEVICE_PAIRING_MAX_LIFETIME_SECONDS must not be below DEVICE_PAIRING_TTL_SECONDS"
            )
        return self

    @model_validator(mode="after")
    def _validate_production_credential_key(self) -> "DeviceSettings":
        """Fail closed rather than key device tokens off ``JWT_SECRET``.

        Sharing the key means a routine ``JWT_SECRET`` rotation - which today
        only logs browsers out - also unpairs every UI-less laptop, which is not
        recoverable without a terminal. In production that is a refusal to start
        once the feature is on, and a loud warning while it is off.
        """
        if self.environment.casefold() != "production":
            return self
        problem = self._device_key_problem()
        if problem is None:
            return self
        if self.pairing_enabled():
            raise ValueError(problem)
        logger.warning("%s Device pairing is disabled, so this is not fatal yet.", problem)
        return self

    def _device_key_problem(self) -> str | None:
        if not self.device_token_hmac_secret:
            return (
                "DEVICE_TOKEN_HMAC_SECRET must be set in production; falling back to "
                "JWT_SECRET makes a JWT rotation unpair every helper."
            )
        if self.device_token_hmac_secret == self._jwt_secret():
            return (
                "DEVICE_TOKEN_HMAC_SECRET must differ from JWT_SECRET; sharing the key "
                "makes a JWT rotation unpair every helper."
            )
        return None

    @staticmethod
    def _jwt_secret() -> str | None:
        from backend.core.settings.auth import get_auth_settings

        try:
            return get_auth_settings().jwt_secret
        except Exception:  # pragma: no cover - AuthSettings reports its own failure
            return None

    def agent_latest_version(self) -> str:
        """Newest published agent, defaulting to the floor.

        Defaulting rather than requiring both is what keeps the kill switch a
        one-variable turn: raising ``INFERENCE_AGENT_MIN_VERSION`` alone must not
        make the API refuse to start because a second variable was left at an
        older release. With only the floor set, nothing is ever "outdated" - the
        newest thing the platform knows about is the oldest thing it accepts.
        """
        return self.inference_agent_latest_version or self.inference_agent_min_version

    def pairing_enabled(self) -> bool:
        """Whether the device layer serves requests at all.

        Off by default in production until the ``/pair`` consent page exists:
        pairing endpoints that mint a 180-day credential must not be reachable
        before the screen that explains what is being granted.
        """
        if self.device_pairing_enabled is not None:
            return self.device_pairing_enabled
        return self.environment.casefold() != "production"

    def hmac_key(self) -> str:
        """Key for every device credential digest.

        A dedicated secret rather than ``JWT_SECRET`` because the blast radius
        differs: rotating ``JWT_SECRET`` logs browsers out, which a researcher
        recovers from by logging in again. Rotating the key that unpairs every
        UI-less laptop is not recoverable without a terminal. The fallback below
        exists for development only - production refuses to start on it, see
        :meth:`_validate_production_credential_key`.
        """
        if self.device_token_hmac_secret:
            return self.device_token_hmac_secret
        from backend.core.settings.auth import get_auth_settings

        return get_auth_settings().jwt_secret

    def pair_url_origin(self) -> str:
        """Origin the helper opens for consent."""
        if self.device_pairing_app_origin:
            return self.device_pairing_app_origin.rstrip("/")
        from backend.core.settings.app import get_app_settings

        return get_app_settings().cors_origin_list[0].rstrip("/")


@settings_cache
def get_device_settings() -> DeviceSettings:
    return DeviceSettings()
