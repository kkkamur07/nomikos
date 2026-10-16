"""API application settings (CORS, media paths)."""

from ipaddress import ip_network
from pathlib import Path
from urllib.parse import urlparse

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings

from backend.core.settings._cache import settings_cache
from backend.core.settings._env import REPO_ROOT, env_settings_config


def _split_origins(value: str) -> list[str]:
    return [origin.strip().rstrip("/") for origin in value.split(",") if origin.strip()]


def _check_origin(origin: str) -> str:
    """Reject anything that is not one explicit http(s) origin."""
    parsed = urlparse(origin)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("CORS_ORIGINS must contain explicit http(s) origins only")
    if parsed.username or parsed.password:
        raise ValueError("CORS_ORIGINS must contain explicit http(s) origins only")
    if parsed.path not in {"", "/"} or parsed.params or parsed.query or parsed.fragment:
        raise ValueError("CORS_ORIGINS must contain explicit http(s) origins only")
    if "*" in origin:
        raise ValueError("CORS_ORIGINS must contain explicit http(s) origins only")
    return origin


class AppSettings(BaseSettings):
    model_config = env_settings_config()

    media_root: Path = Field(default=REPO_ROOT / "backend" / "media", alias="MEDIA_ROOT")
    cors_origins: str = Field(
        default="http://localhost:3000,http://localhost:5173",
        alias="CORS_ORIGINS",
    )
    behind_proxy: bool = Field(default=False, alias="BEHIND_PROXY")
    forwarded_allow_ips: str | None = Field(default=None, alias="FORWARDED_ALLOW_IPS")
    #: Whether the direct TCP peer identifies a single client.
    #:
    #: Set to ``false`` on any platform that terminates connections in a shared
    #: proxy tier whose addresses can't be allowlisted (e.g. Vercel serverless
    #: functions). There, the peer is the platform's proxy, so a "per-IP" limit
    #: keyed on it is really one global bucket that isolates nobody. When this
    #: is ``false`` and no trusted forwarded header is configured, IP-keyed
    #: throttles are skipped and identity-keyed throttles carry the load.
    trust_peer_ip: bool = Field(default=True, alias="TRUST_PEER_IP")

    @field_validator("forwarded_allow_ips")
    @classmethod
    def _validate_forwarded_allow_ips(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None

        entries = [entry.strip() for entry in value.split(",") if entry.strip()]
        if not entries:
            return None
        if "*" in entries:
            raise ValueError(
                "FORWARDED_ALLOW_IPS must contain explicit proxy IPs or CIDRs, never '*'"
            )

        networks: list[str] = []
        for entry in entries:
            try:
                networks.append(str(ip_network(entry, strict=False)))
            except ValueError as exc:
                raise ValueError(
                    "FORWARDED_ALLOW_IPS must contain only explicit proxy IPs or CIDRs"
                ) from exc
        return ",".join(dict.fromkeys(networks))

    @model_validator(mode="after")
    def _require_proxy_allowlist(self) -> "AppSettings":
        if self.behind_proxy and not self.forwarded_allow_ips:
            raise ValueError(
                "BEHIND_PROXY=true requires FORWARDED_ALLOW_IPS with explicit proxy IPs or CIDRs"
            )
        return self

    @field_validator("cors_origins")
    @classmethod
    def _validate_cors_origins(cls, value: str) -> str:
        origins = _split_origins(value)
        if not origins:
            raise ValueError("CORS_ORIGINS must include at least one explicit origin")
        return ",".join(dict.fromkeys(_check_origin(origin) for origin in origins))

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@settings_cache
def get_app_settings() -> AppSettings:
    return AppSettings()
