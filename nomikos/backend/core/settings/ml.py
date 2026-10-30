"""ML service integration settings."""

import os
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings

from backend.core.settings._cache import settings_cache
from backend.core.settings._env import REPO_ROOT, env_settings_config
from backend.core.settings.secrets import is_placeholder_secret


def _default_inference_registry_path() -> Path:
    app_root = Path(os.environ.get("NOMIKOS_APP_ROOT", REPO_ROOT.parent))
    return app_root / "nomikos_inference" / "registry.yaml"


class MLSettings(BaseSettings):
    model_config = env_settings_config()

    environment: str = Field(default="development", alias="ENVIRONMENT")
    cloud_inference_enabled: bool = Field(default=False, alias="CLOUD_INFERENCE_ENABLED")
    # There is no inference service URL to hold. The platform does not call out
    # to inference at all any more: an agent claims work from the platform and
    # reports back through the job callback contract (ADR 0003), so the only
    # inference credential an API process needs is the one that authenticates
    # that inbound callback.
    inference_webhook_secret: str | None = Field(default=None, alias="INFERENCE_WEBHOOK_SECRET")
    inference_registry_path: Path = Field(
        default_factory=_default_inference_registry_path,
        alias="INFERENCE_REGISTRY_PATH",
    )

    @model_validator(mode="after")
    def _validate_production_runtime(self) -> "MLSettings":
        if self.environment.casefold() != "production":
            return self

        if self.inference_webhook_secret is not None and is_placeholder_secret(
            self.inference_webhook_secret
        ):
            raise ValueError(
                "INFERENCE_WEBHOOK_SECRET must be set to a non-placeholder secret in production"
            )
        return self

    def require_callback_receiver_configuration(self) -> None:
        """Fail closed when an API process accepts inference completion callbacks."""
        if self.environment.casefold() != "production":
            return
        if is_placeholder_secret(self.inference_webhook_secret):
            raise ValueError(
                "INFERENCE_WEBHOOK_SECRET must be set to a non-placeholder secret in production"
            )


@settings_cache
def get_inference_settings() -> MLSettings:
    return MLSettings()
