"""Environment settings for the inference runtime.

Nothing here reaches a database. The inference runtime reads a registry file and
runs models; the only job queue is the platform's (ADR 0003).

Nothing here opens a port either. ADR 0002 deleted the loopback service, so
there is no listener whose callers need authenticating and therefore no service
secret: an **inference agent** presents its **device credential** to the
platform, outbound, and nothing presents anything to it.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from nomikos_inference import INFERENCE_ROOT
from nomikos_inference.admission import AdmissionSettings


class InferenceSettings(AdmissionSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    environment: str = Field(default="development", alias="ENVIRONMENT")
    inference_registry_path: Path = Field(
        default=INFERENCE_ROOT / "registry.yaml",
        alias="INFERENCE_REGISTRY_PATH",
    )


@lru_cache
def get_inference_settings() -> InferenceSettings:
    return InferenceSettings()
