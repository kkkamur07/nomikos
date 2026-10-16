"""Background job worker settings."""

from pydantic import Field
from pydantic_settings import BaseSettings

from backend.core.settings._cache import settings_cache
from backend.core.settings._env import env_settings_config


class JobSettings(BaseSettings):
    model_config = env_settings_config()

    enable_test_job_routes: bool = Field(default=False, alias="ENABLE_TEST_JOB_ROUTES")
    job_worker_enabled: bool = Field(default=True, alias="JOB_WORKER_ENABLED")
    media_gc_enabled: bool = Field(
        default=True,
        alias="MEDIA_GC_ENABLED",
        description="Run the abandoned-upload sweeper in the API lifespan. Disable it on "
        "replicas that should not contend for the same rows.",
    )
    job_sse_notifications_enabled: bool = Field(
        default=True,
        alias="JOB_SSE_NOTIFICATIONS_ENABLED",
        description="Postgres NOTIFY listener for SSE; disable on serverless API hosts.",
    )
    job_poll_interval_seconds: float = Field(default=0.25, alias="JOB_POLL_INTERVAL_SECONDS")
    job_poll_max_interval_seconds: float = Field(default=2.0, alias="JOB_POLL_MAX_INTERVAL_SECONDS")
    platform_job_notify_channel: str = Field(
        default="platform_jobs",
        alias="PLATFORM_JOB_NOTIFY_CHANNEL",
    )
    job_sse_heartbeat_seconds: float = Field(default=45.0, alias="JOB_SSE_HEARTBEAT_SECONDS")
    job_worker_claim_test_only: bool | None = Field(
        default=None,
        alias="JOB_WORKER_CLAIM_TEST_ONLY",
        description="True=only test payloads; False=exclude test; None=claim any pending job",
    )
    job_worker_running_timeout_seconds: float = Field(
        default=1800.0,
        alias="JOB_WORKER_RUNNING_TIMEOUT_SECONDS",
    )
    job_worker_waiting_timeout_seconds: float = Field(
        default=240.0,
        alias="JOB_WORKER_WAITING_TIMEOUT_SECONDS",
        description="Fail a job dispatched to inference when no callback arrives in time.",
    )
    job_worker_callback_claim_timeout_seconds: float = Field(
        default=300.0,
        alias="JOB_WORKER_CALLBACK_CLAIM_TIMEOUT_SECONDS",
        description="Release an abandoned callback claim so the job can be cancelled again.",
    )
    job_stale_sweep_on_read_enabled: bool = Field(
        default=True,
        alias="JOB_STALE_SWEEP_ON_READ_ENABLED",
        description=(
            "Run the stale-job sweeps from job read paths. Required on request/response "
            "hosts where JOB_WORKER_ENABLED=false leaves no loop to run them."
        ),
    )
    job_stale_sweep_min_interval_seconds: float = Field(
        default=30.0,
        alias="JOB_STALE_SWEEP_MIN_INTERVAL_SECONDS",
        description="Per-process floor between opportunistic sweeps; caps the cost per request.",
    )
    job_queue_stall_warning_seconds: float = Field(
        default=900.0,
        alias="JOB_QUEUE_STALL_WARNING_SECONDS",
        description=(
            "Age at which /health logs a WARNING about the oldest pending job. Nothing "
            "claims pending work unless a platform worker (JOB_WORKER_ENABLED, a "
            "separate host) or an inference agent is running, and neither is part of "
            "the API deployment - so this is the only signal that the queue has no "
            "consumer. Default 900s: above one full DEVICE_LEASE_SECONDS (600) so a "
            "single abandoned lease cannot trip it, below "
            "JOB_WORKER_RUNNING_TIMEOUT_SECONDS (1800) so a stalled queue is named "
            "before an in-flight job is even reclaimed."
        ),
    )


@settings_cache
def get_job_settings() -> JobSettings:
    return JobSettings()
