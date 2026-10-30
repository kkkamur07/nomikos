"""Device lease vs global job timeout: a config invariant, no client, no database."""

from backend.core.settings.device import get_device_settings
from backend.core.settings.job import get_job_settings


def test_the_device_lease_is_shorter_than_the_global_job_timeout() -> None:
    """1800 seconds is right for a server that does not sleep and wrong for a
    laptop that does. A closed lid must not hold a page for half an hour."""
    device_settings = get_device_settings()
    settings = get_job_settings()

    assert device_settings.device_lease_seconds == 600
    assert device_settings.device_lease_seconds < settings.job_worker_running_timeout_seconds
    assert settings.job_worker_running_timeout_seconds == 1800
    # And distinct from the waiting timeout, which governs the other population.
    assert device_settings.device_lease_seconds != settings.job_worker_waiting_timeout_seconds
