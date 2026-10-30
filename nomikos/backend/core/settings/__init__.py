"""Application settings - split by concern, single .env at backend/core/.env.

Each concern keeps its own class and its own accessor: a caller that needs the
job worker's timeouts should say so by importing ``get_job_settings``, not by
reaching through an aggregate that also carries the JWT secret.

Importing this package imports all eight modules, which is what makes
:func:`reset_settings_caches` complete - it is the one supported way to discard
memoized settings after the environment changes, and it replaces enumerating the
accessors at the call site.
"""

from backend.core.settings._cache import reset_settings_caches
from backend.core.settings.app import AppSettings, get_app_settings
from backend.core.settings.auth import AuthSettings, get_auth_settings
from backend.core.settings.device import DeviceSettings, get_device_settings
from backend.core.settings.email import EmailSettings, get_email_settings
from backend.core.settings.infrastructure import InfrastructureSettings, get_infrastructure_settings
from backend.core.settings.job import JobSettings, get_job_settings
from backend.core.settings.ml import MLSettings, get_inference_settings
from backend.core.settings.storage import StorageSettings, get_storage_settings

__all__ = [
    "AppSettings",
    "AuthSettings",
    "DeviceSettings",
    "EmailSettings",
    "InfrastructureSettings",
    "JobSettings",
    "MLSettings",
    "StorageSettings",
    "get_app_settings",
    "get_auth_settings",
    "get_device_settings",
    "get_infrastructure_settings",
    "get_job_settings",
    "get_email_settings",
    "get_inference_settings",
    "get_storage_settings",
    "reset_settings_caches",
]
