"""Email delivery settings (Resend, sender identity, sign-in notices)."""

from pydantic import Field
from pydantic_settings import BaseSettings

from backend.core.settings._cache import settings_cache
from backend.core.settings._env import env_settings_config


class EmailSettings(BaseSettings):
    model_config = env_settings_config()

    #: Resend API key. Absent until the operator creates one, so the platform
    #: must treat a missing key as "log instead of send", never as a boot error.
    resend_api_key: str | None = Field(default=None, alias="RESEND_API_KEY")
    email_from: str = Field(default="Nomikos <no-reply@nomikos.app>", alias="EMAIL_FROM")
    #: Where replies to the welcome mail go. None means the mail invites no reply.
    reply_to: str | None = Field(default=None, alias="EMAIL_REPLY_TO")
    #: The app origin (not the landing site): mail links and the hosted logo live here.
    app_public_url: str = Field(default="https://app.nomikos.app", alias="APP_PUBLIC_URL")
    signin_notices: bool = Field(default=True, alias="EMAIL_SIGNIN_NOTICES")

    @property
    def enabled(self) -> bool:
        """True only when a key exists to send with. The only switch that matters."""
        return bool(self.resend_api_key)


@settings_cache
def get_email_settings() -> EmailSettings:
    return EmailSettings()
