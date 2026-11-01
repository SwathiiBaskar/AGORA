"""Process configuration.

Settings are read from environment variables (and, for local development, a
``.env`` file in the working directory), validated once, and then immutable.
There is deliberately no module-level singleton: the entry point loads settings
with :func:`load_settings` and passes them to whatever needs them.

Environment variable names are the upper-cased field names. Real environment
variables take precedence over ``.env``.
"""

from pathlib import Path
from typing import Literal, cast

from pydantic import ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from civbox.core.errors import ConfigurationError
from civbox.core.redaction import redact

AppEnv = Literal["development", "test", "production"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseSettings):
    """Validated process settings. Instances are immutable."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    app_env: AppEnv = "development"
    log_level: LogLevel = "INFO"

    @field_validator("app_env", mode="before")
    @classmethod
    def _normalise_app_env(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalise_log_level(cls, value: object) -> object:
        return value.strip().upper() if isinstance(value, str) else value


def load_settings(*, env_file: str | Path | None = ".env") -> Settings:
    """Load and validate settings, failing fast with a safe error.

    ``env_file`` may be ``None`` to ignore dotenv files entirely. On invalid input a
    :class:`ConfigurationError` is raised naming each offending variable and the
    reason, but never the value supplied, since a rejected value can be a credential.
    """
    try:
        return Settings(_env_file=env_file)
    except ValidationError as exc:
        problems = [
            (".".join(str(part) for part in error["loc"]).upper() or "SETTINGS", error["msg"])
            for error in exc.errors(include_input=False, include_url=False, include_context=False)
        ]
        # ``from None`` keeps the original error, which embeds input values, out of tracebacks.
        raise ConfigurationError(problems) from None


def settings_for_display(settings: BaseSettings) -> dict[str, object]:
    """Return the settings as a redacted dict that is safe to print or log."""
    # redact() maps a dict to a dict; the cast records that for the type checker.
    return cast(dict[str, object], redact(settings.model_dump(mode="python")))
