from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from civbox.core.config import Settings, load_settings, settings_for_display
from civbox.core.errors import ConfigurationError


def test_defaults_apply_when_nothing_is_configured() -> None:
    settings = load_settings(env_file=None)
    assert settings.app_env == "development"
    assert settings.log_level == "INFO"


def test_environment_variables_are_read_and_normalised(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", " Production ")
    monkeypatch.setenv("LOG_LEVEL", "debug")
    settings = load_settings(env_file=None)
    assert settings.app_env == "production"
    assert settings.log_level == "DEBUG"


def test_invalid_values_fail_fast_naming_the_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "LOUD")
    with pytest.raises(ConfigurationError) as caught:
        load_settings(env_file=None)
    assert [name for name, _ in caught.value.problems] == ["LOG_LEVEL"]


def test_every_invalid_variable_is_reported_together(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "LOUD")
    monkeypatch.setenv("APP_ENV", "staging")
    with pytest.raises(ConfigurationError) as caught:
        load_settings(env_file=None)
    assert sorted(name for name, _ in caught.value.problems) == ["APP_ENV", "LOG_LEVEL"]


def test_rejected_values_are_never_echoed(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = "sup3r-s3cret-value"
    monkeypatch.setenv("LOG_LEVEL", sentinel)
    with pytest.raises(ConfigurationError) as caught:
        load_settings(env_file=None)
    assert sentinel not in str(caught.value)
    assert sentinel not in repr(caught.value.problems)
    assert caught.value.__suppress_context__


def test_dotenv_file_is_read(tmp_path: Path) -> None:
    dotenv = tmp_path / "custom.env"
    dotenv.write_text("APP_ENV=test\nLOG_LEVEL=warning\n", encoding="utf-8")
    settings = load_settings(env_file=dotenv)
    assert settings.app_env == "test"
    assert settings.log_level == "WARNING"


def test_dotenv_in_the_working_directory_is_the_default(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("LOG_LEVEL=error\n", encoding="utf-8")
    assert load_settings().log_level == "ERROR"


def test_real_environment_overrides_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dotenv = tmp_path / "custom.env"
    dotenv.write_text("LOG_LEVEL=error\n", encoding="utf-8")
    monkeypatch.setenv("LOG_LEVEL", "critical")
    assert load_settings(env_file=dotenv).log_level == "CRITICAL"


def test_unknown_dotenv_entries_are_ignored(tmp_path: Path) -> None:
    dotenv = tmp_path / "custom.env"
    dotenv.write_text("SOMETHING_FROM_A_FUTURE_COMMIT=1\nLOG_LEVEL=info\n", encoding="utf-8")
    assert load_settings(env_file=dotenv).log_level == "INFO"


def test_settings_are_immutable() -> None:
    settings = load_settings(env_file=None)
    with pytest.raises(ValidationError):
        settings.log_level = "DEBUG"  # type: ignore[misc]


def test_display_dump_redacts_secrets_and_credentials_in_urls() -> None:
    class WithSecrets(Settings):
        gemini_api_key: SecretStr = SecretStr("not-a-real-key")
        database_url: str = "postgresql://app:hunter2@db/civ"
        input_tokens: int = 7

    shown = settings_for_display(WithSecrets())
    assert shown["gemini_api_key"] == "[REDACTED]"
    assert "hunter2" not in str(shown["database_url"])
    assert shown["input_tokens"] == 7
    assert shown["app_env"] == "development"
