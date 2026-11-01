from civbox.core.errors import CivboxError, ConfigurationError


def test_configuration_error_lists_every_problem() -> None:
    error = ConfigurationError([("LOG_LEVEL", "bad level"), ("APP_ENV", "bad env")])
    assert error.problems == (("LOG_LEVEL", "bad level"), ("APP_ENV", "bad env"))
    assert str(error) == "LOG_LEVEL: bad level; APP_ENV: bad env"


def test_errors_have_stable_distinct_codes() -> None:
    assert CivboxError.code == "internal_error"
    assert ConfigurationError.code == "configuration_error"
    assert issubclass(ConfigurationError, CivboxError)
