import json
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
import uvicorn
from fastapi import FastAPI

from civbox.api.server import build_server_config
from civbox.cli import EXIT_CONFIGURATION_ERROR, EXIT_OK, main


@pytest.fixture
def served(
    monkeypatch: pytest.MonkeyPatch, restored_root_logger: None
) -> Iterator[list[uvicorn.Config]]:
    """Replace the blocking server loop and record the configuration it was given."""
    configs: list[uvicorn.Config] = []

    def record(self: uvicorn.Server) -> None:
        configs.append(self.config)

    monkeypatch.setattr(uvicorn.Server, "run", record)
    server_logger = logging.getLogger("uvicorn.error")
    filters = list(server_logger.filters)
    yield configs
    server_logger.filters[:] = filters


def test_serve_defaults_to_loopback_on_port_8000(served: list[uvicorn.Config]) -> None:
    assert main(["serve"]) == EXIT_OK
    (config,) = served
    assert (config.host, config.port) == ("127.0.0.1", 8000)
    assert isinstance(config.app, FastAPI)


def test_serve_passes_host_and_port_through(served: list[uvicorn.Config]) -> None:
    assert main(["serve", "--host", "127.0.0.2", "--port", "9001"]) == EXIT_OK
    (config,) = served
    assert (config.host, config.port) == ("127.0.0.2", 9001)


def test_serve_routes_server_logging_through_the_application(
    served: list[uvicorn.Config],
) -> None:
    main(["serve"])
    (config,) = served
    assert config.log_config is None
    assert config.access_log is False
    assert config.server_header is False


def test_serve_follows_the_configured_log_level(
    served: list[uvicorn.Config], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOG_LEVEL", "warning")
    main(["serve"])
    (config,) = served
    assert config.log_level == "warning"
    assert logging.getLogger().level == logging.WARNING


def test_serve_installs_json_logging_for_the_api_service(
    served: list[uvicorn.Config],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("APP_ENV", "test")
    main(["serve"])
    logging.getLogger("civbox.demo").info("hello")
    entry = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert (entry["service"], entry["environment"], entry["message"]) == (
        "civbox-api",
        "test",
        "hello",
    )


def test_serve_strips_uvicorn_colour_duplicates_from_structured_logs(
    served: list[uvicorn.Config], capsys: pytest.CaptureFixture[str]
) -> None:
    main(["serve"])
    logging.getLogger("uvicorn.error").info(
        "Started server process [%d]", 7, extra={"color_message": "\x1b[36m%d\x1b[0m"}
    )
    entry = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert entry["message"] == "Started server process [7]"
    assert "color_message" not in entry
    assert "\x1b" not in json.dumps(entry)


def test_serve_can_be_called_repeatedly_without_stacking_filters(
    served: list[uvicorn.Config],
) -> None:
    main(["serve"])
    main(["serve"])
    assert len(logging.getLogger("uvicorn.error").filters) == 1


def test_serve_enables_hsts_only_in_production(
    served: list[uvicorn.Config], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    main(["serve"])
    app = served[0].app
    assert isinstance(app, FastAPI)
    assert any(m.kwargs.get("hsts") is True for m in app.user_middleware)


@pytest.mark.parametrize("port", ["0", "65536", "-1", "http"])
def test_serve_rejects_invalid_ports(port: str, served: list[uvicorn.Config]) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["serve", "--port", port])
    assert caught.value.code == 2
    assert served == []


def test_serve_refuses_to_start_with_invalid_configuration(
    served: list[uvicorn.Config],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("LOG_LEVEL", "LOUD")
    assert main(["serve"]) == EXIT_CONFIGURATION_ERROR
    assert served == []
    assert "LOG_LEVEL" in capsys.readouterr().err


def test_server_config_binds_what_it_is_given() -> None:
    app = FastAPI()
    config = build_server_config(app, host="127.0.0.1", port=1234, log_level="DEBUG")
    assert config.app is app
    assert (config.host, config.port, config.log_level) == ("127.0.0.1", 1234, "debug")


def test_openapi_command_prints_the_document(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["openapi"]) == EXIT_OK
    document = json.loads(capsys.readouterr().out)
    assert document["info"]["title"] == "Civilization in a Box API"
    assert {"/healthz", "/readyz"} <= set(document["paths"])


def test_openapi_output_is_deterministic(capsys: pytest.CaptureFixture[str]) -> None:
    main(["openapi"])
    first = capsys.readouterr().out
    main(["openapi"])
    assert capsys.readouterr().out == first


def test_openapi_can_be_written_to_a_file_with_lf_line_endings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "openapi.json"
    assert main(["openapi", "--output", str(target)]) == EXIT_OK
    raw = target.read_bytes()
    assert raw.endswith(b"}\n")
    assert b"\r" not in raw
    assert json.loads(raw)["openapi"].startswith("3.")
    assert capsys.readouterr().out == ""


def test_openapi_file_and_stdout_hold_the_same_document(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "openapi.json"
    main(["openapi", "--output", str(target)])
    main(["openapi"])
    assert target.read_text(encoding="utf-8") == capsys.readouterr().out
