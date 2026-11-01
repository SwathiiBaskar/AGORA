import json
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import pytest

import civbox
from civbox.cli import EXIT_CONFIGURATION_ERROR, EXIT_OK, main


def test_version_prints_the_package_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["version"]) == EXIT_OK
    assert capsys.readouterr().out.strip() == civbox.__version__


def test_installed_metadata_matches_the_module_version() -> None:
    assert version("civbox") == civbox.__version__


def test_config_show_prints_effective_settings_as_json(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("APP_ENV", "test")
    assert main(["config", "show"]) == EXIT_OK
    assert json.loads(capsys.readouterr().out) == {"app_env": "test", "log_level": "INFO"}


def test_invalid_configuration_exits_3_with_a_safe_message(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOG_LEVEL", "sup3r-s3cret-value")
    assert main(["config", "show"]) == EXIT_CONFIGURATION_ERROR
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "LOG_LEVEL" in captured.err
    assert "sup3r-s3cret-value" not in captured.err


@pytest.mark.parametrize("argv", [[], ["config"], ["nonsense"]])
def test_usage_errors_exit_2(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        main(argv)
    assert caught.value.code == 2


def test_module_entry_point_runs_in_a_fresh_interpreter(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "civbox", "version"],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
        timeout=60,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == civbox.__version__
