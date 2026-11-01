from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_process_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep every test independent of the developer's shell and local ``.env`` file.

    Settings read ``.env`` from the working directory, so each test runs in an empty
    temporary directory with the application's variables unset.
    """
    for name in ("APP_ENV", "LOG_LEVEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
