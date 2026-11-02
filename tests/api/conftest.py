import io
import json
import logging
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi import FastAPI

from civbox.api.app import create_app
from civbox.core.config import Settings, load_settings
from civbox.core.logging import configure_logging

from .support import AppClient, LogEntries


@pytest.fixture
def settings() -> Settings:
    return load_settings(env_file=None)


@pytest.fixture
def app(settings: Settings) -> FastAPI:
    return create_app(settings)


@pytest.fixture
def client(app: FastAPI) -> AppClient:
    return AppClient(app)


@pytest.fixture
def restored_root_logger() -> Iterator[None]:
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


@pytest.fixture
def json_logs(restored_root_logger: None) -> LogEntries:
    """Install the application's JSON log handler and expose what it has written."""
    stream = io.StringIO()
    configure_logging(service="civbox-test", environment="test", level="DEBUG", stream=stream)

    def entries() -> list[dict[str, Any]]:
        return [json.loads(line) for line in stream.getvalue().splitlines() if line]

    return entries
