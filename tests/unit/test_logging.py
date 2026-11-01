import asyncio
import io
import json
import logging
import re
from collections.abc import Iterator

import pytest

from civbox.core.logging import (
    CORRELATION_FIELDS,
    JsonFormatter,
    configure_logging,
    current_context,
    log_context,
)


def make_record(
    message: str = "hello",
    *,
    level: int = logging.INFO,
    args: tuple[object, ...] = (),
    exc_info: tuple[type[BaseException], BaseException, object] | None = None,
    **extra: object,
) -> logging.LogRecord:
    record = logging.LogRecord("civbox.test", level, __file__, 1, message, args, exc_info)  # type: ignore[arg-type]
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def render(record: logging.LogRecord) -> dict[str, object]:
    line = JsonFormatter(service="svc", environment="test").format(record)
    assert "\n" not in line
    parsed = json.loads(line)
    assert isinstance(parsed, dict)
    return parsed


@pytest.fixture
def restored_root_logger() -> Iterator[None]:
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


def test_record_has_the_base_fields() -> None:
    out = render(make_record("hello %s", args=("world",)))
    assert out["message"] == "hello world"
    assert out["level"] == "INFO"
    assert out["logger"] == "civbox.test"
    assert out["service"] == "svc"
    assert out["environment"] == "test"
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z", str(out["timestamp"]))


def test_no_correlation_fields_when_none_are_bound() -> None:
    out = render(make_record())
    assert not set(CORRELATION_FIELDS) & set(out)


def test_bound_correlation_fields_appear_and_are_restored() -> None:
    with log_context(request_id="r1", job_id="j1"):
        assert dict(current_context()) == {"request_id": "r1", "job_id": "j1"}
        out = render(make_record())
        assert out["request_id"] == "r1"
        assert out["job_id"] == "j1"
    assert dict(current_context()) == {}


def test_contexts_nest_and_inner_values_override() -> None:
    with log_context(request_id="outer", trace_id="t"), log_context(request_id="inner"):
        assert dict(current_context()) == {"request_id": "inner", "trace_id": "t"}
    assert dict(current_context()) == {}


def test_none_leaves_an_outer_value_unchanged() -> None:
    with log_context(request_id="r1"), log_context(request_id=None, job_id="j"):
        assert dict(current_context()) == {"request_id": "r1", "job_id": "j"}


def test_context_is_restored_even_when_the_block_raises() -> None:
    with pytest.raises(RuntimeError), log_context(request_id="r1"):
        raise RuntimeError("boom")
    assert dict(current_context()) == {}


def test_concurrent_asyncio_tasks_do_not_see_each_others_context() -> None:
    async def worker(name: str) -> str:
        with log_context(request_id=name):
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            return current_context()["request_id"]

    async def run() -> list[str]:
        return list(await asyncio.gather(*(worker(f"req-{i}") for i in range(20))))

    assert asyncio.run(run()) == [f"req-{i}" for i in range(20)]


def test_extra_fields_are_included() -> None:
    out = render(make_record(simulation_tick=7, population=500))
    assert out["simulation_tick"] == 7
    assert out["population"] == 500


def test_extra_fields_never_overwrite_base_fields() -> None:
    out = render(make_record(service="spoofed", timestamp="never"))
    assert out["service"] == "svc"
    assert out["extra_service"] == "spoofed"
    assert out["extra_timestamp"] == "never"


def test_sensitive_extra_fields_are_redacted() -> None:
    out = render(make_record(api_key="k-123", input_tokens=40, headers={"Authorization": "x"}))
    assert out["api_key"] == "[REDACTED]"
    assert out["input_tokens"] == 40
    assert out["headers"] == {"Authorization": "[REDACTED]"}


def test_secrets_inside_the_message_are_redacted() -> None:
    out = render(make_record("connecting to %s", args=("postgresql://app:hunter2@db/x",)))
    assert "hunter2" not in str(out["message"])


def test_huge_messages_are_truncated() -> None:
    out = render(make_record("p" * 100_000))
    assert len(str(out["message"])) < 2100
    assert "truncated" in str(out["message"])


def test_exceptions_are_serialised_with_redaction() -> None:
    try:
        raise ValueError("bad config postgresql://app:hunter2@db/x")
    except ValueError as exc:
        record = make_record(
            "failed", level=logging.ERROR, exc_info=(type(exc), exc, exc.__traceback__)
        )
    out = render(record)
    exception = out["exception"]
    assert isinstance(exception, dict)
    assert exception["type"] == "ValueError"
    assert "hunter2" not in json.dumps(out)
    assert "Traceback" in str(exception["traceback"])


def test_unserialisable_extra_values_do_not_break_formatting() -> None:
    out = render(make_record(thing=object()))
    assert isinstance(out["thing"], str)


def test_non_ascii_text_is_preserved() -> None:
    assert render(make_record("naïve — 数据"))["message"] == "naïve — 数据"


@pytest.mark.usefixtures("restored_root_logger")
def test_configure_logging_writes_json_to_the_given_stream() -> None:
    stream = io.StringIO()
    configure_logging(service="civbox-test", environment="test", level="info", stream=stream)
    with log_context(request_id="r9"):
        logging.getLogger("civbox.demo").info("started", extra={"count": 3})
    entry = json.loads(stream.getvalue().strip())
    assert entry["service"] == "civbox-test"
    assert entry["request_id"] == "r9"
    assert entry["count"] == 3


@pytest.mark.usefixtures("restored_root_logger")
def test_configure_logging_is_idempotent() -> None:
    first, second = io.StringIO(), io.StringIO()
    configure_logging(service="s", environment="test", stream=first)
    configure_logging(service="s", environment="test", stream=second)
    logging.getLogger("civbox.demo").warning("once")
    assert first.getvalue() == ""
    assert len(second.getvalue().strip().splitlines()) == 1


@pytest.mark.usefixtures("restored_root_logger")
def test_configure_logging_applies_the_level() -> None:
    stream = io.StringIO()
    configure_logging(service="s", environment="test", level="WARNING", stream=stream)
    logging.getLogger("civbox.demo").info("hidden")
    logging.getLogger("civbox.demo").warning("shown")
    assert [json.loads(line)["message"] for line in stream.getvalue().splitlines()] == ["shown"]


@pytest.mark.usefixtures("restored_root_logger")
def test_configure_logging_rejects_unknown_levels() -> None:
    with pytest.raises(ValueError, match="unknown log level"):
        configure_logging(service="s", environment="test", level="LOUD")
