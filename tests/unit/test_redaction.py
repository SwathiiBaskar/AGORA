import time

import pytest
from pydantic import SecretStr

from civbox.core.redaction import (
    REDACTED,
    is_sensitive_key,
    redact,
    redact_string,
    redact_text,
    truncate,
)

# Built at runtime so no realistic-looking credential literal exists in the source.
FAKE_GOOGLE_KEY = "AI" + "za" + "A" * 35
FAKE_JWT = ".".join(["ey" + "J" + "h" * 10, "ey" + "J" + "p" * 10, "s" * 12])


@pytest.mark.parametrize(
    "name",
    [
        "password",
        "PASSWORD",
        "db_password",
        "GEMINI_API_KEY",
        "apiKey",
        "x-api-key",
        "Authorization",
        "Set-Cookie",
        "access_token",
        "refreshToken",
        "csrf_token",
        "token",
        "ARTIFACT_HMAC_KEY",
        "client_secret",
        "private_key",
        "jwt_secret",
    ],
)
def test_credential_names_are_sensitive(name: str) -> None:
    assert is_sensitive_key(name)


@pytest.mark.parametrize(
    "name",
    [
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "max_tokens",
        "idempotency_key",
        "cache_key",
        "primary_key",
        "request_id",
        "simulation_id",
        "log_level",
        "tokenizer",
        "monkey",
        "",
    ],
)
def test_ordinary_names_are_not_sensitive(name: str) -> None:
    assert not is_sensitive_key(name)


def test_url_credentials_are_masked_but_host_and_user_survive() -> None:
    text = "connecting to postgresql://app:hunter2@db.internal:5432/civ"
    out = redact_text(text)
    assert "hunter2" not in out
    assert "postgresql://app:" in out
    assert "@db.internal:5432/civ" in out


def test_url_with_empty_user_is_masked() -> None:
    assert "s3cr3t" not in redact_text("redis://:s3cr3t@cache:6379/0")


@pytest.mark.parametrize(
    "text",
    ["https://example.org/a:b/c", "http://host:8080/path", "see https://example.org:443/x"],
)
def test_urls_without_credentials_are_unchanged(text: str) -> None:
    assert redact_text(text) == text


def test_bearer_token_is_masked() -> None:
    out = redact_text("Authorization: Bearer abcdefghijklmnop")
    assert "abcdefghijklmnop" not in out
    assert REDACTED in out


def test_jwt_is_masked() -> None:
    out = redact_text(f"cookie value {FAKE_JWT} end")
    assert FAKE_JWT not in out
    assert "end" in out


def test_google_api_key_is_masked() -> None:
    out = redact_text(f"key is {FAKE_GOOGLE_KEY}.")
    assert FAKE_GOOGLE_KEY not in out


@pytest.mark.parametrize(
    ("text", "secret"),
    [
        ("password=hunter2 user=bob", "hunter2"),
        ("PASSWORD=hunter2", "hunter2"),
        ('api_key: "abc 123 def"', "abc 123 def"),
        ("{'client_secret': 'xyz987'}", "xyz987"),
        ('{"password": "hunter2"}', "hunter2"),
        ("access_token=zzz111", "zzz111"),
        ("token=zzz111", "zzz111"),
    ],
)
def test_credential_assignments_are_masked(text: str, secret: str) -> None:
    assert secret not in redact_text(text)


def test_assignment_keeps_unrelated_fields() -> None:
    assert redact_text("password=hunter2 user=bob") == f"password={REDACTED} user=bob"


def test_token_counts_are_not_mistaken_for_credentials() -> None:
    text = "input_tokens=100 output_tokens=50 total_tokens=150 max_tokens: 900"
    assert redact_text(text) == text


def test_text_redaction_is_idempotent() -> None:
    text = f"postgresql://a:b@h/d password=x Bearer abcdefghijkl {FAKE_JWT}"
    once = redact_text(text)
    assert redact_text(once) == once


@pytest.mark.parametrize(
    "payload",
    ["a" * 50_000, "a-" * 25_000, "password" * 5_000, "token " * 10_000, "a://" * 12_000],
)
def test_adversarial_long_input_is_processed_in_bounded_time(payload: str) -> None:
    started = time.perf_counter()
    redact_text(payload)
    assert time.perf_counter() - started < 5.0


def test_truncate_reports_how_much_was_dropped() -> None:
    assert truncate("abcdef", 4) == "abcd...[truncated 2 chars]"
    assert truncate("abc", 4) == "abc"


def test_secret_is_redacted_before_truncation_can_split_it() -> None:
    text = "x" * 10 + " password=" + "s" * 100
    out = redact_string(text, 25)
    assert "s" * 5 not in out
    assert out.startswith("x" * 10)


def test_redact_masks_sensitive_keys_at_any_depth() -> None:
    value = {
        "user": "bob",
        "auth": {"password": "hunter2", "note": "ok"},
        "items": [{"api_key": "k1"}, {"safe": 1}],
        "input_tokens": 12,
    }
    assert redact(value) == {
        "user": "bob",
        "auth": {"password": REDACTED, "note": "ok"},
        "items": [{"api_key": REDACTED}, {"safe": 1}],
        "input_tokens": 12,
    }


def test_redact_does_not_mutate_its_input() -> None:
    value = {"password": "hunter2", "nested": {"token": "t"}}
    redact(value)
    assert value == {"password": "hunter2", "nested": {"token": "t"}}


def test_redact_masks_pydantic_secrets() -> None:
    assert redact({"anything": SecretStr("hunter2")}) == {"anything": REDACTED}


def test_redact_scrubs_secret_shapes_inside_ordinary_string_values() -> None:
    out = redact({"url": "postgresql://a:hunter2@h/d"})
    assert isinstance(out, dict)
    assert "hunter2" not in str(out)


def test_redact_summarises_bytes_instead_of_copying_them() -> None:
    assert redact(b"\x00" * 1000) == "<1000 bytes>"


def test_redact_makes_non_finite_floats_json_safe() -> None:
    assert redact([float("nan"), float("inf"), 1.5]) == ["nan", "inf", 1.5]


def test_redact_truncates_long_strings() -> None:
    out = redact("y" * 50, max_string=10)
    assert out == "y" * 10 + "...[truncated 40 chars]"


def test_redact_cuts_oversized_collections_with_a_marker() -> None:
    assert redact(list(range(5)), max_items=3) == [0, 1, 2, "[+2 more items]"]
    out = redact({str(i): i for i in range(5)}, max_items=3)
    assert out == {"0": 0, "1": 1, "2": 2, "_truncated": "2 more items"}


def test_redact_survives_self_referencing_structures() -> None:
    loop: dict[str, object] = {}
    loop["me"] = loop
    out = redact(loop, max_depth=4)
    assert isinstance(out, dict)


def test_redact_orders_sets_deterministically() -> None:
    assert redact({3, 1, 2}) == [1, 2, 3]
