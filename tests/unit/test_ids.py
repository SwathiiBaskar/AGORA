import re

import pytest

from civbox.core.ids import is_valid_request_id, new_request_id


def test_new_request_ids_have_a_fixed_shape() -> None:
    assert re.fullmatch(r"req_[0-9a-f]{32}", new_request_id())


def test_new_request_ids_do_not_repeat() -> None:
    assert len({new_request_id() for _ in range(1000)}) == 1000


def test_generated_ids_are_accepted_by_the_validator() -> None:
    assert is_valid_request_id(new_request_id())


@pytest.mark.parametrize("value", ["abcdefgh", "trace-1234.AB_cd", "a" * 128])
def test_reasonable_identifiers_are_accepted(value: str) -> None:
    assert is_valid_request_id(value)


@pytest.mark.parametrize(
    "value",
    [
        "",
        "short",
        "a" * 129,
        "has space",
        "tab\there!",
        "line\nbreak",
        "trailing-newline\n",
        "<script>alert(1)</script>",
        "semi;colon:colon",
        "unicode-\u00e9\u00e9\u00e9",
    ],
)
def test_unsafe_or_malformed_identifiers_are_rejected(value: str) -> None:
    assert not is_valid_request_id(value)
