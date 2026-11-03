import hashlib
import random
from typing import Any

import numpy as np
import pytest

from civbox.core.canonical import MAX_DEPTH, canonical_hash, canonical_json


def test_output_is_sorted_compact_ascii_json() -> None:
    value = {"b": 1, "a": [1, 2.5, None, True, "\u00e9"]}
    assert canonical_json(value) == '{"a":[1,2.5,null,true,"\\u00e9"],"b":1}'


def test_key_insertion_order_does_not_matter() -> None:
    keys = [f"k{i}" for i in range(50)]
    reference = canonical_json({k: i for i, k in enumerate(keys)})
    shuffler = random.Random(7)  # noqa: S311 - not security related
    for _ in range(20):
        shuffler.shuffle(keys)
        assert canonical_json({k: int(k[1:]) for k in keys}) == reference


def test_nested_objects_are_sorted_at_every_level() -> None:
    assert canonical_json({"z": {"b": 1, "a": 2}, "a": {"y": 0, "x": 0}}) == (
        '{"a":{"x":0,"y":0},"z":{"a":2,"b":1}}'
    )


def test_tuples_and_lists_are_the_same_array() -> None:
    assert canonical_json({"a": (1, 2)}) == canonical_json({"a": [1, 2]})


def test_integers_and_floats_are_distinct_values() -> None:
    assert canonical_json(1) != canonical_json(1.0)
    assert canonical_hash({"x": 1}) != canonical_hash({"x": 1.0})


def test_negative_zero_is_distinct_from_zero() -> None:
    assert canonical_json(-0.0) == "-0.0"
    assert canonical_json(0.0) == "0.0"


@pytest.mark.parametrize(
    ("value", "text"),
    [(0.1, "0.1"), (1e22, "1e+22"), (1e-7, "1e-07"), (123456789.125, "123456789.125")],
)
def test_floats_use_the_shortest_round_trip_form(value: float, text: str) -> None:
    assert canonical_json(value) == text
    assert float(canonical_json(value)) == value


def test_integers_are_written_exactly_even_beyond_double_precision() -> None:
    assert canonical_json(2**64 + 1) == "18446744073709551617"


def test_non_ascii_text_is_escaped_including_beyond_the_basic_plane() -> None:
    assert canonical_json("\U0001f600") == '"\\ud83d\\ude00"'


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_floats_are_rejected_with_their_location(bad: float) -> None:
    with pytest.raises(ValueError, match=r"\$\.a\[1\]\.b: non-finite float"):
        canonical_json({"a": [1, {"b": bad}]})


def test_non_string_keys_are_rejected_rather_than_coerced() -> None:
    with pytest.raises(TypeError, match=r"object key 1 is not a string"):
        canonical_json({1: "x"})


@pytest.mark.parametrize(
    "bad",
    [{1, 2}, b"bytes", np.int64(1), np.float32(1.5), np.bool_(True), object(), 1 + 2j],
)
def test_unsupported_types_are_rejected(bad: Any) -> None:
    with pytest.raises(TypeError, match=r"\$\.x: .* is not canonical-JSON serialisable"):
        canonical_json({"x": bad})


def test_numpy_float64_is_accepted_because_it_is_a_python_float() -> None:
    assert canonical_json({"x": np.float64(1.5)}) == '{"x":1.5}'


def _nested(levels: int) -> list[Any]:
    value: list[Any] = []
    for _ in range(levels - 1):
        value = [value]
    return value


def test_nesting_up_to_the_limit_is_accepted_and_beyond_it_rejected() -> None:
    canonical_json(_nested(MAX_DEPTH))
    with pytest.raises(ValueError, match="nested deeper than"):
        canonical_json(_nested(MAX_DEPTH + 1))


def test_hash_has_a_prefix_and_matches_an_independent_computation() -> None:
    value = {"a": 1, "b": [1, 2]}
    expected = hashlib.sha256(b'{"a":1,"b":[1,2]}').hexdigest()
    assert canonical_hash(value) == f"sha256:{expected}"


def test_hash_changes_with_content() -> None:
    assert canonical_hash({"a": 1}) != canonical_hash({"a": 2})
    assert canonical_hash({"a": 1}) != canonical_hash({"b": 1})
    assert canonical_hash([1, 2]) != canonical_hash([2, 1])
