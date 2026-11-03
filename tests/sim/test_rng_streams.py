import hashlib
import json
from typing import Any

import numpy as np
import pytest

from civbox.core.canonical import canonical_json
from civbox.sim.rng import RngStreams

# Raw 64-bit PCG64 output is defined by the algorithm, so these values do not depend on
# how NumPy turns bits into floats. They pin the stream derivation: if they change, every
# stored result that depended on a seed is no longer reproducible.
GOLDEN_RAW = {
    (12345, "population"): [
        18294980588097128739,
        1515662557983894634,
        5423311833320718098,
        17102482846115795022,
    ],
    (12345, "events"): [
        10750387276058827108,
        5892078300058683739,
        12496020672963021346,
        8080009436368993899,
    ],
    (0, "graph"): [14192318918848398026, 9848755573338669241, 4786005457916304520],
}


def raw(streams: RngStreams, name: str, count: int = 4) -> list[int]:
    return [int(v) for v in streams.generator(name).bit_generator.random_raw(count)]


@pytest.mark.parametrize(("seed", "name"), list(GOLDEN_RAW))
def test_streams_reproduce_pinned_output(seed: int, name: str) -> None:
    expected = GOLDEN_RAW[(seed, name)]
    assert raw(RngStreams(seed), name, len(expected)) == expected


def test_the_derivation_matches_its_documented_definition() -> None:
    digest = hashlib.sha256(b"civbox/rng/v1/population").digest()
    key = int.from_bytes(digest[:16], "big")
    sequence = np.random.SeedSequence(entropy=12345, spawn_key=(key,))
    reference = np.random.Generator(np.random.PCG64(sequence))
    assert raw(RngStreams(12345), "population") == [
        int(v) for v in reference.bit_generator.random_raw(4)
    ]


def test_the_same_seed_and_name_always_give_the_same_sequence() -> None:
    assert raw(RngStreams(5), "x") == raw(RngStreams(5), "x")


def test_different_names_give_different_sequences() -> None:
    streams = RngStreams(5)
    assert raw(streams, "graph") != raw(streams, "population")


def test_different_seeds_give_different_sequences() -> None:
    assert raw(RngStreams(5), "graph") != raw(RngStreams(6), "graph")


def test_a_stream_does_not_depend_on_the_order_streams_were_created() -> None:
    forward, backward = RngStreams(9), RngStreams(9)
    for name in ("a", "b", "c"):
        forward.generator(name)
    for name in ("c", "extra", "b", "a"):
        backward.generator(name)
    for name in ("a", "b", "c"):
        assert raw(forward, name) == raw(backward, name)


def test_drawing_from_one_stream_never_changes_another() -> None:
    untouched = RngStreams(3)
    busy = RngStreams(3)
    busy.generator("noise").random(10_000)
    assert raw(busy, "signal") == raw(untouched, "signal")


def test_the_same_generator_object_is_returned_each_time() -> None:
    streams = RngStreams(1)
    assert streams.generator("a") is streams.generator("a")


def test_streams_are_statistically_unrelated() -> None:
    streams = RngStreams(2026)
    a = streams.generator("a").random(20_000)
    b = streams.generator("b").random(20_000)
    assert abs(float(np.corrcoef(a, b)[0, 1])) < 0.03


@pytest.mark.parametrize(
    "name", ["", "Upper", "1abc", "has-dash", "has space", "x" * 65, "caf\u00e9"]
)
def test_invalid_stream_names_are_rejected(name: str) -> None:
    with pytest.raises(ValueError, match="invalid stream name"):
        RngStreams(1).generator(name)


def test_stream_names_up_to_64_characters_are_accepted() -> None:
    RngStreams(1).generator("a" + "b" * 63)


@pytest.mark.parametrize("seed", [-1, 2**128])
def test_out_of_range_seeds_are_rejected(seed: int) -> None:
    with pytest.raises(ValueError, match="seed must satisfy"):
        RngStreams(seed)


@pytest.mark.parametrize("seed", [True, 1.0, "1", None])
def test_non_integer_seeds_are_rejected(seed: Any) -> None:
    with pytest.raises(TypeError, match="seed must be an int"):
        RngStreams(seed)


def test_the_largest_seed_is_accepted() -> None:
    assert len(raw(RngStreams(2**128 - 1), "x")) == 4


# --- snapshots ---------------------------------------------------------------


def test_names_lists_created_streams_sorted() -> None:
    streams = RngStreams(1)
    for name in ("zeta", "alpha", "mid"):
        streams.generator(name)
    assert streams.names == ("alpha", "mid", "zeta")


def test_a_snapshot_is_canonical_json_with_big_integers_as_strings() -> None:
    streams = RngStreams(1)
    streams.generator("a").random(3)
    snapshot = streams.snapshot()
    canonical_json(snapshot)  # plain JSON data only
    entry = snapshot["a"]
    assert isinstance(entry["state"], str)
    assert isinstance(entry["increment"], str)
    assert entry["bit_generator"] == "PCG64"


def test_a_stream_restored_from_a_snapshot_continues_exactly_where_it_stopped() -> None:
    original = RngStreams(77)
    for name in ("graph", "events", "exposure"):
        original.generator(name).random(13)
    original.generator("events").integers(0, 10, size=3, dtype=np.uint32)  # buffered 32-bit draw

    restored = RngStreams.from_snapshot(77, original.snapshot())

    for name in ("graph", "events", "exposure"):
        a, b = original.generator(name), restored.generator(name)
        assert a.random(50).tolist() == b.random(50).tolist()
        assert a.normal(size=50).tolist() == b.normal(size=50).tolist()
        assert a.integers(0, 1000, size=50).tolist() == b.integers(0, 1000, size=50).tolist()
        assert a.exponential(size=20).tolist() == b.exponential(size=20).tolist()
        assert a.permutation(30).tolist() == b.permutation(30).tolist()
        p = [0.1, 0.2, 0.7]
        assert a.choice(3, size=40, p=p).tolist() == b.choice(3, size=40, p=p).tolist()


def test_the_buffered_32_bit_state_is_captured() -> None:
    streams = RngStreams(4)
    streams.generator("x").integers(0, 10, size=3, dtype=np.uint32)
    assert streams.snapshot()["x"]["has_uint32"] == 1


def test_a_snapshot_survives_a_json_round_trip() -> None:
    original = RngStreams(8)
    original.generator("a").random(5)
    text = json.dumps(original.snapshot())
    restored = RngStreams.from_snapshot(8, json.loads(text))
    assert raw(original, "a") == raw(restored, "a")


def test_snapshots_of_equal_states_are_equal() -> None:
    first, second = RngStreams(1), RngStreams(1)
    for streams in (first, second):
        streams.generator("a").random(9)
    assert first.snapshot() == second.snapshot()


def test_streams_missing_from_a_snapshot_start_fresh() -> None:
    original = RngStreams(6)
    original.generator("a").random(5)
    restored = RngStreams.from_snapshot(6, original.snapshot())
    assert raw(restored, "b") == raw(RngStreams(6), "b")


def test_an_empty_snapshot_restores_an_unused_set_of_streams() -> None:
    assert RngStreams.from_snapshot(3, {}).names == ()


def valid_entry() -> dict[str, Any]:
    streams = RngStreams(1)
    streams.generator("a").random(2)
    entry: dict[str, Any] = dict(streams.snapshot()["a"])
    return entry


def with_change(**changes: Any) -> dict[str, Any]:
    entry = valid_entry()
    entry.update(changes)
    return entry


MALFORMED: list[tuple[str, object]] = [
    ("not an object", "text"),
    ("missing key", {k: v for k, v in valid_entry().items() if k != "increment"}),
    ("extra key", with_change(extra=1)),
    ("other bit generator", with_change(bit_generator="MT19937")),
    ("state is an int", with_change(state=5)),
    ("state is negative", with_change(state="-1")),
    ("state has a sign", with_change(state="+5")),
    ("state has underscores", with_change(state="1_000")),
    ("state is not digits", with_change(state="abc")),
    ("state has non-ASCII digits", with_change(state="\u0661\u0662\u0663")),
    ("state is empty", with_change(state="")),
    ("state has 40 digits", with_change(state="1" * 40)),
    ("state is 2**128", with_change(state=str(2**128))),
    ("increment is even", with_change(increment="2")),
    ("increment is zero", with_change(increment="0")),
    ("has_uint32 is 2", with_change(has_uint32=2)),
    ("has_uint32 is a bool", with_change(has_uint32=True)),
    ("has_uint32 is a string", with_change(has_uint32="0")),
    ("uinteger is negative", with_change(uinteger=-1)),
    ("uinteger is 2**32", with_change(uinteger=2**32)),
    ("uinteger is a float", with_change(uinteger=1.5)),
]


@pytest.mark.parametrize("entry", [e for _, e in MALFORMED], ids=[label for label, _ in MALFORMED])
def test_malformed_snapshots_are_rejected_and_name_the_stream(entry: object) -> None:
    with pytest.raises(ValueError, match="stream 'a'"):
        RngStreams.from_snapshot(1, {"a": entry})


def test_a_snapshot_naming_an_invalid_stream_is_rejected() -> None:
    with pytest.raises(ValueError, match="invalid stream name"):
        RngStreams.from_snapshot(1, {"Bad-Name": valid_entry()})


def test_the_largest_valid_increment_and_state_are_accepted() -> None:
    entry = with_change(state=str(2**128 - 1), increment=str(2**128 - 1))
    restored = RngStreams.from_snapshot(1, {"a": entry})
    assert restored.snapshot()["a"]["state"] == str(2**128 - 1)
