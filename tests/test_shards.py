"""CI sharding (tests/conftest.py): every module lands in exactly one shard, loads are balanced."""

from __future__ import annotations

import json

import pytest

from tests.conftest import DURATIONS, _parse_shard, assign_shards


def test_longest_first_into_least_loaded_shard() -> None:
    seconds = {"a": 10.0, "b": 7.0, "c": 6.0, "d": 4.0, "e": 3.0}
    assert assign_shards(seconds, 2) == {"a": 0, "b": 1, "c": 1, "d": 0, "e": 1}


def test_assignment_is_deterministic_under_ties() -> None:
    seconds = {name: 1.0 for name in "edcba"}
    assert assign_shards(seconds, 2) == assign_shards(dict(sorted(seconds.items())), 2)
    assert assign_shards(seconds, 2) == {"a": 0, "b": 1, "c": 0, "d": 1, "e": 0}


def test_committed_durations_split_into_four_balanced_shards() -> None:
    seconds = json.loads(DURATIONS.read_text(encoding="utf-8"))
    shard_of = assign_shards(seconds, 4)
    assert set(shard_of) == set(seconds)
    load = [sum(s for g, s in seconds.items() if shard_of[g] == k) for k in range(4)]
    # Greedy packing is within the largest module of the ideal split.
    assert max(load) - sum(load) / 4 <= max(seconds.values())


@pytest.mark.parametrize("value", ["3", "0/4", "5/4", "a/b"])
def test_bad_shard_option(value: str) -> None:
    with pytest.raises(pytest.UsageError):
        _parse_shard(value)


def test_good_shard_option() -> None:
    assert _parse_shard("2/4") == (2, 4)
