"""Distance table, partition, samplers and suites."""

from pathlib import Path

import numpy as np
import pytest

from eqcube.data import (
    SPLITS,
    DistanceTable,
    Partition,
    PartitionConfig,
    RandomWalkSampler,
    TemperedSampler,
    apply_symmetry,
    default_suites,
    orbit_representatives,
    per_depth_suite,
    search_suite,
    uniform_suite,
)
from eqcube.data.partition import uniform_hash
from eqcube.data.samplers import _allocate  # pyright: ignore[reportPrivateUsage]
from eqcube.puzzles.pocket_coords import pocket_coords
from eqcube.puzzles.pocket_cube import task_group
from eqcube.symmetry.action import FunctionAction

C = pocket_coords()

# ── table ─────────────────────────────────────────────────────────────────


def test_table_cache_roundtrip(tmp_path: Path, table: DistanceTable) -> None:
    first = DistanceTable.build(C, cache_dir=tmp_path)
    assert (tmp_path / "pocket_cube_distances.npy").exists()
    second = DistanceTable.build(C, cache_dir=tmp_path)
    np.testing.assert_array_equal(first.dist, table.dist)
    np.testing.assert_array_equal(second.dist, table.dist)


def test_materialize_labels_match_sticker_lookup(table: DistanceTable) -> None:
    index = np.arange(0, C.n_states, 9973)
    states, labels = table.materialize(index)
    np.testing.assert_array_equal(table.distance(states), labels)


# ── symmetry on states ────────────────────────────────────────────────────


def test_apply_symmetry_matches_function_action() -> None:
    states = C.puzzle.scramble(50, 12, np.random.default_rng(0))
    action = FunctionAction(task_group(), n_domain=24)
    for g in task_group().elements:
        expected = np.array([action(g, tuple(int(c) for c in s)) for s in states], dtype=np.int8)
        np.testing.assert_array_equal(apply_symmetry(states, g, 24), expected)


@pytest.mark.slow
def test_task_symmetry_preserves_every_distance(table: DistanceTable, tmp_path: Path) -> None:
    """Exhaustive: all 3,674,160 states have the distance of their S3-orbit representative."""
    reps = orbit_representatives(C, task_group(), cache_path=tmp_path / "reps.npy")
    np.testing.assert_array_equal(table.dist[reps], table.dist)
    assert (reps <= np.arange(C.n_states)).all()
    sizes = np.bincount(np.unique(reps, return_counts=True)[1])
    assert int((sizes * np.arange(len(sizes))).sum()) == C.n_states
    assert set(np.flatnonzero(sizes).tolist()) <= {1, 2, 3, 6}  # divisors of |S3|
    orbit = Partition(table, PartitionConfig(policy="orbit"), orbit_reps=reps)
    np.testing.assert_array_equal(orbit.assignment[reps], orbit.assignment)
    np.testing.assert_array_equal(np.load(tmp_path / "reps.npy"), reps)


# ── partition ─────────────────────────────────────────────────────────────


def test_uniform_hash_is_deterministic_and_uniform() -> None:
    keys = np.arange(200_000)
    u = uniform_hash(keys, 7)
    np.testing.assert_array_equal(u, uniform_hash(keys, 7))
    assert not np.array_equal(u, uniform_hash(keys, 8))
    assert ((u >= 0) & (u < 1)).all()
    assert abs(u.mean() - 0.5) < 0.01


def test_splits_are_disjoint_except_shared(partition: Partition) -> None:
    masks = {s: partition.mask(s) for s in SPLITS}
    for a in SPLITS:
        for b in SPLITS:
            if a < b:
                np.testing.assert_array_equal(masks[a] & masks[b], partition.shared)
    assert (masks["train"] | masks["val"] | masks["test"]).all()


def test_split_fractions(partition: Partition) -> None:
    fractions = [partition.mask(s).mean() for s in SPLITS]
    np.testing.assert_allclose(fractions, [0.8, 0.1, 0.1], atol=0.005)


def test_shared_states_are_shallow(partition: Partition, table: DistanceTable) -> None:
    assert int(partition.shared.sum()) == int(table.counts[:5].sum()) == 688


def test_pools_are_sorted_by_depth(partition: Partition, table: DistanceTable) -> None:
    for d in range(table.max_distance + 1):
        pool = partition.pool("test", d)
        assert (table.dist[pool] == d).all()
    assert int(partition.pool_sizes("train").sum()) == int(partition.mask("train").sum())


def test_partition_depends_only_on_seed(table: DistanceTable) -> None:
    a = Partition(table, PartitionConfig(seed=3))
    b = Partition(table, PartitionConfig(seed=3))
    np.testing.assert_array_equal(a.assignment, b.assignment)


def test_orbit_policy_requires_representatives(table: DistanceTable) -> None:
    with pytest.raises(ValueError):
        Partition(table, PartitionConfig(policy="orbit"))


def test_bad_fractions_rejected() -> None:
    with pytest.raises(ValueError):
        PartitionConfig(fractions=(0.5, 0.5, 0.5))


# ── samplers ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_tempered_probabilities(partition: Partition, alpha: float) -> None:
    p = TemperedSampler(partition, alpha).depth_probabilities()
    sizes = partition.pool_sizes("train").astype(float)
    np.testing.assert_allclose(p, sizes**alpha / (sizes**alpha).sum())


def test_tempered_sample_follows_probabilities(
    partition: Partition, table: DistanceTable, rng: np.random.Generator
) -> None:
    sampler = TemperedSampler(partition, 0.0)
    index = sampler.sample(150_000, rng)
    assert partition.contains("train", index).all()
    observed = np.bincount(table.dist[index], minlength=15) / len(index)
    np.testing.assert_allclose(observed, sampler.depth_probabilities(), atol=0.005)


def test_sample_unique_is_distinct_and_in_split(
    partition: Partition, rng: np.random.Generator
) -> None:
    index = TemperedSampler(partition, 0.5).sample_unique(50_000, rng)
    assert len(index) == 50_000 == len(np.unique(index))
    assert partition.contains("train", index).all()


def test_sample_unique_takes_small_depths_whole(
    partition: Partition, table: DistanceTable, rng: np.random.Generator
) -> None:
    index = TemperedSampler(partition, 0.0).sample_unique(30_000, rng)
    counts = np.bincount(table.dist[index], minlength=15)
    sizes = partition.pool_sizes("train")
    assert (counts[:6] == sizes[:6]).all()  # tiny strata exhausted
    assert counts.sum() == 30_000


def test_sample_unique_rejects_too_many(partition: Partition, rng: np.random.Generator) -> None:
    with pytest.raises(ValueError):
        TemperedSampler(partition, 0.5, split="test").sample_unique(10**7, rng)


def test_allocate_respects_caps_and_total() -> None:
    quota = _allocate(100, np.array([0.5, 0.25, 0.25]), np.array([10, 1000, 1000]))
    assert quota.tolist()[0] == 10
    assert quota.sum() == 100
    assert abs(quota[1] - quota[2]) <= 1


def test_random_walk_sampler(partition: Partition, rng: np.random.Generator) -> None:
    index = RandomWalkSampler(partition, max_length=8).sample(5000, rng)
    assert len(index) == 5000
    assert partition.contains("train", index).all()
    assert partition.table.dist[index].max() <= 8


# ── suites ────────────────────────────────────────────────────────────────


def test_suites_come_from_their_split(partition: Partition, rng: np.random.Generator) -> None:
    for suite in [
        per_depth_suite(partition, 100, rng),
        uniform_suite(partition, 1000, rng),
        search_suite(partition, [5, 10], 50, rng),
    ]:
        assert partition.contains("test", suite.index).all()
        np.testing.assert_array_equal(suite.distance, partition.table.labels(suite.index))
        assert len(np.unique(suite.index)) == len(suite)


def test_per_depth_suite_caps_small_depths(partition: Partition, rng: np.random.Generator) -> None:
    suite = per_depth_suite(partition, 100, rng)
    counts = np.bincount(suite.distance, minlength=15)
    assert counts[0] == 1 and counts[1] == 6 and counts[10] == 100


def test_default_suites_are_reproducible(partition: Partition) -> None:
    a, b = default_suites(partition, seed=1), default_suites(partition, seed=1)
    assert a.keys() == b.keys()
    for name in a:
        np.testing.assert_array_equal(a[name].index, b[name].index)


def test_train_sampler_never_hits_heldout_states(
    partition: Partition, rng: np.random.Generator
) -> None:
    suites = default_suites(partition)
    train = TemperedSampler(partition, 0.5).sample_unique(200_000, rng)
    heldout = np.setdiff1d(suites["test/uniform"].index, np.flatnonzero(partition.shared))
    assert np.intersect1d(train, heldout).size == 0


def test_contains_agrees_with_mask(partition: Partition, rng: np.random.Generator) -> None:
    index = rng.integers(0, C.n_states, size=100_000)
    for split in SPLITS:
        np.testing.assert_array_equal(
            partition.contains(split, index), partition.mask(split)[index]
        )
    test_only = np.setdiff1d(partition.pool("test", 11), np.flatnonzero(partition.mask("train")))
    assert len(test_only) > 0
    assert not partition.contains("train", test_only).any()
