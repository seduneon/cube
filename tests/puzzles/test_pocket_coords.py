"""Coordinates: bijection, move tables vs sticker moves, full BFS."""

import itertools

import numpy as np
import pytest

from eqcube.data.table import bfs
from eqcube.puzzles.base import bfs_distances, state_keys
from eqcube.puzzles.pocket_coords import (
    N_PERM,
    N_STATES,
    N_TWIST,
    PocketCubeCoords,
    pocket_coords,
)
from eqcube.puzzles.pocket_cube import MAX_DISTANCE

C = pocket_coords()
PC = C.puzzle
RNG = np.random.default_rng(0)
STATES = PC.scramble(3000, 25, RNG)

# Number of states at each distance (quarter-turn metric, R/U/F).
KNOWN_COUNTS = [1, 6, 27, 120, 534, 2256, 8969, 33058, 114149, 360508, 930588, 1350852,
                782536, 90280, 276]  # fmt: skip


@pytest.fixture(scope="module")
def dist() -> np.ndarray:
    return bfs(C)


# ── bijections ────────────────────────────────────────────────────────────


def test_perm_rank_is_lexicographic_bijection() -> None:
    perms = np.array(list(itertools.permutations(range(7))))
    np.testing.assert_array_equal(C.rank_perm(perms), np.arange(N_PERM))
    np.testing.assert_array_equal(C.unrank_perm(np.arange(N_PERM)), perms)


def test_twist_rank_is_bijection() -> None:
    twists = C.unrank_twist(np.arange(N_TWIST))
    assert (twists.sum(axis=1) % 3 == 0).all()
    np.testing.assert_array_equal(C.rank_twist(twists), np.arange(N_TWIST))
    assert len({tuple(t) for t in twists}) == N_TWIST


def test_sticker_roundtrip() -> None:
    np.testing.assert_array_equal(C.unrank(C.rank(STATES)), STATES)


def test_index_roundtrip_on_random_indices() -> None:
    index = RNG.integers(0, N_STATES, size=5000)
    np.testing.assert_array_equal(C.rank(C.unrank(index)), index)


def test_solved_is_index_zero() -> None:
    assert C.solved_index == 0


def test_twists_sum_to_zero_on_reachable_states() -> None:
    _, twist = C.decode(STATES)
    assert (twist.sum(axis=1) % 3 == 0).all()


def test_rejects_invalid_state() -> None:
    bad = PC.solved_states(1)
    bad[0, [0, 8]] = bad[0, [8, 0]]  # swap two stickers of different colors
    with pytest.raises(ValueError):
        C.decode(bad)


# ── move tables ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("move", range(6))
def test_move_table_matches_sticker_move(move: int) -> None:
    moves = np.full(len(STATES), move)
    np.testing.assert_array_equal(C.move(C.rank(STATES), moves), C.rank(PC.apply(STATES, moves)))


def test_expand_matches_move() -> None:
    index = C.rank(STATES)
    children = C.expand(index)
    for m in range(PC.n_moves):
        np.testing.assert_array_equal(children[:, m], C.move(index, np.full(len(index), m)))


def test_tables_are_independent_of_construction_order() -> None:
    fresh = PocketCubeCoords(PC)
    np.testing.assert_array_equal(fresh.perm_table, C.perm_table)
    np.testing.assert_array_equal(fresh.twist_table, C.twist_table)


# ── full BFS ──────────────────────────────────────────────────────────────


def test_distance_counts(dist: np.ndarray) -> None:
    assert np.bincount(dist).tolist() == KNOWN_COUNTS
    assert int(dist.max()) == MAX_DISTANCE


def test_agrees_with_sticker_bfs(dist: np.ndarray) -> None:
    shallow = bfs_distances(PC, 6)
    states = np.stack([np.frombuffer(k, dtype=np.int8) for k in shallow])
    np.testing.assert_array_equal(dist[C.rank(states)], list(shallow.values()))


def test_every_move_changes_distance_by_one(dist: np.ndarray) -> None:
    """Parity: a quarter turn is an odd corner permutation, so d(child) = d(parent) +- 1."""
    index = RNG.integers(0, N_STATES, size=200_000)
    delta = dist[C.expand(index)].astype(int) - dist[index, None]
    assert set(np.unique(delta).tolist()) == {-1, 1}


def test_scrambled_states_have_consistent_labels(dist: np.ndarray) -> None:
    keys = state_keys(STATES)
    assert len(keys) == len(STATES)
    assert (dist[C.rank(STATES)] <= MAX_DISTANCE).all()


# ── pattern databases and inversion ──────────────────────────────────────


def test_lower_bound_is_admissible_on_every_state(dist: np.ndarray) -> None:
    """Exhaustive (cheap: two table lookups per state)."""
    bound = C.lower_bound(np.arange(N_STATES))
    d = dist.astype(np.int64)
    assert (bound <= d).all()
    assert ((d - bound) % 2 == 0).all()  # parity-rounded
    shallow = d <= 4
    np.testing.assert_array_equal(bound[shallow], d[shallow])  # exact near solved


def test_sub_puzzle_distances_reach_every_coordinate() -> None:
    assert C.perm_distance.shape == (N_PERM,) and (C.perm_distance >= 0).all()
    assert C.twist_distance.shape == (N_TWIST,) and (C.twist_distance >= 0).all()
    assert C.perm_distance[0] == 0 and C.twist_distance[0] == 0  # solved


def test_invert_undoes_the_scramble_sequence() -> None:
    """s = solved after m1..mk  =>  s^-1 = solved after mk^-1..m1^-1."""
    rng = np.random.default_rng(1)
    inverse = PC.inverse_move
    for _ in range(50):
        seq = rng.integers(0, PC.n_moves, size=rng.integers(1, 20)).tolist()
        s = PC.apply_sequence(PC.solved_states(1), seq)
        undo = PC.apply_sequence(PC.solved_states(1), [inverse[m] for m in reversed(seq)])
        np.testing.assert_array_equal(C.invert(s), undo)


def test_invert_is_an_involution_and_fixes_solved() -> None:
    np.testing.assert_array_equal(C.invert(C.invert(STATES)), STATES)
    np.testing.assert_array_equal(C.invert(PC.solved_states(1)), PC.solved_states(1))


def test_invert_preserves_distance_on_sample(dist: np.ndarray) -> None:
    np.testing.assert_array_equal(dist[C.rank(C.invert(STATES))], dist[C.rank(STATES)])


@pytest.mark.slow
def test_invert_preserves_distance_on_every_state(dist: np.ndarray) -> None:
    """Exhaustive: d(s^-1) = d(s) and s -> s^-1 is a bijection of the state space."""
    images = np.empty(N_STATES, dtype=np.int64)
    for start in range(0, N_STATES, 200_000):
        index = np.arange(start, min(start + 200_000, N_STATES))
        images[index] = C.rank(C.invert(C.unrank(index)))
    np.testing.assert_array_equal(dist[images], dist)
    np.testing.assert_array_equal(images[images], np.arange(N_STATES))
