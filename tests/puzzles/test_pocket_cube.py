"""Pocket cube moves, batching, and the computed task symmetry."""

import sys
from pathlib import Path

import numpy as np
import pytest

from eqcube.puzzles.base import bfs_distances, state_keys
from eqcube.puzzles.pocket_cube import (
    MAX_DISTANCE,
    SYMMETRIES,
    pocket_cube,
    rotation_color_group,
    rotation_group,
    task_group,
)
from eqcube.symmetry.action import FunctionAction

PC = pocket_cube()
DEPTH = 5
DIST = bfs_distances(PC, DEPTH)


def test_moves_have_order_four_and_inverses_pair_up() -> None:
    assert all(m.order() == 4 for m in PC.moves)
    assert PC.inverse_move == (1, 0, 3, 2, 5, 4)


def test_apply_matches_function_action() -> None:
    rng = np.random.default_rng(0)
    states = PC.scramble(20, 7, rng)
    action = FunctionAction(rotation_group(), n_domain=24)  # any degree-24+6 group works
    for k, move in enumerate(PC.moves):
        batched = PC.apply(states, np.full(20, k))
        lifted = PC.lift_move(move)
        for s, b in zip(states, batched, strict=True):
            assert tuple(b) == action(lifted, tuple(int(c) for c in s))


def test_expand_matches_apply() -> None:
    states = PC.scramble(5, 4, np.random.default_rng(1))
    children = PC.expand(states)
    for k in range(PC.n_moves):
        np.testing.assert_array_equal(children[:, k], PC.apply(states, np.full(5, k)))


def test_move_then_inverse_is_identity() -> None:
    states = PC.scramble(10, 6, np.random.default_rng(2))
    for k in range(PC.n_moves):
        back = PC.apply_sequence(states, [k, PC.inverse_move[k]])
        np.testing.assert_array_equal(back, states)


def test_scramble_uses_every_move() -> None:
    first = PC.scramble(600, 1, np.random.default_rng(3))
    assert len(set(state_keys(first))) == PC.n_moves


def test_bfs_layer_sizes() -> None:
    counts = np.bincount(list(DIST.values()))
    assert counts.tolist() == [1, 6, 27, 120, 534, 2256]


def test_matches_legacy_moves() -> None:
    legacy = Path(__file__).resolve().parents[2] / "legacy"
    sys.path.insert(0, str(legacy))
    try:
        cube_env = pytest.importorskip("cube_env")
    finally:
        sys.path.remove(str(legacy))
    rng = np.random.default_rng(4)
    for s in PC.scramble(10, 8, rng):
        for k in range(PC.n_moves):
            ours = PC.apply(s[None], np.array([k]))[0]
            np.testing.assert_array_equal(ours, cube_env.apply_move_idx(s, k))


# ── symmetry ──────────────────────────────────────────────────────────────


def test_group_orders() -> None:
    orders = {name: make().order for name, make in SYMMETRIES.items()}
    assert orders == {
        "trivial": 1,
        "rotation": 24,
        "color": 720,
        "rotation_color": 17280,
        "task": 3,
    }


def test_task_symmetry_preserves_distance() -> None:
    """The theorem in PermutationPuzzle.task_symmetry, checked on all states <= DEPTH."""
    action = FunctionAction(task_group(), n_domain=24)
    for key, d in DIST.items():
        s = tuple(np.frombuffer(key, dtype=np.int8).tolist())
        for g in task_group().elements:
            image = np.array(action(g, s), dtype=np.int8).tobytes()
            assert DIST.get(image) == d


def test_other_symmetries_do_not_preserve_distance() -> None:
    """Every non-task element of O x S6 sends some state out of its distance layer."""
    action = FunctionAction(rotation_color_group(), n_domain=24)
    layer = [tuple(np.frombuffer(k, dtype=np.int8).tolist()) for k, d in DIST.items() if d == 3]
    for g in rotation_color_group().elements:
        if g in task_group():
            continue
        assert any(DIST.get(np.array(action(g, s), dtype=np.int8).tobytes()) != 3 for s in layer)


def test_max_distance_constant() -> None:
    assert MAX_DISTANCE == 14
