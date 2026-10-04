"""Pocket cube moves, batching, and the computed task symmetry."""

import sys
from pathlib import Path

import numpy as np
import pytest

from eqcube.puzzles.base import bfs_distances, state_keys
from eqcube.puzzles.pocket_cube import (
    MAX_DISTANCE,
    SYMMETRIES,
    cube_isometries,
    cube_rotations,
    isometry_color_group,
    pocket_cube,
    rotation_color_group,
    rotation_group,
    task_group,
)
from eqcube.symmetry.action import FunctionAction
from eqcube.symmetry.perm import Perm, PermGroup

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
        "isometry_color": 34560,
        "task": 6,
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
    """Every non-task element of O_h x S6 sends some state out of its distance layer."""
    action = FunctionAction(isometry_color_group(), n_domain=24)
    layer = [tuple(np.frombuffer(k, dtype=np.int8).tolist()) for k, d in DIST.items() if d == 3]
    for g in isometry_color_group().elements:
        if g in task_group():
            continue
        assert any(DIST.get(np.array(action(g, s), dtype=np.int8).tobytes()) != 3 for s in layer)


def test_max_distance_constant() -> None:
    assert MAX_DISTANCE == 14


# ── geometry: moves and symmetries derived from 3D sticker coordinates ────

LEGACY_CYCLES = {
    "R": [(3, 12, 7, 11), (9, 1, 14, 5), (20, 21, 23, 22)],
    "U": [(0, 1, 3, 2), (9, 17, 13, 21), (8, 16, 12, 20)],
    "F": [(8, 9, 11, 10), (3, 22, 4, 17), (20, 5, 19, 2)],
}


def test_geometric_moves_match_hand_written_cycles() -> None:
    for name, cycles in LEGACY_CYCLES.items():
        assert PC.moves[PC.move_names.index(name)] == Perm.from_cycles(cycles, 24)


def test_geometric_rotations_match_legacy_group(cube_rotations: PermGroup) -> None:
    from eqcube.puzzles.pocket_cube import cube_rotations as geometric_rotations

    assert geometric_rotations().elements == cube_rotations.elements


def test_isometries_contain_rotations_with_index_two() -> None:
    assert cube_isometries().order == 48
    assert cube_rotations().elements < cube_isometries().elements


def test_fast_task_symmetry_matches_brute_force() -> None:
    assert PC.task_symmetry(cube_rotations()).elements == (
        PC.task_symmetry_brute_force(rotation_color_group()).elements
    )
    assert task_group().elements == PC.task_symmetry_brute_force(isometry_color_group()).elements


def test_task_group_is_s3_rotations_and_mirrors() -> None:
    """3 rotations keep R/U/F as a set; 3 mirrors send every move to an inverse move."""
    rotations = {g.arr[:24] for g in task_group().elements} & {
        g.arr for g in cube_rotations().elements
    }
    assert len(rotations) == 3
    names = {PC.lift_move(m): n for m, n in zip(PC.moves, PC.move_names, strict=True)}
    for g in task_group().elements:
        images = {n: names[g * m * g.inverse()] for m, n in names.items()}
        is_rotation = g.arr[:24] in rotations
        flips = all(images[n].endswith("'") != n.endswith("'") for n in images)
        assert flips != is_rotation  # rotations keep handedness, mirrors flip it
    assert all(g.order() in (1, 2, 3) for g in task_group().elements)  # S3 element orders


def test_recoloring_for_rejects_impossible_relabelling() -> None:
    # A single face turn mixes colors on faces: no recoloring can fix the solved state.
    assert PC.recoloring_for(PC.moves[0]) is None
    assert PC.recoloring_for(Perm.identity(24)) == Perm.identity(6)
