"""Beam search with an exact oracle must find optimal solutions."""

import numpy as np
import torch

from eqcube.models import PRESETS, NetHeuristic, build_model
from eqcube.puzzles.base import bfs_distances
from eqcube.puzzles.pocket_cube import pocket_cube
from eqcube.search import BeamSearch, SolveResult, TableHeuristic, greedy
from eqcube.search.beam import first_unique_rows, lowest_per_group

PC = pocket_cube()
DEPTH = 5
DIST = bfs_distances(PC, DEPTH)
ORACLE = TableHeuristic(DIST, default=DEPTH + 1)


def sample_by_distance(d: int, n: int) -> np.ndarray:
    keys = [k for k, v in DIST.items() if v == d][:n]
    return np.stack([np.frombuffer(k, dtype=np.int8) for k in keys])


def assert_valid(starts: np.ndarray, results: list[SolveResult]) -> None:
    for s, r in zip(starts, results, strict=True):
        assert r.solved
        end = PC.apply_sequence(s[None], r.moves)
        assert PC.is_solved(end)[0]


def test_greedy_with_oracle_is_optimal() -> None:
    for d in range(DEPTH + 1):
        starts = sample_by_distance(d, 30)
        results = greedy(ORACLE).solve(PC, starts)
        assert_valid(starts, results)
        assert all(r.length == d for r in results)


def test_beam_with_oracle_is_optimal() -> None:
    starts = sample_by_distance(DEPTH, 50)
    results = BeamSearch(ORACLE, width=4).solve(PC, starts)
    assert_valid(starts, results)
    assert all(r.length == DEPTH for r in results)


def test_solved_start_needs_no_moves() -> None:
    results = BeamSearch(ORACLE).solve(PC, PC.solved_states(3))
    assert all(r.solved and r.moves == () for r in results)


def test_gives_up_after_max_steps() -> None:
    starts = sample_by_distance(DEPTH, 5)
    results = BeamSearch(ORACLE, width=2, max_steps=DEPTH - 1).solve(PC, starts)
    assert all(not r.solved and r.moves == () for r in results)


def test_untrained_net_runs_end_to_end() -> None:
    torch.manual_seed(0)
    h = NetHeuristic(build_model(PRESETS["task"]), reduce="expected")
    starts = PC.scramble(8, 3, np.random.default_rng(0))
    results = BeamSearch(h, width=8, max_steps=6).solve(PC, starts)
    assert len(results) == 8
    assert_valid(starts[[r.solved for r in results]], [r for r in results if r.solved])


def testfirst_unique_rows() -> None:
    owner = np.array([0, 0, 1, 0, 1])
    states = np.array([[1, 2], [1, 2], [1, 2], [3, 4], [1, 2]], dtype=np.int8)
    assert first_unique_rows(owner, states).tolist() == [0, 2, 3]


def testlowest_per_group() -> None:
    owner = np.array([1, 0, 1, 0, 1, 0])
    costs = np.array([5.0, 2.0, 1.0, 9.0, 1.0, 0.0])
    keep = lowest_per_group(owner, costs, 2)
    assert sorted(keep.tolist()) == [1, 2, 4, 5]


def test_goal_is_found_even_if_heuristic_hates_the_solved_state() -> None:
    """Regression: a model that scores the solved state worst must not prune it away."""
    solved_key = PC.solved_states(1)[0].tobytes()

    def hostile(states: np.ndarray) -> np.ndarray:
        costs = ORACLE(states)
        is_goal = np.array([s.tobytes() == solved_key for s in states])
        return np.where(is_goal, 1e9, costs)

    for d in (1, 3, DEPTH):
        starts = sample_by_distance(d, 20)
        results = BeamSearch(hostile, width=2).solve(PC, starts)
        assert_valid(starts, results)
        assert all(r.length == d for r in results)
