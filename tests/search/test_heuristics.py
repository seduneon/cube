"""Admissibility-preserving heuristic transformations, against the exact table."""

import numpy as np
import pytest

from eqcube.data import DistanceTable
from eqcube.puzzles.pocket_coords import pocket_coords
from eqcube.search.heuristics import LowerBoundHeuristic, ParityRounded, Pathmax

C = pocket_coords()
PUZZLE = C.puzzle
INDEX = np.random.default_rng(0).choice(C.n_states, size=5000, replace=False)
STATES = C.unrank(INDEX)


@pytest.fixture(scope="module")
def table() -> DistanceTable:
    return DistanceTable.build(C)


def shrunk(table: DistanceTable, factor: float = 0.7):
    """An admissible, non-integer heuristic: factor * d."""

    def h(states: np.ndarray) -> np.ndarray:
        return factor * table.distance(states).astype(np.float64)

    return h


def test_lower_bound_heuristic_is_admissible(table: DistanceTable) -> None:
    h = LowerBoundHeuristic(C)(STATES)
    assert (h <= table.distance(STATES)).all()


def test_parity_rounding_stays_admissible_and_tightens(table: DistanceTable) -> None:
    base = shrunk(table)
    d, h = table.distance(STATES), base(STATES)
    r = ParityRounded(base, C)(STATES)
    assert (r >= h).all() and (r <= d).all()
    assert ((d - r) % 2 == 0).all()
    assert (r > h + 0.5).mean() > 0.5  # it does tighten


def test_parity_rounding_keeps_exact_values(table: DistanceTable) -> None:
    def exact(states: np.ndarray) -> np.ndarray:
        return table.distance(states).astype(np.float64)

    np.testing.assert_array_equal(ParityRounded(exact, C)(STATES), table.distance(STATES))


def patchy(table: DistanceTable):
    """Admissible but inconsistent: exact on about half the states, 0 elsewhere."""

    def h(states: np.ndarray) -> np.ndarray:
        index = C.rank(states)
        keep = (index * 2654435761) % 7 < 3  # deterministic per state
        return np.where(keep, table.distance(states), 0).astype(np.float64)

    return h


@pytest.mark.parametrize("make", ["lower_bound", "shrunk"])
def test_pathmax_leaves_consistent_heuristics_unchanged(table: DistanceTable, make: str) -> None:
    """|h(s) - h(c)| <= 1 for both, so max_c h(c) - 1 <= h(s) already."""
    base = LowerBoundHeuristic(C) if make == "lower_bound" else shrunk(table)
    np.testing.assert_array_equal(Pathmax(base, PUZZLE)(STATES), base(STATES))


def test_pathmax_tightens_an_inconsistent_heuristic(table: DistanceTable) -> None:
    base = patchy(table)
    d, h = table.distance(STATES), base(STATES)
    p = Pathmax(base, PUZZLE)(STATES)
    assert (p >= h).all() and (p <= d).all()
    assert (p > h).mean() > 0.3


def test_pathmax_is_zero_on_solved() -> None:
    lb = LowerBoundHeuristic(C)
    assert Pathmax(lb, PUZZLE)(PUZZLE.solved_states(1)).tolist() == [0.0]
