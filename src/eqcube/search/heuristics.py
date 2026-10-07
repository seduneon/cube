"""Heuristic transformations that keep admissibility (docs/THEORY.md, section 8).

    LowerBoundHeuristic   the pattern-database bound itself (admissible)
    ParityRounded         round up to the next integer of the right parity
    Pathmax               h'(s) = max(h(s), max_c h(c) - 1)

If h <= d then each transformed heuristic is also <= d and never smaller
than h. Pathmax changes only *inconsistent* heuristics (some |h(s) - h(c)| > 1):
the pattern-database bound is consistent, a learned quantile need not be.
These are for admissible heuristics (A*). On a learned, non-admissible
heuristic used for ranking, parity rounding hurts: all siblings share a
parity, so rounding merges nearby costs into ties.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eqcube.puzzles.base import PermutationPuzzle, States
from eqcube.puzzles.pocket_coords import PocketCubeCoords
from eqcube.search.base import Costs, Heuristic


@dataclass
class LowerBoundHeuristic:
    coords: PocketCubeCoords

    def __call__(self, states: States, /) -> Costs:
        return self.coords.lower_bound(self.coords.rank(states)).astype(np.float64)


@dataclass
class ParityRounded:
    """d is an integer with the parity of the corner permutation, so if h <= d,
    the smallest such integer >= h is still <= d."""

    base: Heuristic
    coords: PocketCubeCoords
    tol: float = 1e-9  # 3.0000000001 stays 3

    def __call__(self, states: States, /) -> Costs:
        up = np.ceil(self.base(states) - self.tol)
        parity = self.coords.parity(self.coords.rank(states))
        return up + (np.mod(up, 2) != parity)


@dataclass
class Pathmax:
    """One move changes d by exactly 1, so d(s) >= d(c) - 1 for every child c.

    Costs one extra heuristic call on the M children of every state.
    """

    base: Heuristic
    puzzle: PermutationPuzzle

    def __call__(self, states: States, /) -> Costs:
        own = self.base(states)
        if len(states) == 0:
            return own
        children = self.puzzle.expand(states)
        n, m, p = children.shape
        child = self.base(children.reshape(n * m, p)).reshape(n, m)
        out = np.maximum(own, child.max(axis=1) - 1)
        return np.where(self.puzzle.is_solved(states), 0.0, out)
