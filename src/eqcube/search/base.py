"""Search interfaces. NumPy only: a heuristic is anything states -> costs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from eqcube.puzzles.base import PermutationPuzzle, States, state_keys

Costs = npt.NDArray[np.float64]


class Heuristic(Protocol):
    """Estimated distance-to-solved for a batch of states (lower is better)."""

    def __call__(self, states: States, /) -> Costs: ...


class Searcher(Protocol):
    def solve(self, puzzle: PermutationPuzzle, starts: States) -> list[SolveResult]: ...


@dataclass(frozen=True)
class SolveResult:
    solved: bool
    moves: tuple[int, ...]  # move indices; empty if unsolved

    @property
    def length(self) -> int:
        return len(self.moves)


class TableHeuristic:
    """Exact distances from a lookup table (e.g. ``bfs_distances``); ``default`` if absent."""

    def __init__(self, table: Mapping[bytes, int], default: float) -> None:
        self.table, self.default = table, default

    def __call__(self, states: States, /) -> Costs:
        return np.array([self.table.get(k, self.default) for k in state_keys(states)], dtype=float)
