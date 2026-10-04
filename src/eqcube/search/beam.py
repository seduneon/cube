"""Batched beam search: many start states advance together, one heuristic call per step."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from eqcube.puzzles.base import PermutationPuzzle, States
from eqcube.search.base import Heuristic, SolveResult

IntArray = npt.NDArray[np.int64]


@dataclass
class BeamSearch:
    """Keep the ``width`` lowest-cost states per problem at every depth.

    Goal test happens when children are generated, before pruning, so a
    heuristic that misjudges the solved state cannot hide it.

    ``dedupe`` drops duplicate states inside one problem's candidate set, so a
    beam never wastes slots on copies. ``width=1`` is greedy search.
    """

    heuristic: Heuristic
    width: int = 20
    max_steps: int = 50
    dedupe: bool = True

    def solve(self, puzzle: PermutationPuzzle, starts: States) -> list[SolveResult]:
        n, k = len(starts), puzzle.n_moves
        found: list[tuple[int, ...] | None] = [None] * n
        done = np.zeros(n, dtype=bool)

        states = np.ascontiguousarray(starts, dtype=np.int8)
        owner = np.arange(n, dtype=np.int64)
        paths = np.empty((n, 0), dtype=np.int64)

        def record_solved() -> None:
            """Goal test on the current rows; finished problems leave the beam."""
            nonlocal states, owner, paths
            for row in np.flatnonzero(puzzle.is_solved(states)):
                o = int(owner[row])
                if not done[o]:
                    done[o] = True
                    found[o] = tuple(int(m) for m in paths[row])
            live = ~done[owner]
            states, owner, paths = states[live], owner[live], paths[live]

        record_solved()
        for _ in range(self.max_steps):
            if len(states) == 0:
                break
            # expand every beam entry by every move
            states = puzzle.expand(states).reshape(-1, puzzle.n_positions)
            owner = np.repeat(owner, k)
            moves = np.tile(np.arange(k, dtype=np.int64), len(paths))
            paths = np.concatenate([np.repeat(paths, k, axis=0), moves[:, None]], axis=1)

            # Goal test at generation, *before* the heuristic can prune a solved child:
            # being solved is known exactly, so search must not depend on the model for it.
            record_solved()
            if len(states) == 0:
                break

            if self.dedupe:
                keep = first_unique_rows(owner, states)
                states, owner, paths = states[keep], owner[keep], paths[keep]

            costs = self.heuristic(states)
            keep = lowest_per_group(owner, costs, self.width)
            states, owner, paths = states[keep], owner[keep], paths[keep]

        return [SolveResult(f is not None, f or ()) for f in found]


def greedy(heuristic: Heuristic, max_steps: int = 50) -> BeamSearch:
    return BeamSearch(heuristic, width=1, max_steps=max_steps, dedupe=False)


def first_unique_rows(owner: IntArray, states: States) -> IntArray:
    """Indices of the first occurrence of each (owner, state) pair, in original order."""
    rows = np.concatenate(
        [
            owner.astype(np.int64).view(np.uint8).reshape(len(owner), 8),
            np.ascontiguousarray(states).view(np.uint8),
        ],
        axis=1,
    )
    keys = np.ascontiguousarray(rows).view(np.dtype((np.void, rows.shape[1]))).ravel()
    _, first = np.unique(keys, return_index=True)
    return np.sort(first)


def lowest_per_group(owner: IntArray, costs: npt.NDArray[np.float64], width: int) -> IntArray:
    """Indices of the ``width`` lowest-cost rows of every owner (ties: earlier row wins)."""
    order = np.lexsort((costs, owner))  # by owner, then cost; stable
    sorted_owner = owner[order]
    position = np.arange(len(order))
    is_start = np.r_[True, sorted_owner[1:] != sorted_owner[:-1]]
    group_start = np.maximum.accumulate(np.where(is_start, position, 0))
    return order[position - group_start < width]
