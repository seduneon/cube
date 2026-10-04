"""Permutation puzzles: states are functions positions -> colors.

A move is a ``Perm`` of positions, acting on a state exactly like
``FunctionAction`` (gather with the inverse):

    (m·s)[p] = s[m^{-1}(p)]

All state operations are batched: states are int8 arrays of shape (N, P).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import numpy.typing as npt

from eqcube.symmetry.action import ConjugationAction, FunctionAction, SetAction
from eqcube.symmetry.perm import Perm, PermGroup

States = npt.NDArray[np.int8]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True, eq=False)
class PermutationPuzzle:
    """A puzzle whose moves permute positions and whose goal is one fixed state."""

    name: str
    move_names: tuple[str, ...]
    moves: tuple[Perm, ...]
    solved: tuple[int, ...]
    n_colors: int

    def __post_init__(self) -> None:
        if len(self.move_names) != len(self.moves):
            raise ValueError("one name per move")
        if any(m.degree != self.n_positions for m in self.moves):
            raise ValueError("every move must permute all positions")
        if not all(0 <= c < self.n_colors for c in self.solved):
            raise ValueError("solved state uses an unknown color")
        if set(self.moves) != {m.inverse() for m in self.moves}:
            raise ValueError("move set must be closed under inverses")

    # ── structure ─────────────────────────────────────────────────────────

    @property
    def n_positions(self) -> int:
        return len(self.solved)

    @property
    def n_moves(self) -> int:
        return len(self.moves)

    @cached_property
    def inverse_move(self) -> tuple[int, ...]:
        """``inverse_move[k]`` is the index of moves[k]^{-1}."""
        index = {m: k for k, m in enumerate(self.moves)}
        return tuple(index[m.inverse()] for m in self.moves)

    @cached_property
    def _gather(self) -> IntArray:
        """(n_moves, P) gather indices: row k is moves[k]^{-1}."""
        return np.array([m.inverse().arr for m in self.moves], dtype=np.int64)

    # ── batched state operations ──────────────────────────────────────────

    def solved_states(self, n: int = 1) -> States:
        return np.tile(np.array(self.solved, dtype=np.int8), (n, 1))

    def is_solved(self, states: States) -> BoolArray:
        return np.all(states == np.array(self.solved, dtype=np.int8), axis=1)

    def apply(self, states: States, moves: IntArray) -> States:
        """Apply ``moves[i]`` to ``states[i]``."""
        idx = self._gather[moves]  # (N, P)
        return np.take_along_axis(states, idx, axis=1)

    def expand(self, states: States) -> States:
        """All children: shape (N, n_moves, P), child [i, k] = moves[k]·states[i]."""
        return states[:, self._gather]

    def apply_sequence(self, states: States, sequence: Sequence[int]) -> States:
        for k in sequence:
            states = self.apply(states, np.full(len(states), k, dtype=np.int64))
        return states

    def scramble(self, n: int, depth: int, rng: np.random.Generator) -> States:
        """n random walks of ``depth`` moves, never undoing the previous move."""
        states = self.solved_states(n)
        inverse = np.array(self.inverse_move, dtype=np.int64)
        last = np.full(n, -1, dtype=np.int64)
        for _ in range(depth):
            high = np.where(last >= 0, self.n_moves - 1, self.n_moves)  # one fewer if forbidden
            k = rng.integers(0, high)
            forbidden = np.where(last >= 0, inverse[np.maximum(last, 0)], self.n_moves)
            k = k + (k >= forbidden)  # skip the forbidden index
            states = self.apply(states, k)
            last = k
        return states

    # ── symmetry ──────────────────────────────────────────────────────────

    def lift_move(self, move: Perm) -> Perm:
        """A move as a Perm of positions + colors (it does not touch colors)."""
        return move.direct_sum(Perm.identity(self.n_colors))

    def task_symmetry(self, ambient: PermGroup) -> PermGroup:
        """The symmetries of the solving task inside ``ambient``.

        ``ambient`` acts on P positions followed by C colors (degree P + C).
        g is a task symmetry iff
            (1) g·solved == solved          (goal is preserved)
            (2) g M g^{-1} == M             (move set is preserved)
        Then d(g·s) = d(s) for every state s, since
            g·(m·s) = (g m g^{-1})·(g·s)
        maps shortest solutions of s to shortest solutions of g·s.
        """
        if ambient.degree != self.n_positions + self.n_colors:
            raise ValueError("ambient group must act on positions + colors")
        on_states = FunctionAction(ambient, n_domain=self.n_positions)
        on_move_sets = SetAction(ConjugationAction(ambient))
        moves = frozenset(self.lift_move(m) for m in self.moves)
        return PermGroup(
            g
            for g in ambient.elements
            if on_states(g, self.solved) == self.solved and on_move_sets(g, moves) == moves
        )


def state_keys(states: States) -> list[bytes]:
    """Hashable key per state (for dicts and sets)."""
    contiguous = np.ascontiguousarray(states, dtype=np.int8)
    return [row.tobytes() for row in contiguous]


def bfs_distances(puzzle: PermutationPuzzle, max_depth: int) -> dict[bytes, int]:
    """Exact distance-to-solved for every state within ``max_depth`` moves."""
    frontier = puzzle.solved_states(1)
    dist = {state_keys(frontier)[0]: 0}
    for d in range(1, max_depth + 1):
        children = puzzle.expand(frontier).reshape(-1, puzzle.n_positions)
        fresh: list[int] = []
        for i, key in enumerate(state_keys(children)):
            if key not in dist:
                dist[key] = d
                fresh.append(i)
        if not fresh:
            break
        frontier = children[fresh]
    return dist
