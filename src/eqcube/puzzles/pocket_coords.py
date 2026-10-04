"""Coordinates for the pocket cube: a bijection states <-> {0, ..., 3,674,159}.

Why: storing distances for every state (BFS) and sampling states by index
need a compact integer per state, and BFS must apply moves fast.

Structure used (the puzzle group K is a semidirect product Z_3^6 x| S_7):
  * 7 movable corner slots (DBL never moves). A state puts one of the 7
    movable pieces in each slot (a permutation) with a twist in {0, 1, 2}.
  * Twists are recorded *per slot*: the twist is how many steps (in a fixed
    rotational order of the slot's three stickers) the piece's U/D-colored
    sticker sits from the slot's U/D sticker. The 7 twists sum to 0 mod 3,
    so 6 of them determine the 7th.

  index = perm_coord * 729 + twist_coord
          perm_coord  in [0, 5040)  Lehmer code of the permutation
          twist_coord in [0, 729)   base-3 number of the first 6 twists

Move tables: a move's effect on the permutation ignores twists, and its
effect on slot twists ignores which piece is where. So each coordinate gets
its own small table (5040 x 6 and 729 x 6), and moves act on indices without
decoding to stickers:

  move(index, m) = perm_table[p, m] * 729 + twist_table[t, m]

``rank``/``unrank`` convert batches of sticker states to and from indices.
Full BFS over indices lives in ``eqcube.data.table.bfs``.
"""

from __future__ import annotations

from functools import cache, cached_property
from math import factorial

import numpy as np
import numpy.typing as npt

from eqcube.puzzles.base import PermutationPuzzle, States
from eqcube.puzzles.pocket_cube import corner_slots, pocket_cube

IntArray = npt.NDArray[np.int64]

N_PIECES = 7  # movable corners
N_PERM = factorial(N_PIECES)  # 5040
N_TWIST = 3 ** (N_PIECES - 1)  # 729
N_STATES = N_PERM * N_TWIST  # 3,674,160


class PocketCubeCoords:
    """Rank/unrank pocket cube states and apply moves on indices."""

    def __init__(self, puzzle: PermutationPuzzle) -> None:
        self.puzzle = puzzle
        slots = np.array(corner_slots(), dtype=np.int64)  # (8, 3), DBL last, U/D sticker first
        self.slot_stickers = slots
        solved = np.array(puzzle.solved, dtype=np.int64)
        self.piece_colors = solved[slots]  # (8, 3): piece i's colors in its home slot's order
        self.ud_colors = np.unique(self.piece_colors[:, 0])  # the U and D colors

        # canonical color triple (U/D color first) -> piece id
        code = self.piece_colors @ np.array([36, 6, 1])
        self._piece_of_code = np.full(216, -1, dtype=np.int64)
        self._piece_of_code[code] = np.arange(8)
        self._solved_index = int(self.rank(puzzle.solved_states(1))[0])

    @property
    def n_states(self) -> int:
        return N_STATES

    # ── stickers <-> (perm, twist) ────────────────────────────────────────

    def decode(self, states: States) -> tuple[IntArray, IntArray]:
        """(N, 24) stickers -> perm (N, 7) piece per slot, twist (N, 7)."""
        colors = states.astype(np.int64)[:, self.slot_stickers[:N_PIECES]]  # (N, 7, 3)
        is_ud = np.isin(colors, self.ud_colors)
        twist = is_ud.argmax(axis=2)  # (N, 7)
        k = (twist[..., None] + np.arange(3)) % 3  # read colors starting at the U/D one
        canonical = np.take_along_axis(colors, k, axis=2)
        perm = self._piece_of_code[canonical @ np.array([36, 6, 1])]
        if (perm < 0).any() or (is_ud.sum(axis=2) != 1).any():
            raise ValueError("not a valid pocket cube state")
        return perm, twist

    def encode(self, perm: IntArray, twist: IntArray) -> States:
        """perm (N, 7), twist (N, 7) -> (N, 24) stickers."""
        n = len(perm)
        states = np.tile(np.array(self.puzzle.solved, dtype=np.int8), (n, 1))
        for slot in range(N_PIECES):
            for k in range(3):
                position = self.slot_stickers[slot, (twist[:, slot] + k) % 3]  # (N,)
                states[np.arange(n), position] = self.piece_colors[perm[:, slot], k]
        return states

    # ── (perm, twist) <-> coordinates ─────────────────────────────────────

    @staticmethod
    def rank_perm(perm: IntArray) -> IntArray:
        """Lehmer code: sum_i #{j > i : perm[j] < perm[i]} * (n-1-i)!"""
        n = perm.shape[1]
        rank = np.zeros(len(perm), dtype=np.int64)
        for i in range(n):
            smaller_after = (perm[:, i + 1 :] < perm[:, i : i + 1]).sum(axis=1)
            rank = rank + smaller_after * factorial(n - 1 - i)
        return rank

    @staticmethod
    def unrank_perm(rank: IntArray, n: int = N_PIECES) -> IntArray:
        perm = np.empty((len(rank), n), dtype=np.int64)
        available = np.ones((len(rank), n), dtype=bool)
        rest = rank.copy()
        for i in range(n):
            digit, rest = np.divmod(rest, factorial(n - 1 - i))
            # the (digit)-th still-available value
            position = (np.cumsum(available, axis=1) == (digit + 1)[:, None]) & available
            value = position.argmax(axis=1)
            perm[:, i] = value
            available[np.arange(len(rank)), value] = False
        return perm

    @staticmethod
    def rank_twist(twist: IntArray) -> IntArray:
        if ((twist.sum(axis=1) % 3) != 0).any():
            raise ValueError("twists must sum to 0 mod 3")
        powers = 3 ** np.arange(N_PIECES - 2, -1, -1)
        return twist[:, : N_PIECES - 1] @ powers

    @staticmethod
    def unrank_twist(rank: IntArray) -> IntArray:
        twist = np.empty((len(rank), N_PIECES), dtype=np.int64)
        rest = rank.copy()
        for i in range(N_PIECES - 2, -1, -1):
            rest, twist[:, i] = np.divmod(rest, 3)
        twist[:, -1] = (-twist[:, :-1].sum(axis=1)) % 3
        return twist

    # ── stickers <-> index ────────────────────────────────────────────────

    def rank(self, states: States) -> IntArray:
        perm, twist = self.decode(states)
        return self.rank_perm(perm) * N_TWIST + self.rank_twist(twist)

    def unrank(self, index: IntArray) -> States:
        p, t = np.divmod(np.asarray(index, dtype=np.int64), N_TWIST)
        return self.encode(self.unrank_perm(p), self.unrank_twist(t))

    # ── moves on indices ──────────────────────────────────────────────────

    @cached_property
    def perm_table(self) -> IntArray:
        """(5040, n_moves): perm coordinate after each move (twists irrelevant)."""
        perms = self.unrank_perm(np.arange(N_PERM))
        states = self.encode(perms, np.zeros_like(perms))
        children = self.puzzle.expand(states)  # (5040, M, 24)
        flat = children.reshape(-1, self.puzzle.n_positions)
        return self.rank_perm(self.decode(flat)[0]).reshape(N_PERM, self.puzzle.n_moves)

    @cached_property
    def twist_table(self) -> IntArray:
        """(729, n_moves): twist coordinate after each move (pieces irrelevant)."""
        twists = self.unrank_twist(np.arange(N_TWIST))
        identity = np.tile(np.arange(N_PIECES), (N_TWIST, 1))
        children = self.puzzle.expand(self.encode(identity, twists))
        flat = children.reshape(-1, self.puzzle.n_positions)
        return self.rank_twist(self.decode(flat)[1]).reshape(N_TWIST, self.puzzle.n_moves)

    def move(self, index: IntArray, moves: IntArray) -> IntArray:
        """Apply ``moves[i]`` to state ``index[i]``."""
        p, t = np.divmod(index, N_TWIST)
        return self.perm_table[p, moves] * N_TWIST + self.twist_table[t, moves]

    def expand(self, index: IntArray) -> IntArray:
        """(N,) -> (N, n_moves) child indices."""
        p, t = np.divmod(index, N_TWIST)
        return self.perm_table[p] * N_TWIST + self.twist_table[t]

    @property
    def solved_index(self) -> int:
        return self._solved_index


@cache
def pocket_coords() -> PocketCubeCoords:
    return PocketCubeCoords(pocket_cube())
