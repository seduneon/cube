"""Symmetries acting on states and on state indices."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

from eqcube.data.table import StateIndexer
from eqcube.puzzles.base import States
from eqcube.symmetry.perm import Perm, PermGroup

IntArray = npt.NDArray[np.int64]


def apply_symmetry(states: States, g: Perm, n_positions: int) -> States:
    """Batched FunctionAction: (g·s)[p] = sigma(s[pi^{-1}(p)]).

    ``g`` acts on positions 0..P-1 (pi) followed by colors P..P+C-1 (sigma).
    """
    arr = np.array(g.arr, dtype=np.int64)
    gather = np.argsort(arr[:n_positions])  # pi^{-1}
    sigma = (arr[n_positions:] - n_positions).astype(np.int8)
    if len(sigma) == 0:
        return states[:, gather]
    return sigma[states[:, gather]]


def orbit_representatives(
    indexer: StateIndexer,
    group: PermGroup,
    chunk: int = 500_000,
    cache_path: str | Path | None = None,
) -> IntArray:
    """For every index i, the smallest index in its orbit {g·i : g in group}."""
    if cache_path is not None and Path(cache_path).exists():
        return np.load(cache_path)
    n_positions = indexer.puzzle.n_positions
    others = [g for g in group.elements if not g.is_identity()]
    reps = np.arange(indexer.n_states, dtype=np.int64)
    for start in range(0, indexer.n_states, chunk):
        index = reps[start : start + chunk].copy()
        states = indexer.unrank(index)
        best = index
        for g in others:
            best = np.minimum(best, indexer.rank(apply_symmetry(states, g, n_positions)))
        reps[start : start + chunk] = best
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, reps)
    return reps
