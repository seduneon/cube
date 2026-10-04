"""The exact distance of every state, indexed by its coordinate."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt

from eqcube.puzzles.base import PermutationPuzzle, States

IntArray = npt.NDArray[np.int64]
Int8Array = npt.NDArray[np.int8]


class StateIndexer(Protocol):
    """A bijection between a puzzle's states and 0..n_states-1, with moves on indices."""

    @property
    def puzzle(self) -> PermutationPuzzle: ...

    @property
    def n_states(self) -> int: ...

    @property
    def solved_index(self) -> int: ...

    def rank(self, states: States) -> IntArray: ...

    def unrank(self, index: IntArray) -> States: ...

    def move(self, index: IntArray, moves: IntArray) -> IntArray: ...

    def expand(self, index: IntArray) -> IntArray: ...


def bfs(indexer: StateIndexer) -> Int8Array:
    """Exact distance-to-solved for every index (BFS over move tables)."""
    dist = np.full(indexer.n_states, -1, dtype=np.int8)
    frontier = np.array([indexer.solved_index], dtype=np.int64)
    dist[frontier] = 0
    depth = 0
    while len(frontier):
        depth += 1
        children = np.unique(indexer.expand(frontier))
        frontier = children[dist[children] < 0]
        dist[frontier] = depth
    if (dist < 0).any():
        raise RuntimeError("BFS did not reach every state")
    return dist


@dataclass(frozen=True, eq=False)
class DistanceTable:
    indexer: StateIndexer
    dist: Int8Array  # dist[i] = distance of state i

    def __post_init__(self) -> None:
        if self.dist.shape != (self.indexer.n_states,):
            raise ValueError("distance table does not match the indexer")

    @classmethod
    def build(cls, indexer: StateIndexer, cache_dir: str | Path | None = None) -> DistanceTable:
        """BFS, or load ``<cache_dir>/<puzzle>_distances.npy`` if it exists."""
        if cache_dir is None:
            return cls(indexer, bfs(indexer))
        path = Path(cache_dir) / f"{indexer.puzzle.name}_distances.npy"
        if path.exists():
            return cls(indexer, np.load(path))
        table = cls(indexer, bfs(indexer))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, table.dist)
        return table

    @property
    def max_distance(self) -> int:
        return int(self.dist.max())

    @cached_property
    def counts(self) -> IntArray:
        """Number of states at each distance."""
        return np.bincount(self.dist).astype(np.int64)

    def distance(self, states: States) -> IntArray:
        return self.dist[self.indexer.rank(states)].astype(np.int64)

    def labels(self, index: IntArray) -> IntArray:
        return self.dist[index].astype(np.int64)

    def materialize(self, index: IntArray) -> tuple[States, IntArray]:
        """Indices -> (sticker states, distance labels), ready for a model."""
        return self.indexer.unrank(index), self.labels(index)
