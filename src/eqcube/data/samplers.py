"""Training samplers: strategies for choosing state indices from one split.

Every sampler draws only from its split of a ``Partition``, so switching
samplers never touches the evaluation data.

  TemperedSampler(alpha)   p(depth) ∝ n_depth^alpha * w_depth, uniform within a depth
                           alpha = 1: uniform over states
                           alpha = 0.5: sqrt weighting (v1 default)
                           alpha = 0: uniform over depths
                           weights w: optional per-depth boost (default 1)
                           full_depth: depths <= full_depth are taken whole
                                       (sample_unique), so even alpha = 1
                                       sees the solved state
  RandomWalkSampler        random scrambles of 1..max_length moves, kept
                           only if they land in the split (labels still come
                           from the exact table)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from eqcube.data.partition import Partition, Split

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]


class Sampler(Protocol):
    def sample(self, n: int, rng: np.random.Generator) -> IntArray:
        """n state indices (may repeat)."""
        ...


@dataclass(frozen=True)
class TemperedSampler:
    partition: Partition
    alpha: float = 0.5
    split: Split = "train"
    weights: tuple[float, ...] | None = None  # per-depth multiplier
    full_depth: int = -1  # sample_unique takes depths <= full_depth whole

    def depth_probabilities(self) -> FloatArray:
        sizes = self.partition.pool_sizes(self.split).astype(np.float64)
        weights = np.where(sizes > 0, sizes**self.alpha, 0.0)
        if self.weights is not None:
            if len(self.weights) != len(sizes):
                raise ValueError(f"need {len(sizes)} depth weights, got {len(self.weights)}")
            weights = weights * np.asarray(self.weights, dtype=np.float64)
        return weights / weights.sum()

    def sample(self, n: int, rng: np.random.Generator) -> IntArray:
        depths = rng.choice(len(self.depth_probabilities()), size=n, p=self.depth_probabilities())
        out = np.empty(n, dtype=np.int64)
        for d in np.unique(depths):
            slots = np.flatnonzero(depths == d)
            pool = self.partition.pool(self.split, int(d))
            out[slots] = pool[rng.integers(0, len(pool), size=len(slots))]
        return out

    def depth_quota(self, n: int) -> IntArray:
        """How many distinct states ``sample_unique(n)`` takes from each depth."""
        sizes = self.partition.pool_sizes(self.split)
        if n > sizes.sum():
            raise ValueError(f"asked for {n} distinct states, split has {sizes.sum()}")
        floor = np.where(np.arange(len(sizes)) <= self.full_depth, sizes, 0)
        if floor.sum() > n:
            raise ValueError(f"full_depth={self.full_depth} alone needs {floor.sum()} > {n}")
        p = np.where(floor > 0, 0.0, self.depth_probabilities())
        rest = _allocate(n - int(floor.sum()), p / p.sum(), sizes - floor)
        return floor + rest

    def sample_unique(self, n: int, rng: np.random.Generator) -> IntArray:
        """n *distinct* indices, as close to the tempered depth mix as pools allow.

        Depths whose pool is smaller than their quota are taken whole, and the
        remainder is redistributed over the other depths in proportion.
        """
        quota = self.depth_quota(n)
        parts = [
            rng.choice(self.partition.pool(self.split, d), size=int(q), replace=False)
            for d, q in enumerate(quota)
            if q > 0
        ]
        out = np.concatenate(parts)
        rng.shuffle(out)
        return out


def _allocate(n: int, p: FloatArray, caps: IntArray) -> IntArray:
    """Integer quotas summing to n, proportional to p, never above caps."""
    quota = np.zeros(len(p), dtype=np.int64)
    free = caps > 0
    remaining = n
    while remaining > 0:
        share = np.where(free, p, 0.0)
        share = share / share.sum()
        want = np.floor(share * remaining).astype(np.int64)
        want[np.argmax(share)] += remaining - want.sum()  # rounding leftovers
        take = np.minimum(want, caps - quota)
        quota += take
        remaining -= int(take.sum())
        free &= quota < caps
    return quota


@dataclass(frozen=True)
class RandomWalkSampler:
    partition: Partition
    max_length: int = 20
    split: Split = "train"

    def sample(self, n: int, rng: np.random.Generator) -> IntArray:
        indexer = self.partition.table.indexer
        n_moves = indexer.puzzle.n_moves
        inverse = np.array(indexer.puzzle.inverse_move, dtype=np.int64)
        out: list[IntArray] = []
        have = 0
        while have < n:
            batch = 2 * (n - have) + 16
            index = np.full(batch, indexer.solved_index, dtype=np.int64)
            length = rng.integers(1, self.max_length + 1, size=batch)
            last = np.full(batch, -1, dtype=np.int64)
            for step in range(self.max_length):
                active = step < length
                high = np.where(last >= 0, n_moves - 1, n_moves)
                k = rng.integers(0, high)
                k = k + (k >= np.where(last >= 0, inverse[np.maximum(last, 0)], n_moves))
                index = np.where(active, indexer.move(index, k), index)
                last = np.where(active, k, last)
            kept = index[self.partition.contains(self.split, index)]
            out.append(kept)
            have += len(kept)
        return np.concatenate(out)[:n]
