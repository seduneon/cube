"""A fixed train / val / test partition of the state space.

Every state index is assigned to one split by hashing it (or its orbit
representative) with a seed. The assignment never depends on how data is
later sampled, so any number of training samplers and evaluation suites can
coexist without leaking into each other.

Shared shallow states: states with distance <= ``shared_max_depth`` belong to
*every* split. There are very few of them (688 at depth <= 4), and they are
exhaustively enumerable, so held-out generalization is meaningless there.

Policies:
  random  each state is hashed on its own. A state and its symmetric images
          may land in different splits, so an equivariant model gets credit
          for symmetry ("how much does symmetry help").
  orbit   each state is hashed by its orbit representative, so whole orbits
          stay together ("does the model generalize beyond symmetry").
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import numpy as np
import numpy.typing as npt

from eqcube.data.table import DistanceTable

IntArray = npt.NDArray[np.int64]
Split = Literal["train", "val", "test"]
SPLITS: tuple[Split, ...] = ("train", "val", "test")


@dataclass(frozen=True)
class PartitionConfig:
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1)
    seed: int = 0
    policy: Literal["random", "orbit"] = "random"
    shared_max_depth: int = 4

    def __post_init__(self) -> None:
        if abs(sum(self.fractions) - 1.0) > 1e-9 or min(self.fractions) < 0:
            raise ValueError("fractions must be non-negative and sum to 1")


def uniform_hash(keys: IntArray, seed: int) -> npt.NDArray[np.float64]:
    """Deterministic pseudo-random number in [0, 1) per key (splitmix64)."""
    with np.errstate(over="ignore"):
        z = keys.astype(np.uint64) + np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) / float(1 << 53)


class Partition:
    def __init__(
        self,
        table: DistanceTable,
        config: PartitionConfig | None = None,
        orbit_reps: IntArray | None = None,
    ) -> None:
        config = config or PartitionConfig()
        if config.policy == "orbit" and orbit_reps is None:
            raise ValueError("policy 'orbit' needs orbit representatives")
        self.table, self.config = table, config
        keys = orbit_reps if config.policy == "orbit" else np.arange(len(table.dist))
        assert keys is not None
        u = uniform_hash(keys, config.seed)
        edges = np.cumsum(config.fractions)[:-1]
        self.assignment = np.searchsorted(edges, u, side="right").astype(np.int8)  # 0/1/2
        self.shared = table.dist <= config.shared_max_depth

    def mask(self, split: Split) -> npt.NDArray[np.bool_]:
        return (self.assignment == SPLITS.index(split)) | self.shared

    def contains(self, split: Split, index: IntArray) -> npt.NDArray[np.bool_]:
        return (self.assignment[index] == SPLITS.index(split)) | self.shared[index]

    @cached_property
    def _pools(self) -> dict[Split, list[IntArray]]:
        """pools[split][d] = sorted indices of the split at distance d."""
        pools: dict[Split, list[IntArray]] = {}
        for split in SPLITS:
            index = np.flatnonzero(self.mask(split))
            depth = self.table.dist[index]
            order = np.argsort(depth, kind="stable")
            bounds = np.searchsorted(depth[order], np.arange(self.table.max_distance + 2))
            pools[split] = [
                index[order[bounds[d] : bounds[d + 1]]] for d in range(self.table.max_distance + 1)
            ]
        return pools

    def pool(self, split: Split, depth: int) -> IntArray:
        return self._pools[split][depth]

    def pool_sizes(self, split: Split) -> IntArray:
        return np.array([len(p) for p in self._pools[split]], dtype=np.int64)
