"""Named, fixed evaluation suites drawn from one split.

Report metrics per depth whenever possible: any aggregate is then an
explicit weighting sum_d q(d) * metric(d), and the choice of q is visible.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from eqcube.data.partition import Partition, Split

IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class EvalSuite:
    name: str
    index: IntArray
    distance: IntArray

    def __len__(self) -> int:
        return len(self.index)


def per_depth_suite(
    partition: Partition, n_per_depth: int, rng: np.random.Generator, split: Split = "test"
) -> EvalSuite:
    """Up to ``n_per_depth`` distinct states at every depth (uniform over depths)."""
    parts = []
    for d in range(partition.table.max_distance + 1):
        pool = partition.pool(split, d)
        parts.append(rng.choice(pool, size=min(n_per_depth, len(pool)), replace=False))
    index = np.concatenate(parts)
    return EvalSuite(f"{split}/per_depth", index, partition.table.labels(index))


def uniform_suite(
    partition: Partition, n: int, rng: np.random.Generator, split: Split = "test"
) -> EvalSuite:
    """n distinct states uniform over the split (mirrors 'a random cube')."""
    index = rng.choice(np.flatnonzero(partition.mask(split)), size=n, replace=False)
    return EvalSuite(f"{split}/uniform", index, partition.table.labels(index))


def search_suite(
    partition: Partition,
    depths: Sequence[int],
    n_per_depth: int,
    rng: np.random.Generator,
    split: Split = "test",
) -> EvalSuite:
    """Start states for search: ``n_per_depth`` per requested depth."""
    parts = []
    for d in depths:
        pool = partition.pool(split, d)
        parts.append(rng.choice(pool, size=min(n_per_depth, len(pool)), replace=False))
    index = np.concatenate(parts)
    return EvalSuite(f"{split}/search", index, partition.table.labels(index))


def default_suites(partition: Partition, seed: int = 0) -> dict[str, EvalSuite]:
    """The standard evaluation set used by experiments (fixed by ``seed``)."""
    rng = np.random.default_rng(seed)
    suites = [
        per_depth_suite(partition, 2000, rng),
        uniform_suite(partition, 20_000, rng),
        search_suite(partition, [4, 7, 10, 12, 14], 200, rng),
        per_depth_suite(partition, 1000, rng, split="val"),
    ]
    return {s.name: s for s in suites}
