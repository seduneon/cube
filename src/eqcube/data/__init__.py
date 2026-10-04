"""Exact labels, a fixed partition, training samplers and evaluation suites.

NumPy only, never torch. Data flow:

    DistanceTable (BFS once)  ->  Partition (train/val/test by hash)
        -> samplers (train pool)   -> materialize -> (states, labels)
        -> suites   (test/val)     -> materialize -> (states, labels)
"""

from eqcube.data.partition import SPLITS, Partition, PartitionConfig, Split
from eqcube.data.samplers import RandomWalkSampler, Sampler, TemperedSampler
from eqcube.data.suites import (
    EvalSuite,
    default_suites,
    per_depth_suite,
    search_suite,
    uniform_suite,
)
from eqcube.data.symmetry import apply_symmetry, orbit_representatives
from eqcube.data.table import DistanceTable, StateIndexer, bfs

__all__ = [
    "SPLITS",
    "DistanceTable",
    "EvalSuite",
    "Partition",
    "PartitionConfig",
    "RandomWalkSampler",
    "Sampler",
    "Split",
    "StateIndexer",
    "TemperedSampler",
    "apply_symmetry",
    "bfs",
    "default_suites",
    "orbit_representatives",
    "per_depth_suite",
    "search_suite",
    "uniform_suite",
]
