import numpy as np
import pytest

from eqcube.data import DistanceTable, Partition, PartitionConfig
from eqcube.puzzles.pocket_coords import pocket_coords


@pytest.fixture(scope="session")
def table() -> DistanceTable:
    return DistanceTable.build(pocket_coords())


@pytest.fixture(scope="session")
def partition(table: DistanceTable) -> Partition:
    return Partition(table, PartitionConfig(seed=0))


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)
