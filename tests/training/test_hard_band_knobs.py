"""Knobs added for the hard-band diagnosis: sampler weights/floor, balanced loss, parity."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from eqcube.data import DistanceTable, Partition, TemperedSampler
from eqcube.experiments.hard_band import variants
from eqcube.models import PRESETS, build_model
from eqcube.models.parity import CornerParity
from eqcube.puzzles.pocket_coords import pocket_coords
from eqcube.puzzles.pocket_cube import task_group
from eqcube.symmetry.action import FunctionAction
from eqcube.training import TrainConfig
from eqcube.training.losses import class_weights

C = pocket_coords()


@pytest.fixture(scope="module")
def table() -> DistanceTable:
    return DistanceTable.build(C)


@pytest.fixture(scope="module")
def partition(table: DistanceTable) -> Partition:
    return Partition(table)


def test_depth_weights_shift_probability(partition: Partition) -> None:
    base = TemperedSampler(partition, 0.5).depth_probabilities()
    boost = tuple(3.0 if d in (7, 8, 9, 10) else 1.0 for d in range(15))
    boosted = TemperedSampler(partition, 0.5, weights=boost).depth_probabilities()
    assert boosted[8] > base[8] and boosted[12] < base[12]
    assert abs(boosted.sum() - 1) < 1e-12
    with pytest.raises(ValueError):
        TemperedSampler(partition, 0.5, weights=(1.0,)).depth_probabilities()


def test_full_depth_floor(partition: Partition) -> None:
    sizes = partition.pool_sizes("train")
    quota = TemperedSampler(partition, 1.0, full_depth=5).depth_quota(100_000)
    assert (quota[:6] == sizes[:6]).all()  # alpha = 1 alone would drop depth 0 and 1
    assert quota.sum() == 100_000
    plain = TemperedSampler(partition, 1.0).depth_quota(100_000)
    assert plain[0] == 0
    with pytest.raises(ValueError):
        TemperedSampler(partition, 1.0, full_depth=12).depth_quota(1000)


def test_class_weights_balance_the_loss() -> None:
    labels = torch.tensor([0, 1, 1, 1, 3])
    w = class_weights(labels, 5)
    contribution = torch.bincount(labels, minlength=5).float() * w
    present = contribution[contribution > 0]
    assert torch.allclose(present, present[0].expand_as(present))
    assert w[2] == 0 and w[4] == 0
    assert torch.isclose(w[labels].mean(), torch.tensor(1.0))


def test_index_parity_is_distance_parity(table: DistanceTable) -> None:
    index = np.random.default_rng(0).integers(0, C.n_states, 100_000)
    np.testing.assert_array_equal(C.parity(index), table.dist[index] % 2)


def test_torch_parity_matches_index_parity() -> None:
    index = np.random.default_rng(1).integers(0, C.n_states, 20_000)
    states = torch.as_tensor(C.unrank(index))
    np.testing.assert_array_equal(CornerParity(C)(states).numpy(), C.parity(index))


def test_parity_mask_removes_wrong_parity(table: DistanceTable) -> None:
    torch.manual_seed(0)
    model = build_model(replace(PRESETS["task"], parity_mask=True)).eval()
    index = np.random.default_rng(2).integers(0, C.n_states, 500)
    with torch.no_grad():
        p = model(torch.as_tensor(C.unrank(index))).softmax(dim=1).numpy()
    wrong = (np.arange(15)[None, :] % 2) != (table.dist[index] % 2)[:, None]
    assert p[wrong].max() < 1e-6


def test_parity_mask_keeps_task_invariance() -> None:
    torch.manual_seed(0)
    model = build_model(replace(PRESETS["task"], parity_mask=True)).eval()
    states = C.unrank(np.random.default_rng(3).integers(0, C.n_states, 64))
    action = FunctionAction(task_group(), n_domain=24)
    with torch.no_grad():
        base = model(torch.as_tensor(states))
        for g in task_group().elements:
            moved = np.array([action(g, tuple(int(c) for c in s)) for s in states], dtype=np.int8)
            torch.testing.assert_close(model(torch.as_tensor(moved)), base, atol=1e-3, rtol=1e-4)


def test_train_config_roundtrip_with_new_fields() -> None:
    cfg = TrainConfig(depth_weights=(1.0, 2.0), full_depth=5, class_balance=True)
    d = cfg.to_dict()
    d["depth_weights"] = list(d["depth_weights"])  # as after a JSON round trip
    assert TrainConfig.from_dict(d) == cfg


def test_hard_band_variants_are_distinct() -> None:
    specs = variants()
    assert len({s.run_id for s in specs}) == len(specs) == 7
    assert {s.tag for s in specs} >= {"baseline", "balanced", "parity", "n1m"}
