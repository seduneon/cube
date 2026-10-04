"""Trainer behaviour on tiny runs (CPU), config round trips, device selection."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from eqcube.data import DistanceTable, Partition, PartitionConfig
from eqcube.experiments.runner import EvalConfig, Workspace, grid, run_one
from eqcube.experiments.summarize import band_summary, summarize
from eqcube.models import PRESETS, load_checkpoint
from eqcube.puzzles.pocket_coords import pocket_coords
from eqcube.training import TrainConfig, Trainer, resolve_device, training_indices

TINY = TrainConfig(train_size=3000, max_epochs=3, patience=3, batch_size=256, device="cpu")


@pytest.fixture(scope="module")
def table() -> DistanceTable:
    return DistanceTable.build(pocket_coords())


@pytest.fixture(scope="module")
def partition(table: DistanceTable) -> Partition:
    return Partition(table, TINY.partition)


def test_train_config_roundtrip() -> None:
    cfg = replace(TINY, partition=PartitionConfig(policy="orbit", fractions=(0.7, 0.2, 0.1)))
    assert TrainConfig.from_dict(cfg.to_dict()) == cfg


def test_resolve_device() -> None:
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device("auto").type in {"cpu", "cuda", "mps"}


def test_training_indices(partition: Partition) -> None:
    a = training_indices(TINY, partition)
    assert len(a) == 3000 == len(np.unique(a))
    assert partition.contains("train", a).all()
    np.testing.assert_array_equal(a, training_indices(TINY, partition))
    walk = training_indices(replace(TINY, sampler="random_walk"), partition)
    assert partition.contains("train", walk).all()


def test_fit_writes_checkpoint_and_learns(
    table: DistanceTable, partition: Partition, tmp_path: Path
) -> None:
    result = Trainer(PRESETS["task"], TINY, table, partition).fit(tmp_path)
    assert result.epochs_run == 3 and len(result.history) == 3
    assert result.history[-1]["train_loss"] < result.history[0]["train_loss"]
    _, cfg, meta = load_checkpoint(tmp_path / "best.pt")
    assert cfg == PRESETS["task"]
    assert TrainConfig.from_dict(meta["train_config"]) == TINY
    assert meta["epoch"] == result.best_epoch
    assert (tmp_path / "history.json").exists()


def test_fit_is_deterministic(table: DistanceTable, partition: Partition, tmp_path: Path) -> None:
    a = Trainer(PRESETS["mlp_small"], TINY, table, partition).fit(tmp_path / "a")
    b = Trainer(PRESETS["mlp_small"], TINY, table, partition).fit(tmp_path / "b")
    assert [r["train_loss"] for r in a.history] == [r["train_loss"] for r in b.history]


def test_early_stopping(table: DistanceTable, partition: Partition, tmp_path: Path) -> None:
    frozen = replace(TINY, lr=0.0, lr_min=0.0, max_epochs=10, patience=2)
    result = Trainer(PRESETS["mlp_small"], frozen, table, partition).fit(tmp_path)
    assert result.epochs_run == 3  # best at epoch 0, then 2 epochs without improvement


def test_partition_mismatch_rejected(table: DistanceTable, partition: Partition) -> None:
    other = replace(TINY, partition=PartitionConfig(seed=9))
    with pytest.raises(ValueError):
        Trainer(PRESETS["task"], other, table, partition)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple GPU")
def test_fit_on_mps(table: DistanceTable, partition: Partition, tmp_path: Path) -> None:
    cfg = replace(TINY, max_epochs=1, device="mps")
    result = Trainer(PRESETS["task"], cfg, table, partition).fit(tmp_path)
    assert np.isfinite(result.best_val_ce)


def test_run_one_end_to_end_and_resume(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "cache")
    quick = EvalConfig(
        sibling_per_depth=10, equivariance_states=20, beam_max_steps=8, backup_per_depth=20
    )
    (spec,) = grid(["task"], [2000], [0.5], [1], replace(TINY, max_epochs=2), quick)
    result = run_one(spec, ws, tmp_path / "runs", verbose=False)
    assert (tmp_path / "runs" / spec.run_id / "result.json").exists()
    assert result["eval"]["equivariance"]["task"]["max"] < 1e-4
    assert 0.0 <= result["eval"]["search/expected"]["solve_rate"] <= 1.0
    again = run_one(spec, ws, tmp_path / "runs", verbose=False)  # skipped, read from disk
    assert again["train_seconds"] == result["train_seconds"]
    assert "task/n2000_a0.5_random" in summarize(tmp_path / "runs")
    cal, backup = result["eval"]["calibration"], result["eval"]["backup"]
    assert cal["temperature"] > 0 and cal["tempered"]["nll"] <= cal["raw"]["nll"] + 0.05
    assert backup["w"] + backup["beta"] > 0
    assert {"base", "backup", "siblings/base", "siblings/backup"} <= set(backup)
    assert "| task |" in band_summary(tmp_path / "runs")


def test_grid_is_a_product() -> None:
    specs = grid(["task", "mlp"], [10, 20, 30], [0.0, 1.0], [1, 2])
    assert len(specs) == 2 * 3 * 2 * 2
    assert len({s.run_id for s in specs}) == len(specs)


def test_bellman_variants_are_distinct() -> None:
    from eqcube.experiments.bellman import variants

    specs = variants()
    assert len(specs) == 10 == len({s.run_id for s in specs})
    assert {s.model.head for s in specs} == {"categorical", "ordinal"}
