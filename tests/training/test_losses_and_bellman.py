"""Loss strategies and the Bellman consistency term."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from eqcube.data import DistanceTable, Partition, TemperedSampler
from eqcube.inference import backup_log_probs
from eqcube.models import PRESETS, build_model
from eqcube.puzzles.pocket_coords import pocket_coords
from eqcube.training import TrainConfig, Trainer
from eqcube.training.bellman import BellmanConsistency, backup_probs
from eqcube.training.losses import make_loss, ordinal_bce_loss

C = pocket_coords()


@pytest.fixture(scope="module")
def table() -> DistanceTable:
    return DistanceTable.build(C)


@pytest.fixture(scope="module")
def partition(table: DistanceTable) -> Partition:
    return Partition(table)


def onehot_log_probs(labels: torch.Tensor, c: int = 15) -> torch.Tensor:
    p = torch.full((len(labels), c), 1e-6)
    p[torch.arange(len(labels)), labels] = 1.0
    return (p / p.sum(1, keepdim=True)).log()


def test_ordinal_bce_penalizes_far_errors_more() -> None:
    labels = torch.tensor([10])
    perfect = ordinal_bce_loss(onehot_log_probs(torch.tensor([10])), labels)
    near = ordinal_bce_loss(onehot_log_probs(torch.tensor([9])), labels)
    far = ordinal_bce_loss(onehot_log_probs(torch.tensor([2])), labels)
    assert perfect < 1e-3 < near < far


def test_make_loss() -> None:
    assert make_loss("ce") is not make_loss("ordinal_bce")
    with pytest.raises(ValueError):
        make_loss("nope")  # type: ignore[arg-type]


def test_old_balanced_ce_config_still_loads() -> None:
    cfg = TrainConfig.from_dict({**TrainConfig().to_dict(), "loss": "balanced_ce"})
    assert cfg.loss == "ce" and cfg.class_balance


def test_torch_and_numpy_backups_agree() -> None:
    rng = np.random.default_rng(0)
    child = rng.normal(size=(40, 6, 15))
    child_lp = child - np.log(np.exp(child).sum(-1, keepdims=True))
    solved = rng.random(40) < 0.1
    torch_p = backup_probs(torch.as_tensor(child_lp), torch.as_tensor(solved)).numpy()
    np.testing.assert_allclose(torch_p, np.exp(backup_log_probs(child_lp, solved)), atol=1e-6)


def test_backup_of_exact_children_is_exact(table: DistanceTable) -> None:
    """If every child's distribution is exact, the Bellman backup is exact too."""
    index = np.random.default_rng(1).integers(0, C.n_states, 2000)
    children = C.expand(index)
    child_lp = onehot_log_probs(torch.as_tensor(table.labels(children.reshape(-1))))
    child_lp = child_lp.view(len(index), 6, 15)
    p = backup_probs(child_lp, torch.as_tensor(index == C.solved_index))
    assert (p.argmax(dim=1).numpy() == table.labels(index)).all()


def test_consistency_loss_and_ema(table: DistanceTable, partition: Partition) -> None:
    torch.manual_seed(0)
    model = build_model(PRESETS["task"])
    term = BellmanConsistency(
        model, table, TemperedSampler(partition, 0.5), 64, 0.5, torch.device("cpu"), 0
    )
    loss = term.loss(model)
    assert torch.isfinite(loss) and loss.item() >= -1e-6
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
    assert all(p.grad is None for p in term.target.parameters())
    before = [t.clone() for t in term.target.parameters()]
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    term.update_target(model)
    for b, t, p in zip(before, term.target.parameters(), model.parameters(), strict=True):
        torch.testing.assert_close(t, 0.5 * b + 0.5 * p)  # lerp with decay 0.5


@pytest.mark.parametrize(
    "change",
    [{"loss": "ordinal_bce"}, {"class_balance": True}, {"consistency_weight": 1.0}],
)
def test_fit_with_new_options(
    table: DistanceTable, partition: Partition, tmp_path: Path, change: dict[str, object]
) -> None:
    cfg = replace(
        TrainConfig(train_size=2000, max_epochs=2, patience=2, batch_size=256, device="cpu"),
        consistency_batch=64,
        **change,  # type: ignore[arg-type]
    )
    model_cfg = replace(PRESETS["task"], head="ordinal")
    result = Trainer(model_cfg, cfg, table, partition).fit(tmp_path)
    assert np.isfinite(result.best_val_ce) and result.epochs_run == 2
