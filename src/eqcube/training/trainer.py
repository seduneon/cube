"""Supervised training of value networks on exact distance labels.

    Trainer(model_cfg, train_cfg, table, partition).fit(out_dir) -> TrainResult

* The training set is a fixed sample from the train split (``train_size``
  distinct states for the tempered sampler), materialized on the device once.
* Validation uses the per-depth val suite (uniform over depths), so early
  stopping does not just track the most common depths.
* The best checkpoint (by validation cross-entropy) stores the model config,
  the train config, the epoch and the validation metrics.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from torch import nn

from eqcube.data.partition import Partition
from eqcube.data.samplers import RandomWalkSampler, TemperedSampler
from eqcube.data.suites import EvalSuite, per_depth_suite
from eqcube.data.table import DistanceTable
from eqcube.models.checkpoint import save_checkpoint
from eqcube.models.config import ModelConfig, build_model
from eqcube.training.config import TrainConfig, resolve_device

IntArray = npt.NDArray[np.int64]
Logger = Callable[[dict[str, Any]], None]


@dataclass
class TrainResult:
    best_epoch: int
    best_val_ce: float
    epochs_run: int
    seconds: float
    n_params: int
    history: list[dict[str, Any]] = field(default_factory=list)


def training_indices(cfg: TrainConfig, partition: Partition) -> IntArray:
    """The fixed training set for ``cfg`` (deterministic in cfg.seed)."""
    rng = np.random.default_rng(cfg.seed)
    if cfg.sampler == "tempered":
        return TemperedSampler(partition, cfg.alpha).sample_unique(cfg.train_size, rng)
    return RandomWalkSampler(partition, cfg.walk_length).sample(cfg.train_size, rng)


def validation_suite(partition: Partition, seed: int = 0) -> EvalSuite:
    return per_depth_suite(partition, 1000, np.random.default_rng(seed), split="val")


def _ignore(_: dict[str, Any]) -> None:
    pass


class Trainer:
    def __init__(
        self,
        model_cfg: ModelConfig,
        train_cfg: TrainConfig,
        table: DistanceTable,
        partition: Partition,
        log: Logger | None = None,
    ) -> None:
        if partition.config != train_cfg.partition:
            raise ValueError("partition does not match train_cfg.partition")
        self.model_cfg, self.cfg = model_cfg, train_cfg
        self.table, self.partition = table, partition
        self.device = resolve_device(train_cfg.device)
        self.log: Logger = log or _ignore

    def _tensors(self, index: IntArray) -> tuple[torch.Tensor, torch.Tensor]:
        states, labels = self.table.materialize(index)
        return (
            torch.as_tensor(states, device=self.device),
            torch.as_tensor(labels, device=self.device),
        )

    @torch.no_grad()
    def _validate(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        model.eval()
        total_ce, correct = 0.0, 0
        for start in range(0, len(x), 4096):
            logits = model(x[start : start + 4096])
            target = y[start : start + 4096]
            total_ce += float(F.cross_entropy(logits, target, reduction="sum"))
            correct += int((logits.argmax(dim=1) == target).sum())
        return {"val_ce": total_ce / len(x), "val_accuracy": correct / len(x)}

    def fit(self, out_dir: str | Path) -> TrainResult:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        gen = torch.Generator().manual_seed(cfg.seed)

        x_train, y_train = self._tensors(training_indices(cfg, self.partition))
        x_val, y_val = self._tensors(validation_suite(self.partition).index)

        model = build_model(self.model_cfg).to(self.device)
        n_params = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg.max_epochs, eta_min=cfg.lr_min
        )

        best_ce, best_epoch, waited = float("inf"), -1, 0
        history: list[dict[str, Any]] = []
        start_time = time.time()
        epoch = 0
        for epoch in range(cfg.max_epochs):
            t0 = time.time()
            model.train()
            order = torch.randperm(len(x_train), generator=gen).to(self.device)
            total, seen = 0.0, 0
            for start in range(0, len(order), cfg.batch_size):
                batch = order[start : start + cfg.batch_size]
                loss = F.cross_entropy(model(x_train[batch]), y_train[batch])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += float(loss.detach()) * len(batch)
                seen += len(batch)
            sched.step()

            row: dict[str, Any] = {
                "epoch": epoch,
                "train_ce": total / seen,
                **self._validate(model, x_val, y_val),
                "lr": sched.get_last_lr()[0],
                "seconds": time.time() - t0,
            }
            history.append(row)
            self.log(row)

            if row["val_ce"] < best_ce:
                best_ce, best_epoch, waited = row["val_ce"], epoch, 0
                save_checkpoint(
                    out / "best.pt",
                    model,
                    self.model_cfg,
                    train_config=cfg.to_dict(),
                    epoch=epoch,
                    val_ce=row["val_ce"],
                    val_accuracy=row["val_accuracy"],
                )
            else:
                waited += 1
                if waited >= cfg.patience:
                    break

        result = TrainResult(
            best_epoch=best_epoch,
            best_val_ce=best_ce,
            epochs_run=epoch + 1,
            seconds=time.time() - start_time,
            n_params=n_params,
            history=history,
        )
        (out / "history.json").write_text(json.dumps(history, indent=1))
        return result
