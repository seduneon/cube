"""Training configuration. Stored next to ModelConfig in every checkpoint."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import torch

from eqcube.data.partition import PartitionConfig
from eqcube.training.losses import LossName

SamplerName = Literal["tempered", "random_walk"]


@dataclass(frozen=True)
class TrainConfig:
    train_size: int = 200_000
    sampler: SamplerName = "tempered"
    alpha: float = 0.5  # tempered: p(d) ∝ n_d^alpha * depth_weights[d]
    depth_weights: tuple[float, ...] | None = None  # tempered: per-depth boost
    full_depth: int = -1  # tempered: depths <= full_depth are taken whole
    walk_length: int = 20  # random_walk: max scramble length
    loss: LossName = "ce"  # or "ordinal_bce" (see training/losses.py)
    class_balance: bool = False  # every depth weighs the same in the loss
    consistency_weight: float = 0.0  # > 0 adds Bellman consistency on unlabeled states
    consistency_batch: int = 512  # unlabeled states per step
    ema_decay: float = 0.995  # target network for consistency targets
    partition: PartitionConfig = field(default_factory=PartitionConfig)

    batch_size: int = 512
    max_epochs: int = 200
    patience: int = 15  # early stopping on validation cross-entropy
    lr: float = 1e-3
    lr_min: float = 1e-5
    weight_decay: float = 1e-4
    seed: int = 42
    device: str = "auto"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainConfig:
        d = dict(d)
        if d.get("loss") == "balanced_ce":  # older configs: balancing was a loss name
            d["loss"], d["class_balance"] = "ce", True
        if d.get("depth_weights") is not None:
            d["depth_weights"] = tuple(d["depth_weights"])
        p = dict(d.pop("partition", {}))
        if "fractions" in p:
            p["fractions"] = tuple(p["fractions"])
        return cls(partition=PartitionConfig(**p), **d)


def resolve_device(name: str = "auto") -> torch.device:
    """'auto' picks CUDA, then Apple MPS, then CPU."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
