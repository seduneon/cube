"""Training configuration. Stored next to ModelConfig in every checkpoint."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import torch

from eqcube.data.partition import PartitionConfig

SamplerName = Literal["tempered", "random_walk"]


@dataclass(frozen=True)
class TrainConfig:
    train_size: int = 200_000
    sampler: SamplerName = "tempered"
    alpha: float = 0.5  # tempered: p(d) ∝ n_d^alpha
    walk_length: int = 20  # random_walk: max scramble length
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
