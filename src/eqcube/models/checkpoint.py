"""Checkpoints = config + weights (+ free-form metadata). No shape guessing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from eqcube.models.config import ModelConfig, build_model


def save_checkpoint(path: str | Path, model: nn.Module, cfg: ModelConfig, **meta: Any) -> None:
    torch.save({"config": cfg.to_dict(), "state_dict": model.state_dict(), "meta": meta}, path)


def load_checkpoint(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> tuple[nn.Module, ModelConfig, dict[str, Any]]:
    data = torch.load(path, map_location=map_location, weights_only=True)
    cfg = ModelConfig.from_dict(data["config"])
    model = build_model(cfg)
    model.load_state_dict(data["state_dict"])
    return model, cfg, data["meta"]
