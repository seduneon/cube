"""Adapters from a value network to NumPy callables.

    NetPredictor   states -> logits          (used by evaluation)
    NetHeuristic   states -> estimated cost  (used by search)

Both run batched under ``inference_mode`` on any device. The math after the
forward pass happens in NumPy float64 (MPS has no float64).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

Reduce = Literal["argmax", "expected"]
FloatArray = npt.NDArray[np.float64]


class NetPredictor:
    def __init__(
        self, model: nn.Module, device: str | torch.device = "cpu", batch_size: int = 8192
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.batch_size = batch_size

    @torch.inference_mode()
    def __call__(self, states: npt.NDArray[np.int8]) -> FloatArray:
        out: list[FloatArray] = []
        for start in range(0, len(states), self.batch_size):
            x = torch.as_tensor(states[start : start + self.batch_size], device=self.device)
            out.append(self.model(x).float().cpu().numpy().astype(np.float64))
        return np.concatenate(out) if out else np.empty((0, 0))


def costs_from_logits(logits: FloatArray, reduce: Reduce) -> FloatArray:
    """argmax: most likely distance. expected: sum_k k * softmax(logits)_k."""
    if reduce == "argmax":
        return logits.argmax(axis=1).astype(np.float64)
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    return p @ np.arange(logits.shape[1], dtype=np.float64)


class NetHeuristic:
    """reduce="argmax" (v1 behaviour, many ties) or "expected" (smooth ranking)."""

    reduce: Reduce

    def __init__(
        self,
        model: nn.Module,
        reduce: Reduce = "argmax",
        device: str | torch.device = "cpu",
        batch_size: int = 8192,
    ) -> None:
        self.predict = NetPredictor(model, device, batch_size)
        self.reduce = reduce

    def __call__(self, states: npt.NDArray[np.int8]) -> FloatArray:
        if len(states) == 0:
            return np.empty(0)
        return costs_from_logits(self.predict(states), self.reduce)
