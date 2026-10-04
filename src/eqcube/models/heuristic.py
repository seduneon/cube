"""Turn a value network into a search heuristic (states -> estimated cost)."""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

Reduce = Literal["argmax", "expected"]


class NetHeuristic:
    """Batched, no-grad evaluation of a ValueNet.

    reduce="argmax":   most likely distance (integer; many ties)
    reduce="expected": sum_k k * p(k) (smooth; usually ranks children better)
    """

    def __init__(
        self,
        model: nn.Module,
        reduce: Reduce = "argmax",
        device: str | torch.device = "cpu",
        batch_size: int = 8192,
    ) -> None:
        self.model = model.to(device).eval()
        self.reduce, self.device, self.batch_size = reduce, torch.device(device), batch_size

    @torch.inference_mode()
    def __call__(self, states: npt.NDArray[np.int8]) -> npt.NDArray[np.float64]:
        out: list[npt.NDArray[np.float64]] = []
        for start in range(0, len(states), self.batch_size):
            x = torch.as_tensor(states[start : start + self.batch_size], device=self.device)
            logits = self.model(x)
            if self.reduce == "argmax":
                cost = logits.argmax(dim=1).double()
            else:
                k = torch.arange(logits.shape[1], device=self.device, dtype=torch.float64)
                cost = logits.double().softmax(dim=1) @ k
            out.append(cost.cpu().numpy())
        return np.concatenate(out) if out else np.empty(0)
