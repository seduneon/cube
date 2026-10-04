"""Output heads: features -> log-probabilities over distance classes 0..C-1.

Every head returns *log-probabilities* (each row log-sums to 0), so search,
evaluation and the parity mask never need to know which head is used.

    CategoricalHead   one logit per class, log-softmax
    OrdinalHead       CORN survival head (Shi, Cao, Raschka 2023):
                        z_k = logit P(d > k | d > k-1),      k = 0..C-2
                        S(k) = P(d > k) = prod_{j<=k} sigmoid(z_j)   (always decreasing)
                        P(d = k) = S(k-1) - S(k),  P(d = C-1) = S(C-2)

Both subclass ``nn.Linear`` so checkpoint keys stay ``...head.weight/bias``.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

HeadName = Literal["categorical", "ordinal"]


class CategoricalHead(nn.Linear):
    def __init__(self, in_features: int, num_classes: int) -> None:
        super().__init__(in_features, num_classes)

    def forward(self, input: Tensor) -> Tensor:
        return F.log_softmax(super().forward(input), dim=-1)


class OrdinalHead(nn.Linear):
    def __init__(self, in_features: int, num_classes: int) -> None:
        super().__init__(in_features, num_classes - 1)

    def forward(self, input: Tensor) -> Tensor:
        z = super().forward(input)  # (B, C-1)
        log_s = torch.cumsum(F.logsigmoid(z), dim=-1)  # log S(0..C-2)
        log_s_prev = F.pad(log_s[..., :-1], (1, 0))  # log S(-1) = 0, log S(0..C-3)
        # P(d = k) = S(k-1) * P(stop at k | reached k) = S(k-1) * sigmoid(-z_k)
        return torch.cat([log_s_prev + F.logsigmoid(-z), log_s[..., -1:]], dim=-1)


def make_head(name: HeadName, in_features: int, num_classes: int) -> nn.Module:
    if name == "categorical":
        return CategoricalHead(in_features, num_classes)
    if name == "ordinal":
        return OrdinalHead(in_features, num_classes)
    raise ValueError(f"unknown head {name!r}")
