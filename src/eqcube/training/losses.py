"""Losses on log-probabilities (B, C) and integer distance labels (B,).

    ce            negative log-likelihood of the true class. With the ordinal
                  head this *is* the CORN loss (sum of the conditional
                  binary losses up to the true class).
    ordinal_bce   sum over k of BCE(P(d > k), [label > k]) (Frank & Hall
                  2001; Niu et al. 2016). A prediction far from the label
                  gets more of the questions wrong, so it costs more.

``weight`` (optional, per class) rescales each sample by the weight of its
label, e.g. to make every depth count equally (``class_weights``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

LossName = Literal["ce", "ordinal_bce"]


def class_weights(labels: Tensor, num_classes: int) -> Tensor:
    """Weights that make every present class contribute equally to the mean loss."""
    counts = torch.bincount(labels, minlength=num_classes).float()
    present = counts > 0
    weights = torch.zeros(num_classes, device=labels.device)
    weights[present] = len(labels) / (present.sum() * counts[present])
    return weights


def survival(log_probs: Tensor) -> Tensor:
    """S(k) = P(d > k) for k = 0..C-2, from log-probabilities."""
    cdf = torch.cumsum(log_probs.exp(), dim=-1)
    return (1.0 - cdf[..., :-1]).clamp(1e-7, 1.0 - 1e-7)


def ce_loss(log_probs: Tensor, labels: Tensor, weight: Tensor | None = None) -> Tensor:
    return F.nll_loss(log_probs, labels, weight=weight)


def ordinal_bce_loss(log_probs: Tensor, labels: Tensor, weight: Tensor | None = None) -> Tensor:
    s = survival(log_probs)  # (B, C-1)
    k = torch.arange(s.shape[-1], device=labels.device)
    target = (labels[:, None] > k[None, :]).to(s.dtype)
    per_sample = F.binary_cross_entropy(s, target, reduction="none").sum(dim=-1)
    if weight is None:
        return per_sample.mean()
    w = weight[labels]
    return (w * per_sample).sum() / w.sum()


LossFn = Callable[[Tensor, Tensor, Tensor | None], Tensor]
LOSSES: dict[str, LossFn] = {"ce": ce_loss, "ordinal_bce": ordinal_bce_loss}


def make_loss(name: LossName) -> LossFn:
    if name not in LOSSES:
        raise ValueError(f"unknown loss {name!r}; known: {sorted(LOSSES)}")
    return LOSSES[name]
