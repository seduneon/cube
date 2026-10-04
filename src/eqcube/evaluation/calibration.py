"""Calibration metrics for predicted distance distributions (NumPy only).

nll          mean negative log-likelihood of the true distance
ece          expected calibration error of the top class (15 equal bins):
             |accuracy - confidence| averaged over bins, weighted by size
brier        mean squared error between the probability vector and one-hot
ordinal_brier  mean over k of (P(d > k) - [d > k])^2, an ordinal score
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from eqcube.inference.calibration import nll

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def ece(log_probs: FloatArray, labels: IntArray, bins: int = 15) -> float:
    probs = np.exp(log_probs)
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.digitize(confidence, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        in_bin = which == b
        if in_bin.any():
            total += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(total)


def brier(log_probs: FloatArray, labels: IntArray) -> float:
    probs = np.exp(log_probs)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(labels)), labels] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def ordinal_brier(log_probs: FloatArray, labels: IntArray) -> float:
    s = 1.0 - np.cumsum(np.exp(log_probs), axis=1)[:, :-1]
    k = np.arange(s.shape[1])
    return float(((s - (labels[:, None] > k[None, :])) ** 2).mean())


def calibration_metrics(log_probs: FloatArray, labels: IntArray) -> dict[str, Any]:
    per_depth = {
        int(d): nll(log_probs[labels == d], labels[labels == d]) for d in np.unique(labels)
    }
    return {
        "nll": nll(log_probs, labels),
        "ece": ece(log_probs, labels),
        "brier": brier(log_probs, labels),
        "ordinal_brier": ordinal_brier(log_probs, labels),
        "per_depth": {"nll": per_depth},
    }
