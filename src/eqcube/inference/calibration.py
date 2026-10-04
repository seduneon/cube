"""Temperature scaling (Guo et al. 2017, "On Calibration of Modern Neural Networks").

    p_T(k) ∝ p(k)^(1/T)        T > 1 softens, T < 1 sharpens; argmax unchanged

T is fit on held-out data by minimizing the negative log-likelihood (a 1-D
problem: golden-section search over log T). ``TemperedPredictor`` wraps any
predictor (states -> log-probabilities) and is itself a predictor.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

import numpy as np
import numpy.typing as npt

from eqcube.puzzles.base import States

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
Predictor = Callable[[States], FloatArray]  # states -> log-probabilities (N, C)


def log_softmax(x: FloatArray) -> FloatArray:
    z = x - x.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def temper(log_probs: FloatArray, temperature: float) -> FloatArray:
    return log_softmax(log_probs / temperature)


def nll(log_probs: FloatArray, labels: IntArray) -> float:
    return float(-log_probs[np.arange(len(labels)), labels].mean())


def fit_temperature(
    log_probs: FloatArray, labels: IntArray, low: float = 0.05, high: float = 20.0
) -> float:
    """argmin_T NLL(temper(log_probs, T)) by golden-section search on log T."""
    log_probs = np.maximum(log_probs, -1e4)  # masked classes: finite, still ~0 probability

    def loss(log_t: float) -> float:
        return nll(temper(log_probs, math.exp(log_t)), labels)

    a, b = math.log(low), math.log(high)
    ratio = (math.sqrt(5) - 1) / 2
    c, d = b - ratio * (b - a), a + ratio * (b - a)
    fc, fd = loss(c), loss(d)
    for _ in range(60):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - ratio * (b - a)
            fc = loss(c)
        else:
            a, c, fc = c, d, fd
            d = a + ratio * (b - a)
            fd = loss(d)
    return math.exp((a + b) / 2)


class TemperedPredictor:
    def __init__(self, base: Predictor, temperature: float) -> None:
        self.base, self.temperature = base, temperature

    def __call__(self, states: States) -> FloatArray:
        return temper(np.maximum(self.base(states), -1e4), self.temperature)


Reduce = Literal["argmax", "expected"]


def costs_from_log_probs(log_probs: FloatArray, reduce: Reduce) -> FloatArray:
    """argmax: most likely distance. expected: sum_k k * p(k)."""
    if reduce == "argmax":
        return log_probs.argmax(axis=1).astype(np.float64)
    p = np.exp(log_softmax(log_probs))
    return p @ np.arange(log_probs.shape[1], dtype=np.float64)


class PredictorHeuristic:
    """Any predictor as a search heuristic (states -> estimated cost)."""

    reduce: Reduce

    def __init__(self, predictor: Predictor, reduce: Reduce = "expected") -> None:
        self.predictor, self.reduce = predictor, reduce

    def __call__(self, states: States) -> FloatArray:
        if len(states) == 0:
            return np.empty(0)
        return costs_from_log_probs(self.predictor(states), self.reduce)
