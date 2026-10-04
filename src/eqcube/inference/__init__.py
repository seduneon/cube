"""Predictor transformations (NumPy only): calibration and Bellman backup.

A predictor is any callable states -> log-probabilities (N, C). Everything
here wraps a predictor and returns a predictor (the Decorator pattern), so
evaluation and search accept the result unchanged:

    TemperedPredictor(base, T)                temperature scaling
    BackupPredictor(base, puzzle, w, beta)    Bellman backup pooled with base
"""

from eqcube.inference.backup import (
    BackupPredictor,
    PoolingFit,
    backup_log_probs,
    fit_pooling,
    log_survival,
    pooled_log_probs,
)
from eqcube.inference.calibration import (
    Predictor,
    PredictorHeuristic,
    Reduce,
    TemperedPredictor,
    costs_from_log_probs,
    fit_temperature,
    nll,
    temper,
)

__all__ = [
    "BackupPredictor",
    "PoolingFit",
    "Predictor",
    "PredictorHeuristic",
    "Reduce",
    "TemperedPredictor",
    "backup_log_probs",
    "costs_from_log_probs",
    "fit_pooling",
    "fit_temperature",
    "log_survival",
    "nll",
    "pooled_log_probs",
    "temper",
]
