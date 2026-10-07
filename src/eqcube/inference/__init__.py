"""Predictor transformations (NumPy only): calibration and Bellman backup.

A predictor is any callable states -> log-probabilities (N, C). Everything
here wraps a predictor and returns a predictor (the Decorator pattern), so
evaluation and search accept the result unchanged:

    TemperedPredictor(base, T)                temperature scaling
    BackupPredictor(base, puzzle, w, beta)    Bellman backup pooled with base
    LowerBoundPredictor(base, coords)         exact bound + parity (givens.py)
    InversionPredictor(base, coords)          pool s and s^-1 (givens.py)
"""

from eqcube.inference.backup import (
    BackupPredictor,
    BackupRule,
    PoolingFit,
    SiblingPosterior,
    backup_log_probs,
    consistent_log_likelihood,
    consistent_log_probs,
    fit_pooling,
    log_survival,
    pooled_log_probs,
    sibling_posterior,
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
from eqcube.inference.givens import InversionPredictor, LowerBoundPredictor

__all__ = [
    "BackupPredictor",
    "BackupRule",
    "InversionPredictor",
    "LowerBoundPredictor",
    "PoolingFit",
    "Predictor",
    "PredictorHeuristic",
    "Reduce",
    "SiblingPosterior",
    "TemperedPredictor",
    "backup_log_probs",
    "consistent_log_likelihood",
    "consistent_log_probs",
    "costs_from_log_probs",
    "fit_pooling",
    "fit_temperature",
    "log_survival",
    "nll",
    "pooled_log_probs",
    "sibling_posterior",
    "temper",
]
