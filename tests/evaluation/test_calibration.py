"""Calibration metrics on cases with known answers."""

import numpy as np

from eqcube.evaluation.calibration import brier, calibration_metrics, ece, ordinal_brier


def onehot(labels: np.ndarray, c: int = 15, eps: float = 1e-9) -> np.ndarray:
    p = np.full((len(labels), c), eps)
    p[np.arange(len(labels)), labels] = 1.0
    return np.log(p / p.sum(1, keepdims=True))


def test_perfect_predictions() -> None:
    labels = np.array([0, 5, 9, 14])
    lp = onehot(labels)
    assert ece(lp, labels) < 1e-6 and brier(lp, labels) < 1e-6 and ordinal_brier(lp, labels) < 1e-6


def test_confidently_wrong_is_maximal_ece() -> None:
    labels = np.array([3, 3, 3])
    assert abs(ece(onehot(np.array([4, 4, 4])), labels) - 1.0) < 1e-6


def test_ordinal_brier_prefers_near_misses() -> None:
    labels = np.array([10])
    assert ordinal_brier(onehot(np.array([9])), labels) < ordinal_brier(
        onehot(np.array([2])), labels
    )


def test_report_has_per_depth_nll() -> None:
    labels = np.array([1, 1, 2])
    report = calibration_metrics(onehot(labels), labels)
    assert set(report["per_depth"]["nll"]) == {1, 2}
