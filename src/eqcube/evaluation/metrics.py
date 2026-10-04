"""Evaluation metrics. NumPy only: models enter as callables.

    Predictor  states -> logits (N, num_classes)
    Heuristic  states -> costs  (N,)            (see eqcube.search)

Every metric is reported overall and per true distance, so any aggregate
can be re-weighted afterwards.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from eqcube.data.symmetry import apply_symmetry
from eqcube.data.table import DistanceTable
from eqcube.puzzles.base import States
from eqcube.search.base import Heuristic, Searcher
from eqcube.symmetry.perm import PermGroup

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]
Predictor = Callable[[States], FloatArray]
Report = dict[str, Any]


def _log_softmax(logits: FloatArray) -> FloatArray:
    z = logits - logits.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def _per_depth(values: FloatArray, labels: IntArray) -> dict[int, float]:
    return {int(d): float(values[labels == d].mean()) for d in np.unique(labels)}


def value_metrics(logits: FloatArray, labels: IntArray) -> Report:
    """Cross-entropy, accuracy, MAE (argmax and expected), overall and per depth."""
    logp = _log_softmax(logits)
    ce = -logp[np.arange(len(labels)), labels]
    pred = logits.argmax(axis=1)
    expected = np.exp(logp) @ np.arange(logits.shape[1])
    correct = (pred == labels).astype(np.float64)
    abs_err = np.abs(pred - labels).astype(np.float64)
    abs_err_exp = np.abs(expected - labels)
    return {
        "ce": float(ce.mean()),
        "accuracy": float(correct.mean()),
        "mae": float(abs_err.mean()),
        "mae_expected": float(abs_err_exp.mean()),
        "per_depth": {
            "accuracy": _per_depth(correct, labels),
            "mae": _per_depth(abs_err, labels),
            "ce": _per_depth(ce, labels),
        },
    }


def sibling_ranking(heuristic: Heuristic, table: DistanceTable, index: IntArray) -> Report:
    """How well the heuristic orders each state's children (what search relies on).

    Every child is at d-1 ("down") or d+1 ("up"). For parents at d >= 1:
      pair_accuracy:  P(cost(down) < cost(up)) over all (down, up) pairs,
                      ties counting 1/2 (an AUC); parents without up
                      children (maximum depth) are skipped
      best_child:     the lowest-cost child is a down child (greedy picks well),
                      ties split uniformly
    """
    index = index[table.dist[index] >= 1]
    children = table.indexer.expand(index)  # (N, M)
    n, m = children.shape
    costs = heuristic(table.indexer.unrank(children.reshape(-1))).reshape(n, m)
    down = table.dist[children] < table.dist[index, None]

    diff = costs[:, :, None] - costs[:, None, :]  # [i, a, b] = cost_a - cost_b
    pair = down[:, :, None] & ~down[:, None, :]
    score = (diff < 0) + 0.5 * (diff == 0)
    n_pairs = pair.sum(axis=(1, 2))
    has_pairs = n_pairs > 0  # at the maximum depth every child is a down child
    pair_acc = (score * pair).sum(axis=(1, 2))[has_pairs] / n_pairs[has_pairs]

    is_min = costs == costs.min(axis=1, keepdims=True)
    best = (is_min & down).sum(axis=1) / is_min.sum(axis=1)

    labels = table.labels(index)
    return {
        "pair_accuracy": float(pair_acc.mean()),
        "best_child": float(best.mean()),
        "per_depth": {
            "pair_accuracy": _per_depth(pair_acc, labels[has_pairs]),
            "best_child": _per_depth(best, labels),
        },
    }


def equivariance_error(predict: Predictor, states: States, group: PermGroup) -> Report:
    """max |logits(g·s) - logits(s)| over g in group: ~0 for an exactly invariant model."""
    n_positions = states.shape[1]
    base = predict(states)
    errors = [
        np.abs(predict(apply_symmetry(states, g, n_positions)) - base).max(axis=1)
        for g in group.elements
        if not g.is_identity()
    ]
    if not errors:
        return {"max": 0.0, "mean": 0.0}
    stacked = np.stack(errors)
    return {"max": float(stacked.max()), "mean": float(stacked.mean())}


def solve_metrics(searcher: Searcher, table: DistanceTable, index: IntArray) -> Report:
    """Solve rate, solution length, and excess over optimal, overall and per start depth."""
    puzzle = table.indexer.puzzle
    starts = table.indexer.unrank(index)
    results = searcher.solve(puzzle, starts)
    solved = np.array([r.solved for r in results], dtype=np.float64)
    optimal = table.labels(index)
    length = np.array([r.length if r.solved else np.nan for r in results], dtype=np.float64)
    excess = length - optimal.astype(np.float64)

    def masked_mean(x: FloatArray, mask: npt.NDArray[np.bool_]) -> float:
        return float(np.nanmean(x[mask])) if np.any(mask & ~np.isnan(x)) else float("nan")

    depths = np.unique(optimal)
    return {
        "solve_rate": float(solved.mean()),
        "mean_length": masked_mean(length, np.ones(len(index), dtype=bool)),
        "mean_excess": masked_mean(excess, np.ones(len(index), dtype=bool)),
        "per_depth": {
            "solve_rate": {int(d): float(solved[optimal == d].mean()) for d in depths},
            "mean_excess": {int(d): masked_mean(excess, optimal == d) for d in depths},
        },
    }
