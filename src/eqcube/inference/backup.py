"""Test-time Bellman backup: refine a state's distribution with its children's.

Exact structure: for an unsolved state, d > k  <=>  every child has d > k - 1.
With S(k) = P(d > k) and independent children this gives the "backup"

    S_bk(0) = 1,    S_bk(k) = prod_c S_c(k - 1)                       (k >= 1)

Siblings' errors are correlated, so the raw product is overconfident. We pool
it with the model's own estimate (a logarithmic opinion pool; Genest & Zidek
1986, Hinton 2002 "products of experts"), with weights fit on validation data:

    log S(k) = w * log S_own(k) + beta * log S_bk(k)

(w, beta >= 0 keep S non-increasing, so it is a valid distribution). Solved
states get exactly d = 0. ``BackupPredictor`` wraps any predictor and is a
predictor itself, so evaluation and search use it unchanged; with
``depth = 2`` the children are themselves backed up from grandchildren.

rule="consistent" (docs/THEORY.md, sections 2-4) also uses that every child is
exactly d - 1 or d + 1, and conditions on it:

    L(k) = prod_c (q_c(k-1) + q_c(k+1)) - prod_c q_c(k+1)
    pi(k) ∝ p_s(k)^w * L(k)^beta

``sibling_posterior`` additionally returns, for every child, the posterior
probability that it is a "down" child and its expected distance given its
parent and siblings (the Bayes-optimal way to rank children).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from eqcube.inference.calibration import Predictor, nll
from eqcube.puzzles.base import PermutationPuzzle, States

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]
EPS = 1e-12


def log_survival(log_probs: FloatArray) -> FloatArray:
    """log S(k) = log P(d > k), k = 0..C-2."""
    s = 1.0 - np.cumsum(np.exp(log_probs), axis=-1)[..., :-1]
    return np.log(np.clip(s, EPS, 1.0))


def from_log_survival(log_s: FloatArray) -> FloatArray:
    """log S(0..C-2) -> log P(0..C-1), with S(-1) = 1 and P(C-1) = S(C-2)."""
    s = np.exp(log_s)
    s_prev = np.concatenate([np.ones_like(s[..., :1]), s[..., :-1]], axis=-1)
    probs = np.concatenate([s_prev - s, s[..., -1:]], axis=-1)
    probs = np.clip(probs, 0.0, None)
    probs /= probs.sum(axis=-1, keepdims=True)
    return np.log(np.clip(probs, EPS, None))


def backup_log_survival(child_log_probs: FloatArray) -> FloatArray:
    """(N, M, C) children log-probs -> (N, C-1) parent log S via the Bellman product."""
    child_log_s = log_survival(child_log_probs)  # (N, M, C-1)
    parent = child_log_s[..., :-1].sum(axis=1)  # k = 1..C-2 from children k-1 = 0..C-3
    return np.concatenate([np.zeros_like(parent[:, :1]), parent], axis=-1)


def pooled_log_probs(
    own_log_probs: FloatArray,
    child_log_probs: FloatArray,
    solved: BoolArray,
    w: float,
    beta: float,
) -> FloatArray:
    log_s = w * log_survival(own_log_probs) + beta * backup_log_survival(child_log_probs)
    out = from_log_survival(log_s)
    if solved.any():
        out[solved] = np.log(EPS)
        out[solved, 0] = 0.0
    return out


BackupRule = Literal["min", "consistent"]


def _neighbour_probs(child_log_probs: FloatArray) -> tuple[FloatArray, FloatArray]:
    """For every parent value k: q_c(k-1) ("down") and q_c(k+1) ("up"), shape (N, M, C)."""
    q = np.exp(child_log_probs)
    padded = np.pad(q, [(0, 0)] * (q.ndim - 1) + [(1, 1)])
    return padded[..., :-2], padded[..., 2:]


def consistent_log_likelihood(child_log_probs: FloatArray) -> FloatArray:
    """log L(k), k = 0..C-1, for an unsolved parent (L(0) = 0: a child cannot be at -1)."""
    down, up = _neighbour_probs(child_log_probs)
    sum_log_a = np.log(np.clip(down + up, EPS, None)).sum(axis=-2)  # (N, C)
    sum_log_u = np.log(np.clip(up, EPS, None)).sum(axis=-2)
    ratio = np.exp(np.minimum(sum_log_u - sum_log_a, 0.0))  # P(all up | all neighbours)
    return sum_log_a + np.log(np.clip(1.0 - ratio, EPS, None))


def _normalize(log_x: FloatArray) -> FloatArray:
    z = log_x - log_x.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def consistent_log_probs(
    own_log_probs: FloatArray,
    child_log_probs: FloatArray,
    solved: BoolArray,
    w: float,
    beta: float,
) -> FloatArray:
    """Parent posterior pi(k) ∝ p_s(k)^w L(k)^beta; solved states are exactly 0."""
    log_post = w * np.maximum(own_log_probs, np.log(EPS)) + beta * consistent_log_likelihood(
        child_log_probs
    )
    out = _normalize(log_post)
    if solved.any():
        out[solved] = np.log(EPS)
        out[solved, 0] = 0.0
    return out


@dataclass(frozen=True)
class SiblingPosterior:
    parent_log_probs: FloatArray  # (N, C)
    p_down: FloatArray  # (N, M) P(child is at d(parent) - 1)
    child_expected: FloatArray  # (N, M) E[d(child) | parent, siblings]


def sibling_posterior(
    own_log_probs: FloatArray,
    child_log_probs: FloatArray,
    solved: BoolArray,
    w: float = 1.0,
    beta: float = 1.0,
) -> SiblingPosterior:
    """Joint posterior over the parent's distance and each child's direction."""
    parent = consistent_log_probs(own_log_probs, child_log_probs, solved, w, beta)
    down, up = _neighbour_probs(child_log_probs)
    a = down / np.clip(down + up, EPS, None)  # P(c down | k, c in {k-1, k+1})
    none_down = np.exp(np.log(np.clip(1.0 - a, EPS, None)).sum(axis=-2, keepdims=True))
    p_down_given_k = a / np.clip(1.0 - none_down, EPS, None)  # condition on >= 1 down
    p_down_given_k = np.clip(p_down_given_k, 0.0, 1.0)
    pi = np.exp(parent)[:, None, :]  # (N, 1, C)
    k = np.arange(child_log_probs.shape[-1], dtype=np.float64)
    p_down = (pi * p_down_given_k).sum(axis=-1)
    child_expected = (pi * (k + 1.0 - 2.0 * p_down_given_k)).sum(axis=-1)
    if solved.any():  # every child of the solved state is at distance 1
        p_down[solved] = 0.0
        child_expected[solved] = 1.0
    return SiblingPosterior(parent, p_down, child_expected)


def backup_log_probs(child_log_probs: FloatArray, solved: BoolArray) -> FloatArray:
    """Pure Bellman backup (w = 0, beta = 1); the NumPy twin of training.bellman.backup_probs."""
    c = child_log_probs.shape[-1]
    own = np.zeros((len(solved), c))
    return pooled_log_probs(own, child_log_probs, solved, w=0.0, beta=1.0)


def _combine(
    rule: BackupRule,
    own: FloatArray,
    child_lp: FloatArray,
    solved: BoolArray,
    w: float,
    beta: float,
) -> FloatArray:
    if rule == "min":
        return pooled_log_probs(own, child_lp, solved, w, beta)
    return consistent_log_probs(own, child_lp, solved, w, beta)


@dataclass
class BackupPredictor:
    base: Predictor
    puzzle: PermutationPuzzle
    w: float = 0.5
    beta: float = 0.5
    depth: int = 1
    rule: BackupRule = "min"

    def __call__(self, states: States) -> FloatArray:
        if self.depth < 1:
            return self.base(states)
        own = self.base(states)
        children = self.puzzle.expand(states)  # (N, M, P)
        n, m, p = children.shape
        child_predictor = (
            self.base
            if self.depth == 1
            else BackupPredictor(
                self.base, self.puzzle, self.w, self.beta, self.depth - 1, self.rule
            )
        )
        child_lp = child_predictor(children.reshape(n * m, p)).reshape(n, m, -1)
        solved = self.puzzle.is_solved(states)
        return _combine(self.rule, own, child_lp, solved, self.w, self.beta)


@dataclass(frozen=True)
class PoolingFit:
    w: float
    beta: float
    nll: float
    grid: dict[tuple[float, float], float]


def fit_pooling(
    base: Predictor,
    puzzle: PermutationPuzzle,
    states: States,
    labels: IntArray,
    grid: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    rule: BackupRule = "min",
) -> PoolingFit:
    """Choose (w, beta) minimizing NLL on held-out states (one-step backup).

    The model is evaluated once on the states and their children; every grid
    point is then pure NumPy.
    """
    own = base(states)
    children = puzzle.expand(states)
    n, m, p = children.shape
    child_lp = base(children.reshape(n * m, p)).reshape(n, m, -1)
    solved = puzzle.is_solved(states)
    scores = {
        (w, b): nll(_combine(rule, own, child_lp, solved, w, b), labels)
        for w in grid
        for b in grid
        if w > 0 or b > 0
    }
    (w, b), best = min(scores.items(), key=lambda kv: kv[1])
    return PoolingFit(w, b, best, scores)
