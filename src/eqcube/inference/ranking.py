"""Rules that turn the children's distance distributions into one cost per child.

Search keeps the children with the lowest cost, so the rule is a decision
rule (docs/STATUS.md, open question 1). Every rule takes the children's
log-probabilities (N, M, C) and returns costs (N, M), lower = more promising:

    expected     E[d(c)]                              (used so far)
    argmax       most likely d(c)                     (v1; many ties)
    median, q25, q75
                 quantiles of d(c), ties broken by E[d(c)] (1e-3 weight)
    p_closest    -P(d(c) <= d(c') for every sibling c'), children independent
    p_down       -P(c is a "down" child | parent, siblings)   (THEORY.md §3;
                 needs the parent's own log-probabilities)

The metric (``evaluation.metrics.rank_children``) splits ties uniformly, so
rules with many ties are not given an unfair tie-break.
"""

from __future__ import annotations

from typing import Literal, get_args

import numpy as np
import numpy.typing as npt

from eqcube.inference.backup import sibling_posterior

FloatArray = npt.NDArray[np.float64]
Rule = Literal["expected", "argmax", "median", "q25", "q75", "p_closest", "p_down"]
RULES: tuple[Rule, ...] = get_args(Rule)
TIE_BREAK = 1e-3


def expected(probs: FloatArray) -> FloatArray:
    return probs @ np.arange(probs.shape[-1], dtype=np.float64)


def quantile(probs: FloatArray, q: float) -> FloatArray:
    """Smallest k with P(d <= k) >= q."""
    cdf = np.cumsum(probs, axis=-1)
    return (cdf < q - 1e-12).sum(axis=-1).astype(np.float64)


def p_closest(probs: FloatArray) -> FloatArray:
    """P(X_c <= X_j for all siblings j != c), siblings independent.

    sum_k p_c(k) * prod_{j != c} P(X_j >= k); the product over j != c is the
    product over all j divided by c's own factor (in log space).
    """
    at_least = np.clip(np.flip(np.cumsum(np.flip(probs, -1), -1), -1), 1e-300, 1.0)  # P(X >= k)
    log_all = np.log(at_least).sum(axis=-2, keepdims=True)  # (N, 1, C)
    others = np.exp(log_all - np.log(at_least))  # (N, M, C)
    return (probs * others).sum(axis=-1)


def child_costs(
    rule: Rule, child_log_probs: FloatArray, own_log_probs: FloatArray | None = None
) -> FloatArray:
    probs = np.exp(child_log_probs)
    probs /= probs.sum(axis=-1, keepdims=True)
    mean = expected(probs)
    if rule == "expected":
        return mean
    if rule == "argmax":
        return child_log_probs.argmax(axis=-1).astype(np.float64)
    if rule in ("median", "q25", "q75"):
        q = {"median": 0.5, "q25": 0.25, "q75": 0.75}[rule]
        return quantile(probs, q) + TIE_BREAK * mean
    if rule == "p_closest":
        return -p_closest(probs)
    if own_log_probs is None:
        raise ValueError("p_down needs the parent's own log-probabilities")
    solved = np.zeros(len(own_log_probs), dtype=bool)
    return -sibling_posterior(own_log_probs, child_log_probs, solved).p_down
