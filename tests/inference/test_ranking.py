"""Ranking rules: hand-checked values, a Monte Carlo oracle, and the exact-table oracle."""

import numpy as np
import pytest

from eqcube.data import DistanceTable
from eqcube.inference.calibration import log_softmax
from eqcube.inference.ranking import RULES, child_costs, p_closest, quantile
from eqcube.puzzles.pocket_coords import pocket_coords

C = pocket_coords()


def test_quantile_by_hand() -> None:
    probs = np.array([[0.1, 0.2, 0.3, 0.4]])
    assert quantile(probs, 0.25).tolist() == [1.0]
    assert quantile(probs, 0.3).tolist() == [1.0]  # P(d <= 1) = 0.3 exactly
    assert quantile(probs, 0.5).tolist() == [2.0]
    assert quantile(probs, 0.75).tolist() == [3.0]


def test_p_closest_matches_monte_carlo() -> None:
    rng = np.random.default_rng(0)
    probs = np.exp(log_softmax(rng.normal(size=(1, 4, 6)) * 1.5))
    exact = p_closest(probs)[0]
    draws = np.stack(
        [rng.choice(6, size=200_000, p=probs[0, c] / probs[0, c].sum()) for c in range(4)]
    )
    mc = (draws <= draws.min(axis=0)).mean(axis=1)  # ties: every minimum counts
    np.testing.assert_allclose(exact, mc, atol=5e-3)


def test_p_closest_two_point_masses() -> None:
    probs = np.zeros((1, 2, 4))
    probs[0, 0, 1] = probs[0, 1, 3] = 1.0
    np.testing.assert_allclose(p_closest(probs)[0], [1.0, 0.0], atol=1e-12)


@pytest.mark.parametrize("rule", [r for r in RULES if r not in ("argmax", "p_down")])
def test_rules_ignore_a_shift_shared_by_all_siblings(rule: str) -> None:
    """Moving every sibling's distribution by the same amount keeps the order."""
    rng = np.random.default_rng(1)
    lp = log_softmax(
        np.pad(rng.normal(size=(50, 6, 10)) * 2, [(0, 0), (0, 0), (0, 3)], constant_values=-50)
    )
    shifted = np.roll(lp, 2, axis=-1)
    a = child_costs(rule, lp).argsort(axis=1, kind="stable")  # type: ignore[arg-type]
    b = child_costs(rule, shifted).argsort(axis=1, kind="stable")  # type: ignore[arg-type]
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("rule", RULES)
def test_every_rule_picks_a_down_child_with_an_oracle(rule: str) -> None:
    table = DistanceTable.build(C)
    index = np.random.default_rng(2).choice(np.flatnonzero(table.dist >= 1), 500, replace=False)
    children = C.expand(index)

    def oracle(idx: np.ndarray) -> np.ndarray:
        logits = np.zeros((*idx.shape, 15))
        np.put_along_axis(logits, table.dist[idx][..., None].astype(np.int64), 8.0, axis=-1)
        return log_softmax(logits)

    costs = child_costs(rule, oracle(children), oracle(index))  # type: ignore[arg-type]
    best = costs.argmin(axis=1)
    assert (table.dist[children[np.arange(len(index)), best]] < table.dist[index]).all()


def test_p_down_requires_the_parent() -> None:
    with pytest.raises(ValueError):
        child_costs("p_down", np.zeros((1, 6, 15)))
