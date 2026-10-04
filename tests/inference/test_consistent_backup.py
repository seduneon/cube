"""The two-sided (consistent) backup and sibling posterior, checked by brute force."""

import itertools

import numpy as np
import pytest

from eqcube.data import DistanceTable
from eqcube.evaluation import posterior_sibling_ranking, sibling_ranking
from eqcube.inference import (
    BackupPredictor,
    PredictorHeuristic,
    consistent_log_probs,
    fit_pooling,
    pooled_log_probs,
    sibling_posterior,
)
from eqcube.inference.calibration import log_softmax
from eqcube.puzzles.pocket_coords import pocket_coords

C = pocket_coords()


def brute_force(q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Enumerate all children's values; keep configurations consistent with some parent."""
    m, c = q.shape
    parent = np.zeros(c)
    p_down = np.zeros(m)
    expected = np.zeros(m)
    total = 0.0
    for values in itertools.product(range(c), repeat=m):
        k = min(values) + 1
        if k > c - 1 or any(v not in (k - 1, k + 1) for v in values):
            continue
        weight = float(np.prod([q[j, v] for j, v in enumerate(values)]))
        total += weight
        parent[k] += weight
        for j, v in enumerate(values):
            p_down[j] += weight * (v == k - 1)
            expected[j] += weight * v
    return parent / total, p_down / total, expected / total


@pytest.mark.parametrize("seed", range(5))
def test_closed_form_matches_enumeration(seed: int) -> None:
    rng = np.random.default_rng(seed)
    m, c = 3, 6
    q = rng.dirichlet(np.ones(c) * 0.7, size=m)
    child_lp = np.log(q)[None]  # (1, M, C)
    post = sibling_posterior(np.zeros((1, c)), child_lp, np.array([False]), w=0.0, beta=1.0)
    parent, p_down, expected = brute_force(q)
    np.testing.assert_allclose(np.exp(post.parent_log_probs[0]), parent, atol=1e-9)
    np.testing.assert_allclose(post.p_down[0], p_down, atol=1e-9)
    np.testing.assert_allclose(post.child_expected[0], expected, atol=1e-9)


def test_consistent_rule_removes_impossible_configurations() -> None:
    """Child 1 is surely 3; child 2 is 2 or 5. Only (3, 5) fits a parent (at 4).

    The min rule still gives the impossible (3, 2) half its weight (parent 3).
    """
    c = 8
    q = np.full((1, 2, c), 1e-12)
    q[0, 0, 3] = 1.0
    q[0, 1, [2, 5]] = 0.5
    child_lp = np.log(q / q.sum(-1, keepdims=True))
    solved = np.array([False])
    own = np.zeros((1, c))
    consistent = np.exp(consistent_log_probs(own, child_lp, solved, 0.0, 1.0))[0]
    minimum = np.exp(pooled_log_probs(own, child_lp, solved, 0.0, 1.0))[0]
    assert consistent[4] > 0.999
    assert minimum[3] == pytest.approx(0.5, abs=1e-6)


@pytest.fixture(scope="module")
def table() -> DistanceTable:
    return DistanceTable.build(C)


def oracle(table: DistanceTable, sharp: float = 30.0):
    def predict(states: np.ndarray) -> np.ndarray:
        logits = np.zeros((len(states), 15))
        logits[np.arange(len(states)), table.distance(states)] = sharp
        return log_softmax(logits)

    return predict


def test_exact_children_give_exact_parent_and_directions(table: DistanceTable) -> None:
    index = np.random.default_rng(1).integers(0, C.n_states, 500)
    index = index[table.dist[index] >= 1]
    children = C.expand(index)
    predict = oracle(table)
    own = predict(C.unrank(index))
    child_lp = predict(C.unrank(children.reshape(-1))).reshape(len(index), 6, -1)
    post = sibling_posterior(own, child_lp, np.zeros(len(index), dtype=bool))
    np.testing.assert_array_equal(post.parent_log_probs.argmax(1), table.labels(index))
    down = table.dist[children] < table.dist[index, None]
    np.testing.assert_allclose(post.p_down, down.astype(float), atol=1e-6)
    np.testing.assert_allclose(post.child_expected, table.dist[children], atol=1e-5)


def test_consistent_backup_predictor_and_fit(table: DistanceTable) -> None:
    index = np.random.default_rng(2).integers(0, C.n_states, 300)
    states = C.unrank(index)
    exact = BackupPredictor(oracle(table), C.puzzle, 0.0, 1.0, rule="consistent")(states)
    np.testing.assert_array_equal(exact.argmax(1), table.labels(index))
    fit = fit_pooling(oracle(table, 4.0), C.puzzle, states, table.labels(index), rule="consistent")
    assert fit.nll < 1.0


def test_posterior_ranking_with_oracle_is_perfect(table: DistanceTable) -> None:
    index = np.random.default_rng(3).integers(0, C.n_states, 300)
    report = posterior_sibling_ranking(oracle(table, 6.0), table, index)
    assert report["best_child"] == 1.0 and report["pair_accuracy"] == 1.0
    same = sibling_ranking(PredictorHeuristic(oracle(table, 6.0)), table, index)
    assert same["best_child"] == 1.0
