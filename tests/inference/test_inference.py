"""Temperature scaling and the test-time Bellman backup, against exact oracles."""

import numpy as np
import pytest

from eqcube.data import DistanceTable
from eqcube.inference import (
    BackupPredictor,
    PredictorHeuristic,
    TemperedPredictor,
    costs_from_log_probs,
    fit_pooling,
    fit_temperature,
    temper,
)
from eqcube.inference.calibration import log_softmax
from eqcube.puzzles.pocket_coords import pocket_coords

C = pocket_coords()
PUZZLE = C.puzzle


@pytest.fixture(scope="module")
def table() -> DistanceTable:
    return DistanceTable.build(C)


def oracle(table: DistanceTable, sharp: float = 8.0):
    """A predictor that puts most mass on the true distance."""

    def predict(states: np.ndarray) -> np.ndarray:
        logits = np.zeros((len(states), 15))
        logits[np.arange(len(states)), table.distance(states)] = sharp
        return log_softmax(logits)

    return predict


def test_fit_temperature_recovers_known_scale() -> None:
    rng = np.random.default_rng(0)
    true_logits = rng.normal(size=(20_000, 15)) * 2
    labels = np.array([rng.choice(15, p=np.exp(r)) for r in log_softmax(true_logits)])
    overconfident = log_softmax(true_logits * 3.0)  # temperature 1/3 too sharp
    assert fit_temperature(overconfident, labels) == pytest.approx(3.0, rel=0.1)


def test_temper_keeps_argmax() -> None:
    lp = log_softmax(np.random.default_rng(1).normal(size=(100, 15)))
    for t in (0.3, 1.0, 4.0):
        np.testing.assert_array_equal(temper(lp, t).argmax(1), lp.argmax(1))
        np.testing.assert_allclose(np.exp(temper(lp, t)).sum(1), 1.0)


def test_tempered_predictor(table: DistanceTable) -> None:
    states = C.unrank(np.arange(0, C.n_states, 50_000))
    base = oracle(table)
    np.testing.assert_allclose(TemperedPredictor(base, 2.0)(states), temper(base(states), 2.0))


@pytest.mark.parametrize("depth", [1, 2])
def test_pure_backup_of_oracle_is_exact(table: DistanceTable, depth: int) -> None:
    index = np.random.default_rng(2).integers(0, C.n_states, 300)
    states = C.unrank(index)
    backed = BackupPredictor(oracle(table, 30.0), PUZZLE, w=0.0, beta=1.0, depth=depth)(states)
    np.testing.assert_array_equal(backed.argmax(1), table.labels(index))
    np.testing.assert_allclose(np.exp(backed).sum(1), 1.0, atol=1e-9)


def test_backup_with_only_own_weight_is_base(table: DistanceTable) -> None:
    states = C.unrank(np.random.default_rng(3).integers(0, C.n_states, 100))
    base = oracle(table)
    own = BackupPredictor(base, PUZZLE, w=1.0, beta=0.0)(states)
    np.testing.assert_allclose(np.exp(own), np.exp(base(states)), atol=1e-6)


def test_solved_state_is_exactly_zero(table: DistanceTable) -> None:
    lp = BackupPredictor(oracle(table), PUZZLE)(PUZZLE.solved_states(2))
    np.testing.assert_allclose(np.exp(lp[:, 0]), 1.0)


def test_backup_fixes_a_noisy_parent(table: DistanceTable) -> None:
    """A predictor that is wrong about parents but right about children: backup repairs it."""
    index = np.random.default_rng(4).integers(0, C.n_states, 400)
    parents = set(index.tolist())
    good = oracle(table, 6.0)

    def noisy(states: np.ndarray) -> np.ndarray:
        lp = good(states)
        is_parent = np.isin(C.rank(states), list(parents))
        lp[is_parent] = np.log(np.full(15, 1 / 15))  # know nothing about the parents
        return lp

    states = C.unrank(index)
    fit = fit_pooling(noisy, PUZZLE, states, table.labels(index))
    assert fit.beta > 0
    backed = BackupPredictor(noisy, PUZZLE, fit.w, fit.beta)(states)
    assert (backed.argmax(1) == table.labels(index)).mean() > 0.95


def test_fit_pooling_min_beta_restricts_the_grid(table: DistanceTable) -> None:
    """An exact base needs no backup; min_beta forces one anyway (default: unrestricted)."""
    index = np.random.default_rng(6).integers(0, C.n_states, 300)
    states, labels = C.unrank(index), table.labels(index)
    free = fit_pooling(oracle(table), PUZZLE, states, labels)
    forced = fit_pooling(oracle(table), PUZZLE, states, labels, min_beta=0.5)
    assert forced.beta >= 0.5
    assert all(b >= 0.5 for _, b in forced.grid)
    assert any(b == 0.0 for _, b in free.grid)
    assert forced.nll >= free.nll


def test_predictor_heuristic(table: DistanceTable) -> None:
    states = C.unrank(np.random.default_rng(5).integers(0, C.n_states, 50))
    h = PredictorHeuristic(oracle(table, 30.0), "argmax")
    np.testing.assert_array_equal(h(states), table.distance(states))
    lp = oracle(table)(states)
    np.testing.assert_allclose(
        PredictorHeuristic(oracle(table), "expected")(states), costs_from_log_probs(lp, "expected")
    )
