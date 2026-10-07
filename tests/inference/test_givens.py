"""Exact-fact wrappers: lower bound + parity, and inversion pooling."""

import numpy as np
import pytest

from eqcube.data import DistanceTable
from eqcube.inference import (
    BackupPredictor,
    InversionPredictor,
    LowerBoundPredictor,
    TemperedPredictor,
)
from eqcube.inference.calibration import log_softmax
from eqcube.puzzles.pocket_coords import pocket_coords

C = pocket_coords()
PUZZLE = C.puzzle
STATES = PUZZLE.scramble(2000, 25, np.random.default_rng(0))


@pytest.fixture(scope="module")
def table() -> DistanceTable:
    return DistanceTable.build(C)


def random_predictor(seed: int = 0, scale: float = 3.0):
    """A fixed, deterministic function of the state (so s and s^-1 differ)."""
    weights = np.random.default_rng(seed).normal(size=(24 * 6, 15)) * scale / 12

    def predict(states: np.ndarray) -> np.ndarray:
        onehot = np.eye(6)[states.astype(np.int64)].reshape(len(states), -1)
        return log_softmax(onehot @ weights)

    return predict


def oracle(table: DistanceTable, sharp: float = 8.0):
    def predict(states: np.ndarray) -> np.ndarray:
        logits = np.zeros((len(states), 15))
        logits[np.arange(len(states)), table.distance(states)] = sharp
        return log_softmax(logits)

    return predict


def test_lower_bound_removes_impossible_mass() -> None:
    out = LowerBoundPredictor(random_predictor(), C)(STATES)
    np.testing.assert_allclose(np.exp(out).sum(axis=1), 1.0)
    index = C.rank(STATES)
    k = np.arange(15)
    impossible = (k[None] < C.lower_bound(index)[:, None]) | (
        k[None] % 2 != C.parity(index)[:, None]
    )
    assert np.exp(out[impossible]).max() < 1e-300


def test_lower_bound_never_lowers_the_true_class(table: DistanceTable) -> None:
    """The true distance is always possible, so renormalizing only adds to it."""
    base = random_predictor()
    labels = table.distance(STATES)
    rows = np.arange(len(STATES))
    before = base(STATES)[rows, labels]
    after = LowerBoundPredictor(base, C)(STATES)[rows, labels]
    assert (after >= before - 1e-12).all()
    assert (after > before + 1e-6).mean() > 0.9  # and almost always strictly


def test_lower_bound_keeps_an_oracle(table: DistanceTable) -> None:
    out = LowerBoundPredictor(oracle(table), C)(STATES)
    np.testing.assert_array_equal(out.argmax(1), table.distance(STATES))


@pytest.mark.parametrize("pool", ["log", "linear"])
def test_inversion_is_exactly_invariant(pool: str) -> None:
    wrapped = InversionPredictor(random_predictor(), C, pool)  # type: ignore[arg-type]
    a, b = wrapped(STATES), wrapped(C.invert(STATES))
    np.testing.assert_allclose(a, b, atol=1e-12)
    np.testing.assert_allclose(np.exp(a).sum(axis=1), 1.0)


@pytest.mark.parametrize("pool", ["log", "linear"])
def test_inversion_keeps_an_oracle(table: DistanceTable, pool: str) -> None:
    out = InversionPredictor(oracle(table), C, pool)(STATES)  # type: ignore[arg-type]
    np.testing.assert_array_equal(out.argmax(1), table.distance(STATES))


def test_inversion_of_an_invariant_predictor_is_identity() -> None:
    """If the base already satisfies p(s) = p(s^-1), pooling changes nothing."""
    base = random_predictor()

    def symmetric(states: np.ndarray) -> np.ndarray:
        return InversionPredictor(base, C, "log")(states)

    np.testing.assert_allclose(InversionPredictor(symmetric, C, "log")(STATES), symmetric(STATES))
    np.testing.assert_allclose(
        InversionPredictor(symmetric, C, "linear")(STATES), symmetric(STATES), atol=1e-12
    )


def test_wrappers_compose_with_temperature_and_backup(table: DistanceTable) -> None:
    chain = BackupPredictor(
        InversionPredictor(LowerBoundPredictor(TemperedPredictor(oracle(table), 2.0), C), C),
        PUZZLE,
        w=0.5,
        beta=0.5,
    )
    out = chain(STATES[:200])
    np.testing.assert_allclose(np.exp(out).sum(axis=1), 1.0)
    np.testing.assert_array_equal(out.argmax(1), table.distance(STATES[:200]))
