"""Exact puzzle facts applied to any predictor (states -> log-probabilities).

    LowerBoundPredictor   d(s) >= lower_bound(s), and d(s) has a known parity:
                          remove the impossible distances and renormalize
    InversionPredictor    d(s) = d(s^-1): pool the predictions for s and s^-1

Both are exact facts about the pocket cube (pattern databases; inversion),
checked on every state in tests/puzzles/test_pocket_coords.py. Both wrap a
predictor and are predictors, so they compose with temperature scaling and
the backup (``BackupPredictor`` calls its base on the children, so a wrapped
base applies the fact to the children too).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from eqcube.inference.calibration import Predictor, log_softmax
from eqcube.puzzles.base import States
from eqcube.puzzles.pocket_coords import PocketCubeCoords

FloatArray = npt.NDArray[np.float64]
FLOOR = -1e4  # log-probability of an impossible class (finite, as in TemperedPredictor)

Pool = Literal["log", "linear"]


@dataclass
class LowerBoundPredictor:
    """Zero the mass below the pattern-database bound and on the wrong parity.

    The bound itself always has the right parity, so at least one class stays.
    """

    base: Predictor
    coords: PocketCubeCoords

    def __call__(self, states: States) -> FloatArray:
        log_probs = np.maximum(self.base(states), FLOOR)
        index = self.coords.rank(states)
        bound = self.coords.lower_bound(index)
        k = np.arange(log_probs.shape[1])
        possible = (k[None, :] >= bound[:, None]) & (
            k[None, :] % 2 == self.coords.parity(index)[:, None]
        )
        return log_softmax(np.where(possible, log_probs, FLOOR))


@dataclass
class InversionPredictor:
    """Pool the predictions for s and s^-1; the result is exactly inversion-invariant.

    pool="log":    p ∝ sqrt(p_s * p_inv)   (geometric mean, a product of experts)
    pool="linear": p = (p_s + p_inv) / 2   (a mixture; never sharper than either)
    """

    base: Predictor
    coords: PocketCubeCoords
    pool: Pool = "log"

    def __call__(self, states: States) -> FloatArray:
        n = len(states)
        both = self.base(np.concatenate([states, self.coords.invert(states)]))
        own, inv = np.maximum(both[:n], FLOOR), np.maximum(both[n:], FLOOR)
        if self.pool == "log":
            return log_softmax((own + inv) / 2)
        return np.logaddexp(own, inv) - np.log(2.0)
