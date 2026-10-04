"""Finite permutation groups, actions and G-sets.

Pure Python/NumPy. This package must never import torch.
"""

from eqcube.symmetry.action import (
    Action,
    ConjugationAction,
    FunctionAction,
    PointAction,
    SetAction,
    TupleAction,
    orbit,
    orbits,
    stabilizer,
)
from eqcube.symmetry.gset import GSet, pair_orbit_labels
from eqcube.symmetry.perm import Perm, PermGroup, direct_product, symmetric_group

__all__ = [
    "Action",
    "ConjugationAction",
    "FunctionAction",
    "GSet",
    "Perm",
    "PermGroup",
    "PointAction",
    "SetAction",
    "TupleAction",
    "direct_product",
    "orbit",
    "orbits",
    "pair_orbit_labels",
    "stabilizer",
    "symmetric_group",
]
