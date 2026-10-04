"""G-set constructions and the vectorised orbit labelling."""

import numpy as np
import pytest

from eqcube.symmetry.action import TupleAction, orbits
from eqcube.symmetry.gset import GSet, pair_orbit_labels
from eqcube.symmetry.perm import Perm, PermGroup, direct_product, symmetric_group

S4 = symmetric_group(4)


def test_natural_and_trivial() -> None:
    assert GSet.natural(S4).size == 4
    assert GSet.trivial(S4, 5).is_trivial()
    assert GSet.trivial(S4, 5).orbit_labels[1] == 5


def test_block_rejects_non_invariant_block() -> None:
    with pytest.raises(ValueError):
        GSet.block(S4, 0, 2)


def test_block_of_direct_product() -> None:
    g = direct_product(S4, symmetric_group(3))
    assert GSet.block(g, 4, 3).orbit_labels[1] == 1  # S3 transitive on its block


def test_product_requires_same_group() -> None:
    with pytest.raises(ValueError):
        GSet.natural(S4).product(GSet.natural(symmetric_group(4)))


def test_image_of_is_a_homomorphism() -> None:
    a = GSet.natural(S4)
    pairs = a.product(a)
    for g in S4.elements:
        for h in S4.elements:
            assert pairs.image_of[g * h] == pairs.image_of[g] * pairs.image_of[h]


def test_image_of_rejects_non_homomorphism() -> None:
    # Sending the 4-cycle generator to a transposition is not a homomorphism.
    bad = GSet(S4, (Perm.identity(2), Perm.from_cycles([(0, 1)], 2)))
    with pytest.raises(ValueError):
        _ = bad.image_of


def test_vectorised_pair_orbits_match_generic_orbits() -> None:
    g = PermGroup([Perm.from_cycles([(0, 1, 2), (3, 4)], 5)])
    a = GSet.natural(g)
    labels, n = pair_orbit_labels(a, a)
    expected = orbits(TupleAction(g), [(i, j) for i in range(5) for j in range(5)])
    assert n == len(expected)
    for o in expected:
        assert len({int(labels[i, j]) for i, j in o}) == 1  # one label per orbit


def test_gather_index_is_the_function_action() -> None:
    a = GSet.natural(S4)
    x = np.arange(4)
    for g in S4.elements:
        y = x[a.gather_index(g)]
        # (g·x)[g(p)] == x[p]
        assert all(y[g(p)] == x[p] for p in range(4))
