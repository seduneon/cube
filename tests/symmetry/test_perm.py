"""Law tests for Perm and PermGroup."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from eqcube.symmetry.perm import Perm, PermGroup, direct_product, symmetric_group

N = 6


def perms(n: int = N) -> st.SearchStrategy[Perm]:
    """Hypothesis strategy: random permutations of 0..n-1."""
    return st.permutations(range(n)).map(lambda p: Perm(tuple(p)))


# ── Perm: examples ────────────────────────────────────────────────────────


def test_from_cycles() -> None:
    g = Perm.from_cycles([(0, 1, 2)], n=4)
    assert g.arr == (1, 2, 0, 3)


def test_rejects_non_permutation() -> None:
    with pytest.raises(ValueError):
        Perm((0, 0, 1))


def test_is_hashable_value() -> None:
    a = Perm((1, 0, 2))
    b = Perm((1, 0, 2))
    assert a == b
    assert len({a, b}) == 1


def test_degree_mismatch() -> None:
    with pytest.raises(ValueError):
        _ = Perm.identity(3) * Perm.identity(4)


# ── Perm: laws (hypothesis runs each on ~100 random inputs) ───────────────


@given(perms(), perms())
def test_composition_order(g: Perm, h: Perm) -> None:
    for i in range(N):
        assert (g * h)(i) == g(h(i))


@given(perms(), perms(), perms())
def test_associativity(a: Perm, b: Perm, c: Perm) -> None:
    assert (a * b) * c == a * (b * c)


@given(perms())
def test_identity(g: Perm) -> None:
    e = Perm.identity(N)
    assert e * g == g == g * e


@given(perms())
def test_inverse(g: Perm) -> None:
    assert (g * g.inverse()).is_identity()
    assert (g.inverse() * g).is_identity()


# ── Perm.order ────────────────────────────────────────────────────────────


def test_order_examples() -> None:
    assert Perm.identity(5).order() == 1
    assert Perm.from_cycles([(0, 1, 2, 3)], n=5).order() == 4
    assert Perm.from_cycles([(0, 1), (2, 3, 4)], n=5).order() == 6  # lcm(2, 3)


@given(perms())
def test_order_is_minimal_power(g: Perm) -> None:
    p = Perm.identity(N)
    for _ in range(g.order()):
        p = p * g
    assert p.is_identity()


# ── PermGroup ─────────────────────────────────────────────────────────────


@pytest.fixture
def s4() -> PermGroup:
    return PermGroup([Perm.from_cycles([(0, 1)], n=4), Perm.from_cycles([(0, 1, 2, 3)], n=4)])


def test_s4_order(s4: PermGroup) -> None:
    assert s4.order == 24


def test_cyclic_group_order() -> None:
    assert PermGroup([Perm.from_cycles([(0, 1, 2, 3, 4)], n=5)]).order == 5


def test_cube_rotation_group_order(cube_rotations: PermGroup) -> None:
    assert cube_rotations.order == 24


def test_closure(s4: PermGroup) -> None:
    for g in s4.elements:
        for h in s4.elements:
            assert g * h in s4


def test_contains_inverses(s4: PermGroup) -> None:
    for g in s4.elements:
        assert g.inverse() in s4


def test_lagrange(cube_rotations: PermGroup) -> None:
    for g in cube_rotations.elements:
        assert cube_rotations.order % g.order() == 0


def test_symmetric_group_orders() -> None:
    assert [symmetric_group(n).order for n in range(1, 6)] == [1, 2, 6, 24, 120]


def test_direct_product_order(s4: PermGroup) -> None:
    assert direct_product(s4, symmetric_group(3)).order == 24 * 6


@given(perms(3), perms(2), perms(3), perms(2))
def test_direct_sum_is_homomorphism(a: Perm, b: Perm, c: Perm, d: Perm) -> None:
    # (a ⊕ b)(c ⊕ d) == (ac) ⊕ (bd)
    assert a.direct_sum(b) * c.direct_sum(d) == (a * c).direct_sum(b * d)
