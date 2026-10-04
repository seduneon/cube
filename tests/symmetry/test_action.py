"""Action laws, plus the cube facts they let us *compute* instead of assert."""

from collections.abc import Hashable
from typing import TypeVar

import pytest
from hypothesis import given
from hypothesis import strategies as st

from eqcube.symmetry.action import (
    Action,
    FunctionAction,
    PointAction,
    TupleAction,
    orbit,
    orbits,
    stabilizer,
)
from eqcube.symmetry.perm import Perm, PermGroup, direct_product, symmetric_group
from tests.conftest import SOLVED

S4 = symmetric_group(4)
S4_X_S3 = direct_product(S4, symmetric_group(3))  # acts on 4 positions + 3 colors


def elements_of(group: PermGroup) -> st.SearchStrategy[Perm]:
    return st.sampled_from(sorted(group.elements, key=lambda g: g.arr))


functions_4_to_3 = st.tuples(*[st.integers(0, 2)] * 4)

X = TypeVar("X", bound=Hashable)


# ── action laws ───────────────────────────────────────────────────────────


def assert_action_laws(action: Action[X], g: Perm, h: Perm, x: X) -> None:
    e = action.group.identity
    assert action(e, x) == x
    assert action(g, action(h, x)) == action(g * h, x)


@given(elements_of(S4), elements_of(S4), st.integers(0, 3))
def test_point_action_laws(g: Perm, h: Perm, x: int) -> None:
    assert_action_laws(PointAction(S4), g, h, x)


@given(elements_of(S4), elements_of(S4), st.tuples(st.integers(0, 3), st.integers(0, 3)))
def test_tuple_action_laws(g: Perm, h: Perm, x: tuple[int, int]) -> None:
    assert_action_laws(TupleAction(S4), g, h, x)


@given(elements_of(S4_X_S3), elements_of(S4_X_S3), functions_4_to_3)
def test_function_action_laws(g: Perm, h: Perm, f: tuple[int, ...]) -> None:
    assert_action_laws(FunctionAction(S4_X_S3, n_domain=4), g, h, f)


def test_naive_gather_is_a_right_action_not_left() -> None:
    """Why FunctionAction needs g^{-1}: without it, compatibility flips order."""

    def naive(g: Perm, f: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(f[g(p)] for p in range(len(f)))

    g = Perm.from_cycles([(0, 1)], 4)
    h = Perm.from_cycles([(1, 2, 3)], 4)
    f = (0, 1, 2, 3)
    assert naive(g, naive(h, f)) != naive(g * h, f)  # broken as a left action
    assert naive(g, naive(h, f)) == naive(h * g, f)  # it is a right action


def test_function_action_rejects_mixing_domain_and_codomain() -> None:
    with pytest.raises(ValueError):
        FunctionAction(symmetric_group(7), n_domain=4)  # (0 1 ... 6) mixes them


# ── generic algorithms ────────────────────────────────────────────────────


@pytest.mark.parametrize("i", range(4))
def test_orbit_stabilizer_points(i: int) -> None:
    action = PointAction(S4)
    assert len(orbit(action, i)) * stabilizer(action, i).order == S4.order


def test_orbits_partition() -> None:
    pairs = [(i, j) for i in range(4) for j in range(4)]
    parts = orbits(TupleAction(S4), pairs)
    assert sorted(len(o) for o in parts) == [4, 12]  # diagonal, off-diagonal
    assert set().union(*parts) == set(pairs)


# ── cube facts, computed ──────────────────────────────────────────────────


def test_rotations_act_freely_and_transitively(cube_rotations: PermGroup) -> None:
    assert orbit(PointAction(cube_rotations), 0) == frozenset(range(24))
    assert stabilizer(PointAction(cube_rotations), 0).order == 1


def test_sticker_pair_orbits(cube_rotations: PermGroup) -> None:
    """Number of free weights per channel pair in a rotation-equivariant layer."""
    pairs = [(i, j) for i in range(24) for j in range(24)]
    assert len(orbits(TupleAction(cube_rotations), pairs)) == 24  # 576 / 24 (free action)


def test_pure_rotation_does_not_fix_solved(cube_rotations: PermGroup) -> None:
    """Rotating without recoloring: only the identity keeps SOLVED solved."""
    action = FunctionAction(cube_rotations, n_domain=24)
    assert stabilizer(action, SOLVED).order == 1


def test_rotation_plus_recolor_fixes_solved(cube_rotations: PermGroup) -> None:
    """In O x S6, the stabilizer of SOLVED is a 24-element 'diagonal' copy of O:
    each rotation paired with the unique recoloring that undoes it.
    Nobody hand-derived that recoloring -- the stabilizer found it."""
    o_x_s6 = direct_product(cube_rotations, symmetric_group(6))
    assert o_x_s6.order == 24 * 720
    stab = stabilizer(FunctionAction(o_x_s6, n_domain=24), SOLVED)
    assert stab.order == 24
