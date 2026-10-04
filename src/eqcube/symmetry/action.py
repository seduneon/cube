"""Group actions and the generic algorithms on them (orbit, stabilizer).

An action of a group G on a set X is a map ``(g, x) -> g·x`` such that

    e·x == x                      (identity law)
    g·(h·x) == (g * h)·x          (compatibility law)

Orbits and stabilizers belong to the *action*, not the group: the same
group has different orbits on points, on pairs of points, on states, ...
"""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from eqcube.symmetry.perm import Perm, PermGroup

X = TypeVar("X", bound=Hashable)


class Action(Protocol[X]):
    """A left action of ``group`` on a set of hashable values X."""

    @property
    def group(self) -> PermGroup: ...

    def __call__(self, g: Perm, x: X, /) -> X: ...


# ── concrete actions ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class PointAction:
    """g·i = g(i): the natural action on points 0..n-1."""

    group: PermGroup

    def __call__(self, g: Perm, x: int, /) -> int:
        return g(x)


@dataclass(frozen=True)
class TupleAction:
    """g·(i1, ..., ik) = (g(i1), ..., g(ik)).

    With k = 2 its orbits are exactly the weight-sharing classes of an
    equivariant linear layer.
    """

    group: PermGroup

    def __call__(self, g: Perm, x: tuple[int, ...], /) -> tuple[int, ...]:
        return tuple(g(i) for i in x)


@dataclass(frozen=True)
class FunctionAction:
    """Action on functions f: domain -> codomain, stored as ``f[p]``.

    ``group`` acts on the disjoint union of both sets: points
    0..n_domain-1 are the domain, the remaining points are the codomain.
    (For cube states: domain = 24 positions, codomain = 6 colors.)

        (g·f)(p) = g( f( g^{-1}(p) ) )

    The g^{-1} is what makes this a *left* action: the value now at p is
    the one that g moved there, i.e. the one that was at g^{-1}(p).
    If the group has no codomain points, values are left unchanged.
    """

    group: PermGroup
    n_domain: int

    def __post_init__(self) -> None:
        domain = range(self.n_domain)
        for s in self.group.generators:
            if any(s(p) >= self.n_domain for p in domain):
                raise ValueError("group must map the domain to itself")

    @property
    def n_codomain(self) -> int:
        return self.group.degree - self.n_domain

    def __call__(self, g: Perm, x: tuple[int, ...], /) -> tuple[int, ...]:
        g_inv = g.inverse()
        n = self.n_domain
        if self.n_codomain == 0:
            return tuple(x[g_inv(p)] for p in range(n))
        return tuple(g(x[g_inv(p)] + n) - n for p in range(n))


@dataclass(frozen=True)
class ConjugationAction:
    """g·h = g h g^{-1}: the group acting on permutations of the same degree.

    Used on puzzle moves: a symmetry must map moves to moves.
    """

    group: PermGroup

    def __call__(self, g: Perm, x: Perm, /) -> Perm:
        return g * x * g.inverse()


@dataclass(frozen=True)
class SetAction(Generic[X]):
    """g·{x1, ..., xk} = {g·x1, ..., g·xk}: lifts any action to finite sets."""

    base: Action[X]

    @property
    def group(self) -> PermGroup:
        return self.base.group

    def __call__(self, g: Perm, x: frozenset[X], /) -> frozenset[X]:
        return frozenset(self.base(g, y) for y in x)


# ── generic algorithms (work for any Action) ──────────────────────────────


def orbit(action: Action[X], x: X) -> frozenset[X]:
    """G·x, by BFS over the generators (never enumerates the group)."""
    seen = {x}
    queue = deque(seen)
    while queue:
        y = queue.popleft()
        for s in action.group.generators:
            z = action(s, y)
            if z not in seen:
                seen.add(z)
                queue.append(z)
    return frozenset(seen)


def stabilizer(action: Action[X], x: X) -> PermGroup:
    """Subgroup {g in G : g·x == x}, by filtering all elements."""
    return PermGroup(g for g in action.group.elements if action(g, x) == x)


def orbits(action: Action[X], xs: Iterable[X]) -> list[frozenset[X]]:
    """Partition ``xs`` into orbits. ``xs`` must be closed under the action."""
    remaining = set(xs)
    result: list[frozenset[X]] = []
    while remaining:
        o = orbit(action, next(iter(remaining)))
        result.append(o)
        remaining -= o
    return result
