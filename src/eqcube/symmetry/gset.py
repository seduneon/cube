"""Finite G-sets: the index sets that feature tensors live on.

A ``GSet`` is the set {0, ..., size-1} together with an action of a fixed
group G, given as a homomorphism G -> S_size. We store the homomorphism by
the images of G's generators (that determines it completely).

Feature tensors in ``eqcube.nn`` have shape ``(batch, gset.size, channels)``;
G moves the middle axis and leaves channels alone.

Constructions mirror the math:
    GSet.natural(G)           G on its own points
    GSet.block(G, start, n)   G restricted to an invariant block of points
    GSet.trivial(G, n)        every g acts as the identity
    A.product(B)              G on pairs (a, b), index a * |B| + b
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import cache, cached_property

import numpy as np
import numpy.typing as npt

from eqcube.symmetry.perm import Perm, PermGroup

IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True, eq=False)
class GSet:
    """{0..size-1} with G acting via ``images[k]`` = action of ``group.generators[k]``.

    ``eq=False``: equality and hashing are by identity. Build each GSet once
    and share it; that also makes the orbit caches below effective.
    """

    group: PermGroup
    images: tuple[Perm, ...]

    def __post_init__(self) -> None:
        if len(self.images) != len(self.group.generators):
            raise ValueError("need exactly one image per generator")
        if len({p.degree for p in self.images}) != 1:
            raise ValueError("all images must have the same degree")

    # ── constructors ──────────────────────────────────────────────────────

    @classmethod
    def natural(cls, group: PermGroup) -> GSet:
        return cls(group, group.generators)

    @classmethod
    def block(cls, group: PermGroup, start: int, size: int) -> GSet:
        """Restrict to points start..start+size-1, relabelled to 0..size-1."""
        stop = start + size
        images: list[Perm] = []
        for s in group.generators:
            moved = [s(p) for p in range(start, stop)]
            if any(not start <= q < stop for q in moved):
                raise ValueError(f"block [{start}, {stop}) is not invariant under {s}")
            images.append(Perm(tuple(q - start for q in moved)))
        return cls(group, tuple(images))

    @classmethod
    def trivial(cls, group: PermGroup, size: int) -> GSet:
        e = Perm.identity(size)
        return cls(group, tuple(e for _ in group.generators))

    def product(self, other: GSet) -> GSet:
        """G acting diagonally on pairs: g·(a, b) = (g·a, g·b), index a*|B| + b."""
        if other.group is not self.group:
            raise ValueError("product of G-sets requires the same group object")
        m = other.size
        images = tuple(
            Perm(tuple(a.arr[i] * m + b.arr[j] for i in range(self.size) for j in range(m)))
            for a, b in zip(self.images, other.images, strict=True)
        )
        return GSet(self.group, images)

    # ── structure ─────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return self.images[0].degree

    def is_trivial(self) -> bool:
        return all(p.is_identity() for p in self.images)

    @cached_property
    def image_of(self) -> dict[Perm, Perm]:
        """The homomorphism on *every* group element (BFS over the group).

        Raises if the generator images do not define a homomorphism.
        """
        e = self.group.identity
        table = {e: Perm.identity(self.size)}
        queue = deque([e])
        while queue:
            g = queue.popleft()
            for s, s_img in zip(self.group.generators, self.images, strict=True):
                h, h_img = s * g, s_img * table[g]
                if h not in table:
                    table[h] = h_img
                    queue.append(h)
                elif table[h] != h_img:
                    raise ValueError("generator images do not define a homomorphism")
        return table

    def gather_index(self, g: Perm) -> IntArray:
        """Index ``idx`` with ``(g·x)[..., p, :] == x[..., idx[p], :]`` (gather with g^{-1})."""
        return np.array(self.image_of[g].inverse().arr, dtype=np.int64)

    @cached_property
    def orbit_labels(self) -> tuple[IntArray, int]:
        """Orbit id of every point, and the number of orbits."""
        return _propagate_min_labels(np.arange(self.size), [p.arr for p in self.images])


@cache
def pair_orbit_labels(out: GSet, inp: GSet) -> tuple[IntArray, int]:
    """Orbit id of every pair (j, i) in out x inp, shape (|out|, |inp|).

    These orbits are the weight-sharing classes of an equivariant linear map
    from functions on ``inp`` to functions on ``out``: W[j, i] may depend only
    on the orbit of (j, i).
    """
    pairs = out.product(inp)
    labels, n = pairs.orbit_labels
    return labels.reshape(out.size, inp.size), n


def _propagate_min_labels(labels: IntArray, perms: list[tuple[int, ...]]) -> tuple[IntArray, int]:
    """Orbits = connected components of the graph x -- g(x) over generators g.

    Every point repeatedly takes the minimum label of its neighbours (forward
    and backward along each generator) until nothing changes; then labels are
    renumbered 0..k-1. Vectorised, so fast even for 10^5 points.
    """
    fwd = [np.array(p, dtype=np.int64) for p in perms]
    bwd = [np.argsort(f) for f in fwd]
    while True:
        new = labels.copy()
        for f, b in zip(fwd, bwd, strict=True):
            new = np.minimum(new, labels[f])
            new = np.minimum(new, labels[b])
        if np.array_equal(new, labels):
            break
        labels = new
    _, compact = np.unique(labels, return_inverse=True)
    return compact.astype(np.int64), int(compact.max()) + 1
