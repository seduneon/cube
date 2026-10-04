"""The 2x2x2 Pocket Cube with moves R, U, F (quarter turns).

Sticker labeling (0-23), reading order per face:
  U: 0=UBL  1=UBR  2=UFL  3=UFR        D: 4=DFL  5=DFR  6=DBL  7=DBR
  F: 8=UFL  9=UFR 10=DFL 11=DFR        B: 12=UBR 13=UBL 14=DBR 15=DBL
  L: 16=UBL 17=UFL 18=DBL 19=DFL       R: 20=UFR 21=UBR 22=DFR 23=DBR
Sticker i has color i // 4 when solved (U=0, D=1, F=2, B=3, L=4, R=5).

Geometry is the single source of truth. Every sticker gets a 3D position
(its corner, its face normal), with R = +x, U = +y, F = +z. Then

    whole-cube symmetry M (3x3 signed permutation matrix):
        sticker (c, n) -> (M c, M n)
    face turn (e.g. R): the same, but only for stickers with c_x = +1

so the moves and the symmetry groups follow one convention by construction.

R, U, F never move the DBL corner (stickers 6, 15, 18), which pins the
cube's orientation. Every symmetry group here acts on 24 positions + 6
colors (degree 30), so all models share one input encoding.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache

import numpy as np
import numpy.typing as npt

from eqcube.puzzles.base import PermutationPuzzle
from eqcube.symmetry.perm import Perm, PermGroup, direct_product, symmetric_group

N_POSITIONS = 24
N_COLORS = 6
MAX_DISTANCE = 14  # diameter of this puzzle's Cayley graph (quarter-turn metric, R/U/F)

Vec = tuple[int, int, int]
Matrix = npt.NDArray[np.int64]

# ── geometry ──────────────────────────────────────────────────────────────

_FACES = "UDFBLR"  # color c = _FACES.index(face)
_NORMAL: dict[str, Vec] = {
    "U": (0, 1, 0), "D": (0, -1, 0),
    "F": (0, 0, 1), "B": (0, 0, -1),
    "R": (1, 0, 0), "L": (-1, 0, 0),
}  # fmt: skip

# Corner name -> its stickers, in the same order as the name's letters.
_CORNERS: dict[str, tuple[int, int, int]] = {
    "UFL": (2, 8, 17), "UFR": (3, 9, 20), "UBL": (0, 13, 16), "UBR": (1, 12, 21),
    "DFL": (4, 10, 19), "DFR": (5, 11, 22), "DBL": (6, 15, 18), "DBR": (7, 14, 23),
}  # fmt: skip


def _add(*vs: Vec) -> Vec:
    return (sum(v[0] for v in vs), sum(v[1] for v in vs), sum(v[2] for v in vs))


def _sticker_geometry() -> dict[int, tuple[Vec, Vec]]:
    """sticker -> (corner position, face normal)."""
    geo: dict[int, tuple[Vec, Vec]] = {}
    for name, stickers in _CORNERS.items():
        corner = _add(*(_NORMAL[f] for f in name))
        for face, s in zip(name, stickers, strict=True):
            geo[s] = (corner, _NORMAL[face])
    return geo


_GEOMETRY = _sticker_geometry()
_STICKER_AT = {v: k for k, v in _GEOMETRY.items()}
assert len(_STICKER_AT) == N_POSITIONS


def _apply(m: Matrix, v: Vec) -> Vec:
    x, y, z = (int(a) for a in m @ np.array(v))
    return (x, y, z)


def _rotation(axis: int, quarter_turns: int) -> Matrix:
    """Rotation by quarter_turns * 90 degrees about +axis (right-hand rule)."""
    m = np.eye(3, dtype=np.int64)
    i, j = (axis + 1) % 3, (axis + 2) % 3  # cyclic order: (y,z), (z,x), (x,y) is right-handed
    q = np.array([[0, -1], [1, 0]], dtype=np.int64)  # +90 degrees in the (i, j) plane
    m[np.ix_([i, j], [i, j])] = np.linalg.matrix_power(q, quarter_turns % 4)
    return m


def sticker_perm(m: Matrix, layer: tuple[int, int] | None = None) -> Perm:
    """The sticker permutation of the isometry ``m``.

    With ``layer = (axis, side)`` only stickers whose corner has
    coordinate ``side`` on ``axis`` move (a face turn); others stay put.
    """
    arr: list[int] = []
    for s in range(N_POSITIONS):
        corner, normal = _GEOMETRY[s]
        if layer is not None and corner[layer[0]] != layer[1]:
            arr.append(s)
        else:
            arr.append(_STICKER_AT[(_apply(m, corner), _apply(m, normal))])
    return Perm(tuple(arr))


def _face_turn(axis: int, side: int) -> Perm:
    """Clockwise quarter turn of the face on ``side`` of ``axis``, seen from outside."""
    return sticker_perm(_rotation(axis, -side), layer=(axis, side))


_X, _Y, _Z = 0, 1, 2
_R = _face_turn(_X, +1)
_U = _face_turn(_Y, +1)
_F = _face_turn(_Z, +1)

# Whole-cube symmetries: x and y turns generate O; adding the point
# reflection -I (det -1) generates O_h = O x {I, -I}.
_X_ROT = sticker_perm(_rotation(_X, -1))
_Y_ROT = sticker_perm(_rotation(_Y, -1))
_POINT_REFLECTION = sticker_perm(-np.eye(3, dtype=np.int64))


def _face_of(sticker: int) -> str:
    normal = _GEOMETRY[sticker][1]
    return next(f for f, n in _NORMAL.items() if n == normal)


@cache
def pocket_cube() -> PermutationPuzzle:
    return PermutationPuzzle(
        name="pocket_cube",
        move_names=("R", "R'", "U", "U'", "F", "F'"),
        moves=(_R, _R.inverse(), _U, _U.inverse(), _F, _F.inverse()),
        solved=tuple(_FACES.index(_face_of(s)) for s in range(N_POSITIONS)),
        n_colors=N_COLORS,
    )


def corner_slots() -> list[tuple[int, int, int]]:
    """The 8 corner slots as sticker triples, DBL (the fixed corner) last.

    Each triple starts with the slot's U/D sticker, and the other two follow
    in one rotational sense for every corner (det(n0, n1, n2) = +1 for the
    face normals), so "twist" means the same thing in every slot.
    """
    order = ("UFR", "UFL", "UBL", "UBR", "DFR", "DFL", "DBR", "DBL")
    slots: list[tuple[int, int, int]] = []
    for name in order:
        stickers = _CORNERS[name]
        ud = next(s for s in stickers if _GEOMETRY[s][1][1] != 0)
        a, b = (s for s in stickers if s != ud)
        normals = np.array([_GEOMETRY[s][1] for s in (ud, a, b)])
        slots.append((ud, a, b) if round(np.linalg.det(normals)) == 1 else (ud, b, a))
    return slots


# ── spatial groups (degree 24) ────────────────────────────────────────────


@cache
def cube_rotations() -> PermGroup:
    """O: the 24 rotations of the cube, acting on stickers."""
    return PermGroup([_X_ROT, _Y_ROT])


@cache
def cube_isometries() -> PermGroup:
    """O_h: the 48 rotations and reflections of the cube, acting on stickers."""
    return PermGroup([_X_ROT, _Y_ROT, _POINT_REFLECTION])


# ── symmetry groups for models (degree 24 + 6) ────────────────────────────


def _no_recolor(spatial: PermGroup) -> PermGroup:
    return direct_product(spatial, PermGroup([Perm.identity(N_COLORS)]))


@cache
def rotation_group() -> PermGroup:
    """O (24) acting on positions only; colors fixed."""
    return _no_recolor(cube_rotations())


@cache
def color_group() -> PermGroup:
    """S6 (720) relabelling colors; positions fixed."""
    return direct_product(PermGroup([Perm.identity(N_POSITIONS)]), symmetric_group(N_COLORS))


@cache
def rotation_color_group() -> PermGroup:
    """O x S6 (17,280): rotations and recolorings independently."""
    return direct_product(cube_rotations(), symmetric_group(N_COLORS))


@cache
def isometry_color_group() -> PermGroup:
    """O_h x S6 (34,560): rotations, reflections and recolorings independently."""
    return direct_product(cube_isometries(), symmetric_group(N_COLORS))


@cache
def trivial_group() -> PermGroup:
    return PermGroup([Perm.identity(N_POSITIONS + N_COLORS)])


@cache
def task_group() -> PermGroup:
    """The true spatial symmetry of the task, computed from O_h.

    Result: S3 (6 elements) -- the rotations about the DBL-UFR diagonal and
    the reflections in planes containing it, each with its recoloring.
    """
    return pocket_cube().task_symmetry(cube_isometries())


SYMMETRIES: dict[str, Callable[[], PermGroup]] = {
    "trivial": trivial_group,
    "rotation": rotation_group,
    "color": color_group,
    "rotation_color": rotation_color_group,
    "isometry_color": isometry_color_group,
    "task": task_group,
}
