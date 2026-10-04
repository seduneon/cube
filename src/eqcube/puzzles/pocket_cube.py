"""The 2x2x2 Pocket Cube with moves R, U, F (quarter turns).

Sticker labeling (0-23), reading order per face:
  U: 0=UBL  1=UBR  2=UFL  3=UFR        D: 4=DFL  5=DFR  6=DBL  7=DBR
  F: 8=UFL  9=UFR 10=DFL 11=DFR        B: 12=UBR 13=UBL 14=DBR 15=DBL
  L: 16=UBL 17=UFL 18=DBL 19=DFL       R: 20=UFR 21=UBR 22=DFR 23=DBR
Sticker i has color i // 4 when solved (U=0, D=1, F=2, B=3, L=4, R=5).

R, U, F never move the DBL corner (stickers 6, 15, 18), which pins the
cube's orientation.

Every symmetry group here acts on 24 positions + 6 colors (degree 30), so
all models share one input encoding and the groups are directly comparable.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache

from eqcube.puzzles.base import PermutationPuzzle
from eqcube.symmetry.perm import Perm, PermGroup, direct_product, symmetric_group

N_POSITIONS = 24
N_COLORS = 6
MAX_DISTANCE = 14  # diameter of this puzzle's Cayley graph (quarter-turn metric, R/U/F)

_R = Perm.from_cycles([(3, 12, 7, 11), (9, 1, 14, 5), (20, 21, 23, 22)], N_POSITIONS)
_U = Perm.from_cycles([(0, 1, 3, 2), (9, 17, 13, 21), (8, 16, 12, 20)], N_POSITIONS)
_F = Perm.from_cycles([(8, 9, 11, 10), (3, 22, 4, 17), (20, 5, 19, 2)], N_POSITIONS)

# Whole-cube rotations: x along the R-L axis, y along the U-D axis.
_X_ROT = Perm.from_cycles(
    [
        (3, 12, 7, 11),
        (9, 1, 14, 5),
        (20, 21, 23, 22),
        (2, 13, 6, 10),
        (8, 0, 15, 4),
        (17, 16, 18, 19),
    ],
    N_POSITIONS,
)
_Y_ROT = Perm.from_cycles(
    [
        (0, 1, 3, 2),
        (9, 17, 13, 21),
        (8, 16, 12, 20),
        (4, 6, 7, 5),
        (10, 18, 14, 22),
        (19, 15, 23, 11),
    ],
    N_POSITIONS,
)


@cache
def pocket_cube() -> PermutationPuzzle:
    return PermutationPuzzle(
        name="pocket_cube",
        move_names=("R", "R'", "U", "U'", "F", "F'"),
        moves=(_R, _R.inverse(), _U, _U.inverse(), _F, _F.inverse()),
        solved=tuple(i // 4 for i in range(N_POSITIONS)),
        n_colors=N_COLORS,
    )


# ── symmetry groups (all of degree 24 + 6) ────────────────────────────────


@cache
def rotation_group() -> PermGroup:
    """O (24) acting on positions only; colors fixed."""
    return direct_product(PermGroup([_X_ROT, _Y_ROT]), PermGroup([Perm.identity(N_COLORS)]))


@cache
def color_group() -> PermGroup:
    """S6 (720) relabelling colors; positions fixed."""
    return direct_product(PermGroup([Perm.identity(N_POSITIONS)]), symmetric_group(N_COLORS))


@cache
def rotation_color_group() -> PermGroup:
    """O x S6 (17,280): rotations and recolorings independently."""
    return direct_product(PermGroup([_X_ROT, _Y_ROT]), symmetric_group(N_COLORS))


@cache
def trivial_group() -> PermGroup:
    return PermGroup([Perm.identity(N_POSITIONS + N_COLORS)])


@cache
def task_group() -> PermGroup:
    """The true symmetry of the task, computed (not assumed) inside O x S6."""
    return pocket_cube().task_symmetry(rotation_color_group())


SYMMETRIES: dict[str, Callable[[], PermGroup]] = {
    "trivial": trivial_group,
    "rotation": rotation_group,
    "color": color_group,
    "rotation_color": rotation_color_group,
    "task": task_group,
}
