"""Shared fixtures. Cube data lives here until a puzzles/ package exists."""

import pytest

from eqcube.symmetry.perm import Perm, PermGroup

# Sticker i has color i // 4 (see legacy/cube_env.py for the labeling).
SOLVED: tuple[int, ...] = tuple(i // 4 for i in range(24))


@pytest.fixture(scope="session")
def cube_rotations() -> PermGroup:
    """Whole-cube rotations O acting on the 24 stickers (from legacy/cube_group.py)."""
    x_rot = Perm.from_cycles(
        [
            (3, 12, 7, 11),
            (9, 1, 14, 5),
            (20, 21, 23, 22),
            (2, 13, 6, 10),
            (8, 0, 15, 4),
            (17, 16, 18, 19),
        ],
        n=24,
    )
    y_rot = Perm.from_cycles(
        [
            (0, 1, 3, 2),
            (9, 17, 13, 21),
            (8, 16, 12, 20),
            (4, 6, 7, 5),
            (10, 18, 14, 22),
            (19, 15, 23, 11),
        ],
        n=24,
    )
    return PermGroup([x_rot, y_rot])
