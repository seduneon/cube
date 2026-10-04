"""Puzzles and their symmetry groups. NumPy only, never torch."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from eqcube.puzzles.base import PermutationPuzzle, bfs_distances, state_keys
from eqcube.puzzles.pocket_coords import PocketCubeCoords, pocket_coords
from eqcube.puzzles.pocket_cube import MAX_DISTANCE, SYMMETRIES, pocket_cube
from eqcube.symmetry.perm import PermGroup


@dataclass(frozen=True)
class PuzzleSpec:
    """Everything a model builder needs to know about a puzzle."""

    make: Callable[[], PermutationPuzzle]
    symmetries: Mapping[str, Callable[[], PermGroup]]
    max_distance: int
    make_indexer: Callable[[], PocketCubeCoords] | None = None


PUZZLES: dict[str, PuzzleSpec] = {
    "pocket_cube": PuzzleSpec(pocket_cube, SYMMETRIES, MAX_DISTANCE, pocket_coords),
}

__all__ = ["PUZZLES", "PermutationPuzzle", "PuzzleSpec", "bfs_distances", "state_keys"]
