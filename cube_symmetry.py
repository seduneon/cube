"""
Symmetry groups for the 2x2x2 Pocket Cube.

Centralizes both spatial (O, 24 elements) and color (S6, 720 elements)
symmetries used by the equivariant layers.

Exports:
  ALL_ROTATIONS              - list[ndarray]  24 spatial rotation perms (int8, len 24)
  ALL_COLOR_PERMS            - ndarray (720, 6)  all S6 permutations (int8)
  compute_spatial_pair_orbits() -> (pair_orbit (24,24 int64), n_orbits int)
  compute_color_pair_orbits()   -> (color_orbit (6,6 int64), n_orbits=2)
  apply_color_perm_batch(X, perm) -> ndarray  permute color channels of flat one-hot batch
"""

from itertools import permutations
import numpy as np

# Re-export spatial rotation utilities from cube_group
from cube_group import ALL_ROTATIONS, apply_rotation  # noqa: F401

# ─── Color permutation group S6 ──────────────────────────────────────────────

# All 720 permutations of 6 color indices.  ALL_COLOR_PERMS[i] is a length-6
# int8 array that maps old color index -> new color index.
ALL_COLOR_PERMS = np.array(list(permutations(range(6))), dtype=np.int8)  # (720, 6)


# ─── Orbit computation ────────────────────────────────────────────────────────

def compute_spatial_pair_orbits():
    """Orbit labels for all ordered position pairs (i, j) under the 24 spatial rotations.

    Two pairs (i,j) and (i',j') are in the same orbit iff there exists a
    rotation g with g(i)=i' and g(j)=j'.

    Returns
    -------
    pair_orbit : int64 ndarray shape (24, 24)
        pair_orbit[i, j] in [0, n_orbits)
    n_orbits : int
    """
    pair_orbit = np.full((24, 24), -1, dtype=np.int64)
    orbit_id = 0
    for i in range(24):
        for j in range(24):
            if pair_orbit[i, j] == -1:
                for g in ALL_ROTATIONS:
                    gi, gj = int(g[i]), int(g[j])
                    if pair_orbit[gi, gj] == -1:
                        pair_orbit[gi, gj] = orbit_id
                orbit_id += 1
    assert (pair_orbit >= 0).all(), "Some (i,j) pairs were not assigned an orbit."
    return pair_orbit, orbit_id


def compute_color_pair_orbits():
    """Orbit labels for all ordered color pairs (a, b) under S6.

    Under S6 acting on pairs:
      - All same-color pairs (a, a) form one orbit  → label 0
      - All different-color pairs (a, b) with a≠b form one orbit → label 1

    Returns
    -------
    color_orbit : int64 ndarray shape (6, 6)
        color_orbit[a, b] in {0, 1}
    n_orbits : int  (always 2)
    """
    color_orbit = np.ones((6, 6), dtype=np.int64)   # off-diagonal = 1
    np.fill_diagonal(color_orbit, 0)                 # diagonal = 0
    return color_orbit, 2


# ─── Augmentation helper ─────────────────────────────────────────────────────

def apply_color_perm_batch(X_flat, perm):
    """Apply a color permutation to a batch of flat one-hot encoded states.

    Parameters
    ----------
    X_flat : float32 ndarray shape (N, 144)
        Batch of one-hot encoded states (24 positions × 6 colors, flattened).
    perm : int array-like, length 6
        Color permutation: new_state[pos, perm[c]] = old_state[pos, c].
        Equivalently, the output color channel perm[c] gets the input's channel c.

    Returns
    -------
    ndarray shape (N, 144)  — copy with color channels permuted.
    """
    return X_flat.reshape(-1, 24, 6)[:, :, perm].reshape(-1, 144).copy()


# ─── Verification ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"ALL_ROTATIONS:   {len(ALL_ROTATIONS)} elements")
    print(f"ALL_COLOR_PERMS: {ALL_COLOR_PERMS.shape}  (expect (720, 6))")

    sp_orbit, n_sp = compute_spatial_pair_orbits()
    print(f"Spatial pair orbits: {n_sp}  (expect 10–40)")

    co_orbit, n_co = compute_color_pair_orbits()
    print(f"Color pair orbits:   {n_co}  (expect 2)")
    print(f"Total kernel orbits: {n_sp * n_co}")

    # Verify color orbit structure
    assert co_orbit[0, 0] == 0 and co_orbit[0, 1] == 1, "Color orbit structure wrong"
    assert int(co_orbit.sum()) == 30, f"Expected 30 off-diagonal entries, got {co_orbit.sum()}"
    print("Color orbit structure: OK")

    # Verify apply_color_perm_batch
    import numpy as np
    X = np.eye(6, dtype=np.float32)[np.newaxis].repeat(24, axis=1).reshape(1, 144)  # noqa
    perm = np.array([1, 0, 2, 3, 4, 5], dtype=np.int8)  # swap colors 0 and 1
    X_perm = apply_color_perm_batch(X, perm)
    assert not np.allclose(X, X_perm), "Permutation had no effect"
    print("apply_color_perm_batch: OK")
