"""
2x2x2 Pocket Cube simulator.

Sticker labeling (0-23), reading order per face:
  U face (viewed from above, F at bottom):
    0=UBL  1=UBR  2=UFL  3=UFR
  D face (viewed from below, F at top):
    4=DFL  5=DFR  6=DBL  7=DBR
  F face (viewed from front):
    8=UFL  9=UFR  10=DFL  11=DFR
  B face (viewed from back):
    12=UBR  13=UBL  14=DBR  15=DBL
  L face (viewed from left, F at right):
    16=UBL  17=UFL  18=DBL  19=DFL
  R face (viewed from right, F at left):
    20=UFR  21=UBR  22=DFR  23=DBR

Corner sticker triples:
  UFL: U=2  F=8   L=17
  UFR: U=3  F=9   R=20
  UBL: U=0  B=13  L=16
  UBR: U=1  B=12  R=21
  DFL: D=4  F=10  L=19
  DFR: D=5  F=11  R=22
  DBL: D=6  B=15  L=18
  DBR: D=7  B=14  R=23
"""

import numpy as np
import random

# Solved state: sticker i has color i//4 (face index)
SOLVED_STATE = np.array([i // 4 for i in range(24)], dtype=np.int8)

NUM_STICKERS = 24
NUM_COLORS = 6
NUM_MOVES = 6


def _cycles_to_perm(cycles, n=24):
    """Convert list of cycle tuples to a permutation array.
    perm[i] = j means sticker at position i goes to position j.
    Cycle (a,b,c,d) means a->b->c->d->a.
    """
    perm = list(range(n))
    for cycle in cycles:
        for k in range(len(cycle)):
            perm[cycle[k]] = cycle[(k + 1) % len(cycle)]
    return np.array(perm, dtype=np.int8)


def _inverse_perm(perm):
    """Compute the inverse of a permutation array."""
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm), dtype=perm.dtype)
    return inv


# ─── Move permutations ────────────────────────────────────────────────────────
#
# Derivation:
#   R (right face CW from right): U->B, B->D, D->F, F->U for right layer.
#     Corners: UFR->UBR->DBR->DFR->UFR
#     Cycles: (3,12,7,11)(9,1,14,5)(20,21,23,22)
#
#   U (up face CW from above): F->L, L->B, B->R, R->F for top layer.
#     Corners: UFR->UFL->UBL->UBR->UFR
#     Cycles: (0,1,3,2)(9,17,13,21)(8,16,12,20)
#
#   F (front face CW from front): U->R, R->D, D->L, L->U for front layer.
#     Corners: UFR->DFR->DFL->UFL->UFR
#     Cycles: (8,9,11,10)(3,22,4,17)(20,5,19,2)

_R_CYCLES = [(3, 12, 7, 11), (9, 1, 14, 5), (20, 21, 23, 22)]
_U_CYCLES = [(0, 1, 3, 2), (9, 17, 13, 21), (8, 16, 12, 20)]
_F_CYCLES = [(8, 9, 11, 10), (3, 22, 4, 17), (20, 5, 19, 2)]

R_PERM = _cycles_to_perm(_R_CYCLES)
U_PERM = _cycles_to_perm(_U_CYCLES)
F_PERM = _cycles_to_perm(_F_CYCLES)
RI_PERM = _inverse_perm(R_PERM)  # R'
UI_PERM = _inverse_perm(U_PERM)  # U'
FI_PERM = _inverse_perm(F_PERM)  # F'

# All moves in order: R, R', U, U', F, F'
MOVES = [R_PERM, RI_PERM, U_PERM, UI_PERM, F_PERM, FI_PERM]
MOVE_NAMES = ["R", "R'", "U", "U'", "F", "F'"]

# Inverse move index: INVERSE_MOVE[i] is the index of the inverse of move i
INVERSE_MOVE = [1, 0, 3, 2, 5, 4]


def apply_move(state: np.ndarray, move_perm: np.ndarray) -> np.ndarray:
    """Apply a move permutation to a state.

    move_perm[i] = j means the sticker at position i goes to position j.
    new_state[j] = state[i] for all i.
    """
    new_state = np.empty(NUM_STICKERS, dtype=state.dtype)
    new_state[move_perm] = state
    return new_state


def apply_move_idx(state: np.ndarray, move_idx: int) -> np.ndarray:
    """Apply move by index (0=R, 1=R', 2=U, 3=U', 4=F, 5=F')."""
    return apply_move(state, MOVES[move_idx])


def is_solved(state: np.ndarray) -> bool:
    """Check if the cube is in the solved state."""
    return np.all(state == SOLVED_STATE)


def encode_state(state: np.ndarray) -> np.ndarray:
    """One-hot encode a state: (24,) int -> (144,) float32.

    The 6-dim one-hot block for sticker i starts at index 6*i.
    This encoding is equivariant under CubeRotationGroup:
    rotating sticker positions permutes the 6-dim blocks accordingly.
    """
    return np.eye(NUM_COLORS, dtype=np.float32)[state].flatten()


def scramble(state: np.ndarray, n_moves: int, rng=None):
    """Apply n_moves random moves to state, avoiding immediate backtracks.

    Returns (new_state, move_sequence).
    """
    if rng is None:
        rng = random
    moves_applied = []
    current = state.copy()
    last_move = -1
    for _ in range(n_moves):
        # Avoid immediately undoing the last move
        candidates = [m for m in range(NUM_MOVES) if m != INVERSE_MOVE[last_move]] \
            if last_move >= 0 else list(range(NUM_MOVES))
        move = rng.choice(candidates)
        current = apply_move(current, MOVES[move])
        moves_applied.append(move)
        last_move = move
    return current, moves_applied


def state_to_tuple(state: np.ndarray) -> tuple:
    """Convert state array to a hashable tuple (for BFS dict keys)."""
    return tuple(state.tolist())


def tuple_to_state(t: tuple) -> np.ndarray:
    """Convert tuple back to state array."""
    return np.array(t, dtype=np.int8)


# ─── Self-tests ────────────────────────────────────────────────────────────────

def _verify_moves():
    """Verify that each move applied 4 times returns to identity."""
    state = SOLVED_STATE.copy()
    for move_idx, name in enumerate(MOVE_NAMES):
        s = state.copy()
        for _ in range(4):
            s = apply_move_idx(s, move_idx)
        assert np.all(s == state), f"Move {name}: 4x application did not return to identity!"
    # Verify move + inverse = identity
    for i in range(0, NUM_MOVES, 2):
        s = apply_move_idx(state, i)
        s = apply_move_idx(s, INVERSE_MOVE[i])
        assert np.all(s == state), f"Move {MOVE_NAMES[i]} + inverse did not return to identity!"
    print("All move verification tests passed.")


if __name__ == "__main__":
    _verify_moves()
    print("Solved state:", SOLVED_STATE)
    print("Encoded shape:", encode_state(SOLVED_STATE).shape)
    s, moves = scramble(SOLVED_STATE, 10)
    print(f"After 10-move scramble: {moves}")
    print("State:", s)
    print("Is solved:", is_solved(s))
