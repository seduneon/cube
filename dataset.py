"""
Dataset generation for the 2x2x2 Pocket Cube experiment.

Step 1: BFS from the solved state to compute exact optimal distances
        and optimal first moves for all 3,674,160 reachable states.

Step 2: Generate training/validation/test splits by backward scrambling.
"""

import numpy as np
import os
import pickle
from collections import deque
from tqdm import tqdm

from cube_env import (
    SOLVED_STATE, MOVES, MOVE_NAMES, INVERSE_MOVE, NUM_MOVES,
    apply_move, encode_state, state_to_tuple, tuple_to_state,
)

# Pre-computed move candidates for each last_move value (-1 = no last move).
# Avoids rebuilding the list comprehension on every scramble step.
_CANDIDATES: dict = {-1: list(range(NUM_MOVES))}
_CANDIDATES.update(
    {m: [i for i in range(NUM_MOVES) if i != INVERSE_MOVE[m]] for m in range(NUM_MOVES)}
)

# Expected number of reachable states (God's number = 14 in QTM with R,U,F quarter turns)
TOTAL_STATES = 3_674_160
MAX_DISTANCE = 14  # God's number in QTM (we use only R,R',U,U',F,F' quarter turns)

DATA_DIR = os.path.join(os.path.dirname(__file__), "results", "data")


# ─── BFS ──────────────────────────────────────────────────────────────────────

def run_bfs(verbose=True):
    """BFS from solved state over all 3,674,160 reachable states.

    Returns:
        dist_table: dict mapping state_tuple -> optimal_distance (int)
        move_table: dict mapping state_tuple -> optimal_first_move_index (int)
                    The move that leads to a neighbor with distance d-1.
                    (For the solved state, move is -1.)
    """
    solved_tuple = state_to_tuple(SOLVED_STATE)
    dist_table = {solved_tuple: 0}
    move_table = {solved_tuple: -1}

    queue = deque([SOLVED_STATE.copy()])
    n_visited = 1

    if verbose:
        print(f"Starting BFS from solved state...")
        print(f"Expected states: {TOTAL_STATES:,}")
        pbar = tqdm(total=TOTAL_STATES, desc="BFS")
        pbar.update(1)

    while queue:
        state = queue.popleft()
        current_dist = dist_table[state_to_tuple(state)]

        for move_idx in range(NUM_MOVES):
            next_state = apply_move(state, MOVES[move_idx])
            next_tuple = state_to_tuple(next_state)

            if next_tuple not in dist_table:
                dist_table[next_tuple] = current_dist + 1
                # The optimal move from next_state towards solved is the inverse
                # of the move that got us here (since BFS is from solved outward).
                move_table[next_tuple] = INVERSE_MOVE[move_idx]
                queue.append(next_state)
                n_visited += 1

                if verbose:
                    pbar.update(1)

    if verbose:
        pbar.close()
        print(f"BFS complete. Visited {n_visited:,} states.")
        # Print distance distribution
        from collections import Counter
        dist_counts = Counter(dist_table.values())
        actual_max = max(dist_counts.keys())
        print(f"\nDistance distribution (max_dist={actual_max}):")
        for d in range(actual_max + 1):
            print(f"  d={d:2d}: {dist_counts.get(d, 0):>10,} states")

    return dist_table, move_table


def save_bfs_tables(dist_table, move_table, path=None):
    """Save BFS tables to disk."""
    if path is None:
        os.makedirs(DATA_DIR, exist_ok=True)
        path = os.path.join(DATA_DIR, "bfs_tables.pkl")
    with open(path, "wb") as f:
        pickle.dump({"dist": dist_table, "move": move_table}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    print(f"BFS tables saved to {path}")


def load_bfs_tables(path=None):
    """Load BFS tables from disk."""
    if path is None:
        path = os.path.join(DATA_DIR, "bfs_tables.pkl")
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["dist"], data["move"]


def bfs_tables_exist(path=None):
    if path is None:
        path = os.path.join(DATA_DIR, "bfs_tables.pkl")
    return os.path.exists(path)


# ─── Dataset generation ───────────────────────────────────────────────────────

def generate_dataset(n_samples, dist_table, rng=None,
                     max_scramble=MAX_DISTANCE, verbose=True):
    """Generate a dataset of (encoded_state, distance) pairs.

    For each sample:
      1. Start from solved state
      2. Apply k ~ Uniform(1, max_scramble) random moves
      3. Look up true BFS distance

    Returns:
        X: (n_samples, 144) float32  — one-hot encoded states
        y_dist: (n_samples,) int32   — BFS optimal distance (0-14)
    """
    if rng is None:
        rng = np.random.RandomState(42)

    X = np.zeros((n_samples, 144), dtype=np.float32)
    y_dist = np.zeros(n_samples, dtype=np.int32)

    iterator = tqdm(range(n_samples), desc="Generating") if verbose else range(n_samples)

    for i in iterator:
        state = SOLVED_STATE.copy()
        k = rng.randint(1, max_scramble + 1)

        last_move = -1
        for _ in range(k):
            move = rng.choice(_CANDIDATES[last_move])
            state = apply_move(state, MOVES[move])
            last_move = move

        state_t = state_to_tuple(state)
        X[i] = encode_state(state)
        y_dist[i] = dist_table[state_t]

    return X, y_dist


def _index_states_by_depth(dist_table, verbose=True):
    """Return dict[depth -> list[state_tuple]] for depths 1..MAX_DISTANCE."""
    states_by_depth = {d: [] for d in range(1, MAX_DISTANCE + 1)}
    iterator = tqdm(dist_table.items(), total=len(dist_table),
                    desc="Indexing BFS table", disable=not verbose)
    for state_t, d in iterator:
        if 1 <= d <= MAX_DISTANCE:
            states_by_depth[d].append(state_t)
    return states_by_depth


def _sample_from_depth_buckets(n_per_depth_dict, states_by_depth, rng, verbose):
    """Given a dict[depth -> n_samples], draw states and return X, y arrays."""
    X_list, y_list = [], []
    for d in range(1, MAX_DISTANCE + 1):
        pool = states_by_depth[d]
        n = n_per_depth_dict[d]
        if n == 0:
            continue
        replace = len(pool) < n
        indices = rng.choice(len(pool), size=n, replace=replace)
        for i in indices:
            X_list.append(encode_state(tuple_to_state(pool[i])))
            y_list.append(d)
        if verbose:
            print(f"  depth {d:2d}: {len(pool):>8,} states, sampled {n}"
                  + (" (with replacement)" if replace else ""))

    perm = rng.permutation(len(X_list))
    X = np.array(X_list, dtype=np.float32)[perm]
    y = np.array(y_list, dtype=np.int32)[perm]
    return X, y


def generate_train_dataset_stratified(n_total, dist_table, rng=None, verbose=True):
    """Generate a training set with equal samples per BFS depth (1–14).

    Samples n_total // 14 states from each depth bucket.  Depths with fewer
    states than required are oversampled with replacement (only depths 1–3
    are small enough to trigger this).  The result is shuffled.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    states_by_depth = _index_states_by_depth(dist_table, verbose)

    n_depths = MAX_DISTANCE  # 14 depth levels
    base = n_total // n_depths
    remainder = n_total - base * n_depths
    n_per_depth = {d: base + (1 if d > MAX_DISTANCE - remainder else 0)
                   for d in range(1, MAX_DISTANCE + 1)}

    return _sample_from_depth_buckets(n_per_depth, states_by_depth, rng, verbose)


def generate_train_dataset_sqrtweighted(n_total, dist_table, rng=None, verbose=True):
    """Generate a training set with samples per depth ∝ √(state count at that depth).

    This is a principled compromise between:
      - Equal stratification: over-represents rare depths (depth 14 has only 276
        states, so equal sampling repeats each one ~52× at 200K total)
      - Proportional sampling: ~70% of samples from depths 10–12, depths 0–6 starved

    √count weighting: hard depths still get more coverage than proportional gives
    them, but depth-14 inflation is dramatically reduced (717 samples instead of
    14K), and the bulk of training is concentrated where most of the state space
    actually lives (depths 8–12).

    Example (200K total, approximate):
      depth  1:     6 states  →   104 samples  (17× each — vs 14K equal)
      depth  9:  360K states  →  25.9K samples  (vs 14K equal)
      depth 11:  1.35M states → 50.2K samples  (vs 14K equal)
      depth 14:   276 states  →   717 samples  (2.6× each — vs 14K equal)
    """
    if rng is None:
        rng = np.random.RandomState(42)

    states_by_depth = _index_states_by_depth(dist_table, verbose)

    # √count weights
    counts = np.array([len(states_by_depth[d]) for d in range(1, MAX_DISTANCE + 1)],
                      dtype=np.float64)
    sqrt_counts = np.sqrt(counts)
    proportions = sqrt_counts / sqrt_counts.sum()

    # Distribute n_total proportionally, resolving rounding to hit exact total
    n_float = proportions * n_total
    n_per_arr = np.floor(n_float).astype(int)
    remainder = n_total - n_per_arr.sum()
    frac_parts = n_float - n_per_arr
    for idx in np.argsort(-frac_parts)[:remainder]:
        n_per_arr[idx] += 1

    n_per_depth = {d: int(n_per_arr[d - 1]) for d in range(1, MAX_DISTANCE + 1)}

    if verbose:
        print(f"√-weighted sampling: {n_total:,} total over {MAX_DISTANCE} depths")
    return _sample_from_depth_buckets(n_per_depth, states_by_depth, rng, verbose)


def generate_test_dataset_stratified(n_per_depth, dist_table,
                                     rng=None, verbose=True):
    """Generate a test set stratified by depth (roughly n_per_depth per depth).

    Returns X, y_dist (same format as generate_dataset).
    """
    if rng is None:
        rng = np.random.RandomState(999)

    # Collect states at each distance
    if verbose:
        print("Collecting states by depth for stratified test set...")

    actual_max = max(dist_table.values())
    states_by_depth = {d: [] for d in range(actual_max + 1)}
    for state_t, d in tqdm(dist_table.items(), total=len(dist_table),
                            disable=not verbose, desc="Indexing"):
        states_by_depth[d].append(state_t)

    X_list, y_dist_list = [], []

    for d in range(MAX_DISTANCE + 1):
        pool = states_by_depth[d]
        n = min(n_per_depth, len(pool))
        indices = rng.choice(len(pool), size=n, replace=False)
        selected = [pool[i] for i in indices]

        for state_t in selected:
            state = tuple_to_state(state_t)
            X_list.append(encode_state(state))
            y_dist_list.append(dist_table[state_t])

    X = np.array(X_list, dtype=np.float32)
    y_dist = np.array(y_dist_list, dtype=np.int32)
    return X, y_dist


_STRATEGY_SUFFIX = {
    'random': '',       # legacy: X_val.npy, X_train_200k.npy
    'equal':  '_strat', # legacy: X_train_200k_strat.npy; new: X_val_strat.npy
    'sqrt':   '_sqrt',  # new:    X_train_200k_sqrt.npy
}

# RNG seeds per split so datasets are reproducible
_SPLIT_RNG_BASE = {'train': 300, 'val': 100, 'test': 200}


def load_dataset(split_name, n_train=None, strategy='random'):
    """Load dataset arrays from disk, generating them on first use.

    strategy : 'random' | 'equal' | 'sqrt'
      'random' — random scramble (legacy baseline)
      'equal'  — equal samples per BFS depth (stratified)
      'sqrt'   — samples ∝ √(state count at depth) — recommended for training
    For val/test splits use 'random' or 'equal'.
    """
    suffix = f"_{n_train // 1000}k" if n_train is not None else ""
    tag = f"{split_name}{suffix}{_STRATEGY_SUFFIX[strategy]}"

    x_path = os.path.join(DATA_DIR, f"X_{tag}.npy")
    y_path = os.path.join(DATA_DIR, f"y_dist_{tag}.npy")

    if not os.path.exists(x_path):
        print(f"Dataset not found — generating: {tag}")
        dist_table, _ = load_bfs_tables()
        seed = _SPLIT_RNG_BASE.get(split_name, 0) + (n_train or 0)
        rng = np.random.RandomState(seed)

        if strategy == 'sqrt':
            if split_name != 'train' or n_train is None:
                raise ValueError("strategy='sqrt' is only valid for train splits")
            X, y = generate_train_dataset_sqrtweighted(n_train, dist_table, rng=rng)
        elif strategy == 'equal':
            if split_name == 'train' and n_train is not None:
                X, y = generate_train_dataset_stratified(n_train, dist_table, rng=rng)
            else:
                # val / test: stratified with up to 1000 per depth
                X, y = generate_test_dataset_stratified(1000, dist_table, rng=rng)
        else:  # 'random'
            n = n_train if (split_name == 'train' and n_train) else 20_000
            X, y = generate_dataset(n, dist_table, rng=rng)

        save_dataset(X, y, split_name, n_train, strategy=strategy)

    return np.load(x_path), np.load(y_path)


def save_dataset(X, y_dist, split_name, n_train=None, strategy='random'):
    """Save dataset arrays to disk."""
    os.makedirs(DATA_DIR, exist_ok=True)
    suffix = f"_{n_train // 1000}k" if n_train is not None else ""
    tag = f"{split_name}{suffix}{_STRATEGY_SUFFIX[strategy]}"
    np.save(os.path.join(DATA_DIR, f"X_{tag}.npy"), X)
    np.save(os.path.join(DATA_DIR, f"y_dist_{tag}.npy"), y_dist)
    print(f"Saved {tag}: X={X.shape}, y_dist={y_dist.shape}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bfs-only", action="store_true",
                        help="Only run BFS, skip dataset generation")
    parser.add_argument("--skip-bfs", action="store_true",
                        help="Load existing BFS tables instead of recomputing")
    args = parser.parse_args()

    # ── Step 1: BFS ──────────────────────────────────────────────────────────
    if args.skip_bfs and bfs_tables_exist():
        print("Loading existing BFS tables...")
        dist_table, move_table = load_bfs_tables()
        print(f"Loaded {len(dist_table):,} states.")
    else:
        dist_table, move_table = run_bfs(verbose=True)
        save_bfs_tables(dist_table, move_table)

    if args.bfs_only:
        print("BFS only mode — done.")
        exit(0)

    # ── Step 2: Generate datasets ─────────────────────────────────────────────
    TRAIN_SIZES = [50_000, 200_000, 1_000_000]
    TEST_N_PER_DEPTH = 1000  # ~15K total (1000 per depth × 15 depths)

    # Validation: stratified (equal per depth) — better early-stopping signal
    print("\nGenerating stratified validation set (equal, 1000/depth)...")
    X_val, y_val = generate_test_dataset_stratified(
        TEST_N_PER_DEPTH, dist_table, rng=np.random.RandomState(100))
    save_dataset(X_val, y_val, "val", strategy='equal')

    # Test: stratified (equal per depth)
    print("\nGenerating stratified test set (equal, 1000/depth)...")
    X_test, y_test = generate_test_dataset_stratified(
        TEST_N_PER_DEPTH, dist_table, rng=np.random.RandomState(200))
    save_dataset(X_test, y_test, "test", strategy='equal')

    # Training: √-weighted (recommended) + equal (for ablation)
    for n_train in TRAIN_SIZES:
        print(f"\nGenerating √-weighted training set ({n_train:,} samples)...")
        X_tr, y_tr = generate_train_dataset_sqrtweighted(
            n_train, dist_table, rng=np.random.RandomState(300 + n_train))
        save_dataset(X_tr, y_tr, "train", n_train, strategy='sqrt')

        print(f"\nGenerating equal-stratified training set ({n_train:,} samples)...")
        X_tr, y_tr = generate_train_dataset_stratified(
            n_train, dist_table, rng=np.random.RandomState(300 + n_train))
        save_dataset(X_tr, y_tr, "train", n_train, strategy='equal')

    print("\nAll datasets generated.")
