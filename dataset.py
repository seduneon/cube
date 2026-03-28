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


def generate_train_dataset_stratified(n_total, dist_table, rng=None, verbose=True):
    """Generate a training set with equal samples per BFS depth (1–14).

    Samples n_total // 14 states from each depth bucket.  Depths with fewer
    states than required are oversampled with replacement (only depths 1–3
    are small enough to trigger this).  The result is shuffled.

    This eliminates the bias of random scrambling, which heavily undersamples
    the hardest states (depth 13–14) because long scrambles often cancel out.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    # Index states by exact BFS depth (skip depth 0 — the solved state)
    states_by_depth = {d: [] for d in range(1, MAX_DISTANCE + 1)}
    iterator = tqdm(dist_table.items(), total=len(dist_table),
                    desc="Indexing BFS table", disable=not verbose)
    for state_t, d in iterator:
        if 1 <= d <= MAX_DISTANCE:
            states_by_depth[d].append(state_t)

    n_depths = MAX_DISTANCE  # 14 depth levels
    n_per_depth = n_total // n_depths
    remainder = n_total - n_per_depth * n_depths  # distribute among deepest levels

    X_list, y_list = [], []
    for d in range(1, MAX_DISTANCE + 1):
        pool = states_by_depth[d]
        # give remainder samples to the deepest depths
        n = n_per_depth + (1 if d > MAX_DISTANCE - remainder else 0)
        replace = len(pool) < n  # oversample only if pool too small (depths 1–3)
        indices = rng.choice(len(pool), size=n, replace=replace)
        for i in indices:
            state = tuple_to_state(pool[i])
            X_list.append(encode_state(state))
            y_list.append(d)
        if verbose:
            print(f"  depth {d:2d}: {len(pool):>8,} states, sampled {n}"
                  + (" (with replacement)" if replace else ""))

    # Shuffle so depth order doesn't affect training
    idx = rng.permutation(len(X_list))
    X = np.array(X_list, dtype=np.float32)[idx]
    y = np.array(y_list, dtype=np.int32)[idx]
    return X, y


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


def load_dataset(split_name, n_train=None, stratified=False):
    """Load dataset arrays from disk.

    If stratified=True and the stratified file doesn't exist yet, it is
    generated from the BFS table and cached for future runs.
    """
    suffix = f"_{n_train // 1000}k" if n_train is not None else ""
    strat_suffix = "_strat" if stratified else ""
    tag = f"{split_name}{suffix}{strat_suffix}"

    x_path = os.path.join(DATA_DIR, f"X_{tag}.npy")
    y_path = os.path.join(DATA_DIR, f"y_dist_{tag}.npy")

    if not os.path.exists(x_path):
        if stratified and split_name == "train" and n_train is not None:
            print(f"Stratified train file not found — generating from BFS table...")
            dist_table, _ = load_bfs_tables()
            rng = np.random.RandomState(300 + n_train)
            X, y_dist = generate_train_dataset_stratified(
                n_train, dist_table, rng=rng, verbose=True)
            save_dataset(X, y_dist, split_name, n_train, stratified=True)
        else:
            raise FileNotFoundError(f"Dataset file not found: {x_path}")

    X = np.load(x_path)
    y_dist = np.load(y_path)
    return X, y_dist


def save_dataset(X, y_dist, split_name, n_train=None, stratified=False):
    """Save dataset arrays to disk."""
    os.makedirs(DATA_DIR, exist_ok=True)
    suffix = f"_{n_train // 1000}k" if n_train is not None else ""
    strat_suffix = "_strat" if stratified else ""
    tag = f"{split_name}{suffix}{strat_suffix}"
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
    VAL_SIZE = 20_000
    TEST_N_PER_DEPTH = 1000  # ~11K total test samples

    rng_val = np.random.RandomState(100)
    rng_test = np.random.RandomState(200)

    print("\nGenerating validation set...")
    X_val, y_val_dist = generate_dataset(
        VAL_SIZE, dist_table, rng=rng_val)
    save_dataset(X_val, y_val_dist, "val")

    print("\nGenerating stratified test set...")
    X_test, y_test_dist = generate_test_dataset_stratified(
        TEST_N_PER_DEPTH, dist_table, rng=rng_test)
    save_dataset(X_test, y_test_dist, "test")

    for n_train in TRAIN_SIZES:
        print(f"\nGenerating training set ({n_train:,} samples)...")
        rng_train = np.random.RandomState(300 + n_train)
        X_train, y_train_dist = generate_dataset(
            n_train, dist_table, rng=rng_train)
        save_dataset(X_train, y_train_dist, "train", n_train)

    print("\nAll datasets generated.")
