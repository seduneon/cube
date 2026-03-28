"""
Orchestrator: run the full EMLP vs MLP pocket cube pipeline end-to-end.

Steps:
  1. Verify cube environment (move sanity checks)
  2. Verify rotation group (24 elements)
  3. Run BFS to compute all optimal distances
  4. Generate train/val/test datasets
  5. Train all 18 configurations (EMLP+MLP × 50K/200K/1M × 3 seeds)
  6. Evaluate and generate plots

Usage:
    python run_all.py                  # full pipeline
    python run_all.py --skip-bfs       # reuse existing BFS table
    python run_all.py --skip-data      # reuse existing datasets
    python run_all.py --skip-train     # only evaluate existing checkpoints
    python run_all.py --quick          # small test run (1K samples, 2 epochs)
"""

import time
import argparse

# ─── Argument parsing ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Full EMLP cube pipeline")
    p.add_argument("--skip-bfs", action="store_true",
                   help="Skip BFS, load existing table")
    p.add_argument("--skip-data", action="store_true",
                   help="Skip dataset generation, load existing files")
    p.add_argument("--skip-train", action="store_true",
                   help="Skip training, load existing checkpoints")
    p.add_argument("--quick", action="store_true",
                   help="Quick test: 1K train samples, 2 epochs, 1 seed")
    p.add_argument("--model", choices=["emlp", "mlp", "all"], default="all",
                   help="Which model(s) to train")
    p.add_argument("--eval-size", type=int, default=200_000,
                   choices=[50_000, 200_000, 1_000_000])
    return p.parse_args()


# ─── Step utilities ───────────────────────────────────────────────────────────

def section(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def step_verify_env():
    section("Step 1: Verify cube environment")
    from cube_env import _verify_moves, SOLVED_STATE, encode_state
    _verify_moves()
    enc = encode_state(SOLVED_STATE)
    assert enc.shape == (144,), f"Expected (144,) encoding, got {enc.shape}"
    print("Cube environment OK.")


def step_verify_group():
    section("Step 2: Verify rotation group")
    from cube_group import verify_rotation_group, EMLP_AVAILABLE
    verify_rotation_group()
    if not EMLP_AVAILABLE:
        print("\nWARNING: emlp is not installed!")
        print("Install with one of:")
        print("  pip install emlp")
        print("  pip install 'jax[cpu]' objax emlp")
        print("\nContinuing to verify other steps...")
    else:
        print("emlp available.")


def step_bfs(skip_bfs):
    section("Step 3: BFS for optimal distances")
    from dataset import run_bfs, save_bfs_tables, load_bfs_tables, bfs_tables_exist

    if skip_bfs and bfs_tables_exist():
        print("Loading existing BFS tables...")
        t0 = time.time()
        dist_table, move_table = load_bfs_tables()
        print(f"Loaded {len(dist_table):,} states in {time.time()-t0:.1f}s")
    else:
        t0 = time.time()
        dist_table, move_table = run_bfs(verbose=True)
        elapsed = time.time() - t0
        print(f"BFS completed in {elapsed:.1f}s")
        save_bfs_tables(dist_table, move_table)

    return dist_table, move_table


def step_generate_data(dist_table, skip_data, quick):
    section("Step 4: Generate datasets")
    from dataset import (
        generate_dataset, generate_test_dataset_stratified,
        save_dataset, load_dataset, DATA_DIR,
    )
    import numpy as np
    import os

    if quick:
        TRAIN_SIZES = [1_000]
        VAL_SIZE = 200
        TEST_N_PER_DEPTH = 10
    else:
        TRAIN_SIZES = [50_000, 200_000, 1_000_000]
        VAL_SIZE = 20_000
        TEST_N_PER_DEPTH = 1_000

    if skip_data:
        print("Skipping dataset generation (using existing files).")
        return TRAIN_SIZES

    # Validation set
    val_path = os.path.join(DATA_DIR, "X_val.npy")
    if not os.path.exists(val_path):
        print(f"\nGenerating validation set ({VAL_SIZE:,} samples)...")
        rng = np.random.RandomState(100)
        X_val, y_val_dist = generate_dataset(
            VAL_SIZE, dist_table, rng=rng)
        save_dataset(X_val, y_val_dist, "val")
    else:
        print("Validation set exists — skipping.")

    # Test set
    test_path = os.path.join(DATA_DIR, "X_test.npy")
    if not os.path.exists(test_path):
        print(f"\nGenerating stratified test set (~{TEST_N_PER_DEPTH*12:,} samples)...")
        rng = np.random.RandomState(200)
        X_test, y_test_dist = generate_test_dataset_stratified(
            TEST_N_PER_DEPTH, dist_table, rng=rng)
        save_dataset(X_test, y_test_dist, "test")
    else:
        print("Test set exists — skipping.")

    # Training sets
    for n_train in TRAIN_SIZES:
        suffix = f"_{n_train // 1000}k"
        train_path = os.path.join(DATA_DIR, f"X_train{suffix}.npy")
        if not os.path.exists(train_path):
            print(f"\nGenerating training set ({n_train:,} samples)...")
            rng = np.random.RandomState(300 + n_train)
            X_train, y_train_dist = generate_dataset(
                n_train, dist_table, rng=rng)
            save_dataset(X_train, y_train_dist, "train", n_train)
        else:
            print(f"Train set ({n_train:,}) exists — skipping.")

    return TRAIN_SIZES


def step_train(skip_train, model_filter, quick, train_sizes):
    section("Step 5: Train models")
    from models import EMLP_AVAILABLE
    if not EMLP_AVAILABLE:
        print("emlp not installed — skipping training.")
        print("Install emlp and re-run with --skip-bfs --skip-data")
        return

    from train import train_one, SEEDS, size_label
    import train as train_module

    if skip_train:
        print("Skipping training (using existing checkpoints).")
        return

    if quick:
        # Override training hyperparameters for quick test
        train_module.MAX_EPOCHS = 2
        train_module.EARLY_STOP_PATIENCE = 2
        seeds = [0]
    else:
        seeds = SEEDS

    model_types = ["emlp", "mlp"] if model_filter == "all" else [model_filter]

    results = {}
    for model_type in model_types:
        for train_size in train_sizes:
            for seed in seeds:
                try:
                    best_val, _ = train_one(model_type, train_size, seed, verbose=True)
                    key = f"{model_type}_{size_label(train_size)}_seed{seed}"
                    results[key] = best_val
                except Exception as e:
                    print(f"  ERROR training {model_type} {train_size} seed{seed}: {e}")
                    import traceback
                    traceback.print_exc()

    if results:
        print("\nTraining summary:")
        for k, v in sorted(results.items()):
            print(f"  {k:<35s}  best_val_mse={v:.4f}")


def step_evaluate(eval_size):
    section("Step 6: Evaluate and generate plots")
    from models import EMLP_AVAILABLE
    if not EMLP_AVAILABLE:
        print("emlp not installed — skipping evaluation.")
        return

    from evaluate import run_full_evaluation
    try:
        run_full_evaluation(train_size_for_detail=eval_size)
    except Exception as e:
        print(f"Evaluation error: {e}")
        import traceback
        traceback.print_exc()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 60)
    print("  EMLP vs MLP — 2×2×2 Pocket Cube Experiment")
    print("=" * 60)

    if args.quick:
        print("\n[QUICK MODE] Using reduced dataset sizes and epochs.")

    t_start = time.time()

    step_verify_env()
    step_verify_group()

    if not args.skip_data and not args.skip_train:
        dist_table, move_table = step_bfs(args.skip_bfs)
    else:
        dist_table, move_table = None, None

    if dist_table is None and not args.skip_data and not args.skip_train:
        # Need to load BFS tables even if skipping BFS generation
        from dataset import load_bfs_tables, bfs_tables_exist
        if bfs_tables_exist():
            dist_table, move_table = load_bfs_tables()

    train_sizes = step_generate_data(
        dist_table, args.skip_data, args.quick)

    step_train(args.skip_train, args.model, args.quick, train_sizes)

    step_evaluate(args.eval_size)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
