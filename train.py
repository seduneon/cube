"""
Training loop for EMLP and MLP value networks on the 2x2x2 Pocket Cube.
PyTorch backend — works on Kaggle without JAX version conflicts.

Usage:
    python train.py                          # train all 18 configs
    python train.py --model emlp --size 50k --seed 0  # single run
    python train.py --model mlp  --size 200k --seed 1
"""

import os
import csv
import time
import argparse
import numpy as np
import torch

from models import (
    build_emlp_model, build_mlp_model, cosine_decay_lr,
    save_model, load_model, get_param_count, EMLP_AVAILABLE, DEVICE,
)
from dataset import load_dataset, DATA_DIR

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
CKPT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
LOG_DIR = os.path.join(RESULTS_DIR, "logs")

# ─── Hyperparameters ──────────────────────────────────────────────────────────
BATCH_SIZE = 256
MAX_EPOCHS = 200
EARLY_STOP_PATIENCE = 20
LR_INIT = 3e-4
LR_MIN = 1e-5
CH = 384
NUM_LAYERS = 3

TRAIN_SIZES = [50_000, 200_000, 1_000_000]
SEEDS = [0, 1, 2]


def size_label(n):
    if n >= 1_000:
        return f"{n // 1000}k"
    return str(n)


def load_split(split, n_train=None):
    return load_dataset(split, n_train)


def make_batches(X, y, batch_size, rng):
    """Yield shuffled (x_batch, y_batch) numpy arrays."""
    n = len(X)
    idx = rng.permutation(n)
    for start in range(0, n, batch_size):
        batch_idx = idx[start:start + batch_size]
        yield X[batch_idx], y[batch_idx]


# ─── Training ─────────────────────────────────────────────────────────────────

def train_one(model_type, train_size, seed, verbose=True):
    label = f"{model_type}_{size_label(train_size)}_seed{seed}"
    ckpt_path = os.path.join(CKPT_DIR, f"{label}.pt")
    log_path = os.path.join(LOG_DIR, f"{label}.csv")
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Training: {label}")
        print(f"Device:   {DEVICE}")
        print(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── Data ─────────────────────────────────────────────────────────────────
    X_train, y_train = load_split("train", train_size)
    X_val, y_val = load_split("val")
    y_train = y_train.astype(np.float32)
    y_val = y_val.astype(np.float32)

    if verbose:
        print(f"Train: {X_train.shape}, Val: {X_val.shape}")

    # ── Model ────────────────────────────────────────────────────────────────
    if model_type == "emlp":
        model, G, _, _ = build_emlp_model(ch=CH, num_layers=NUM_LAYERS)
    else:
        model, G, _, _ = build_mlp_model(ch=CH, num_layers=NUM_LAYERS)

    model = model.to(DEVICE)
    n_params = get_param_count(model)
    if verbose:
        print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR_INIT)

    # Pre-place validation data on device
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=DEVICE)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=DEVICE)

    # ── Training loop ────────────────────────────────────────────────────────
    total_steps = MAX_EPOCHS * (len(X_train) // BATCH_SIZE)
    step = 0
    best_val_mse = float("inf")
    patience_counter = 0
    rng = np.random.RandomState(seed + 1000)
    log_rows = []

    for epoch in range(MAX_EPOCHS):
        t0 = time.time()
        model.train()
        train_losses = []

        for x_batch, y_batch in make_batches(X_train, y_train, BATCH_SIZE, rng):
            lr = cosine_decay_lr(step, total_steps, LR_INIT, LR_MIN)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            x_t = torch.tensor(x_batch, dtype=torch.float32, device=DEVICE)
            y_t = torch.tensor(y_batch, dtype=torch.float32, device=DEVICE)

            optimizer.zero_grad()
            pred = model(x_t).squeeze()
            loss = ((pred - y_t) ** 2).mean()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            step += 1

        # ── Validation ───────────────────────────────────────────────────────
        model.eval()
        val_losses = []
        with torch.no_grad():
            for start in range(0, len(X_val), BATCH_SIZE * 4):
                end = min(start + BATCH_SIZE * 4, len(X_val))
                pred_v = model(X_val_t[start:end]).squeeze()
                vl = ((pred_v - y_val_t[start:end]) ** 2).mean()
                val_losses.append(vl.item())

        train_mse = float(np.mean(train_losses))
        val_mse = float(np.mean(val_losses))
        wall_time = time.time() - t0
        lr_now = cosine_decay_lr(step, total_steps, LR_INIT, LR_MIN)

        log_rows.append({
            "epoch": epoch,
            "train_mse": train_mse,
            "val_mse": val_mse,
            "lr": lr_now,
            "wall_time": wall_time,
        })

        if verbose:
            print(f"Epoch {epoch+1:3d} | train_mse={train_mse:.4f} "
                  f"val_mse={val_mse:.4f} lr={lr_now:.2e} t={wall_time:.1f}s")

        # ── Early stopping + checkpointing ────────────────────────────────────
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            patience_counter = 0
            save_model(model, ckpt_path)
            if verbose:
                print(f"  → New best val_mse={best_val_mse:.4f}, checkpoint saved.")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1} (patience={EARLY_STOP_PATIENCE})")
                break

    # ── Save logs ─────────────────────────────────────────────────────────────
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)

    if verbose:
        print(f"\nBest val MSE: {best_val_mse:.4f}")
        print(f"Checkpoint: {ckpt_path}")
        print(f"Log: {log_path}")

    return best_val_mse, log_rows


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not EMLP_AVAILABLE:
        print("ERROR: emlp not installed. Run: pip install emlp")
        return

    parser = argparse.ArgumentParser(description="Train EMLP/MLP on 2x2x2 cube.")
    parser.add_argument("--model", choices=["emlp", "mlp", "all"], default="all")
    parser.add_argument("--size", choices=["50k", "200k", "1m", "all"], default="all")
    parser.add_argument("--seed", type=int, choices=[0, 1, 2, -1], default=-1,
                        help="-1 means all seeds")
    args = parser.parse_args()

    model_types = ["emlp", "mlp"] if args.model == "all" else [args.model]
    size_map = {"50k": 50_000, "200k": 200_000, "1m": 1_000_000}
    train_sizes = TRAIN_SIZES if args.size == "all" else [size_map[args.size]]
    seeds = SEEDS if args.seed == -1 else [args.seed]

    results = {}
    for model_type in model_types:
        for train_size in train_sizes:
            for seed in seeds:
                config_key = f"{model_type}_{size_label(train_size)}_seed{seed}"
                best_val_mse, _ = train_one(model_type, train_size, seed, verbose=True)
                results[config_key] = best_val_mse

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    for config, val_mse in sorted(results.items()):
        print(f"  {config:<30s}  val_mse={val_mse:.4f}")


if __name__ == "__main__":
    main()
