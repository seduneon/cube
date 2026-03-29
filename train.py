"""
Training loop for cube value networks.
PyTorch backend — works on Kaggle without JAX version conflicts.

Usage:
    python train.py                                         # train all configs
    python train.py --model emlp    --size 200k --seed 0
    python train.py --model emlp_col --size 200k --seed 0
    python train.py --model mlp     --size 200k --seed 0
    python train.py --model mlp_aug --size 200k --seed 0
    python train.py --model all     --size all  --seed -1  # 18+ runs
"""

import os
import csv
import time
import argparse
import numpy as np
import torch

from models import MODEL_REGISTRY, build_model, cosine_decay_lr, save_model, get_param_count, DEVICE
from cube_symmetry import ALL_COLOR_PERMS
from dataset import load_dataset, DATA_DIR

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
CKPT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
LOG_DIR = os.path.join(RESULTS_DIR, "logs")

# ─── Hyperparameters ──────────────────────────────────────────────────────────
BATCH_SIZE = 1024
MAX_EPOCHS = 200
EARLY_STOP_PATIENCE = 20
LR_INIT = 3e-4
LR_MIN = 1e-5
NUM_LAYERS = 3

# Per-model width hyperparameter.
#   emlp:     c_hidden=84  → ~351K params
#   emlp_col: k_hidden=14  → 14 blocks × 6 = 84 effective channels, ~16K params
#   mlp:      hidden_dim=384 → ~351K params (matched to emlp for fair comparison)
#   mlp_aug:  same as mlp
MODEL_WIDTHS = {
    "emlp":     84,
    "emlp_col": 14,
    "mlp":      384,
    "mlp_aug":  384,
}

TRAIN_SIZES = [50_000, 200_000, 1_000_000]
SEEDS = [0, 1, 2]

ALL_MODEL_TYPES = list(MODEL_REGISTRY.keys())  # ["emlp", "emlp_col", "mlp", "mlp_aug"]


def size_label(n):
    if n >= 1_000:
        return f"{n // 1000}k"
    return str(n)


def load_split(split, n_train=None):
    stratified = (split == "train" and n_train is not None)
    return load_dataset(split, n_train, stratified=stratified)


def make_batches(X, y, batch_size, rng):
    """Yield shuffled (x_batch, y_batch) numpy arrays."""
    n = len(X)
    idx = rng.permutation(n)
    for start in range(0, n, batch_size):
        batch_idx = idx[start:start + batch_size]
        yield X[batch_idx], y[batch_idx]


# ─── Color augmentation ───────────────────────────────────────────────────────

def apply_color_aug(x_batch: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Apply a random S6 color permutation to each sample in the batch.

    x_batch: (N, 144) float32 one-hot
    Returns: (N, 144) float32 with color channels permuted (copy).
    """
    perm = ALL_COLOR_PERMS[rng.randint(0, len(ALL_COLOR_PERMS))]
    return x_batch.reshape(-1, 24, 6)[:, :, perm].reshape(-1, 144).copy()


# ─── Training ─────────────────────────────────────────────────────────────────

def train_one(model_type, train_size, seed, verbose=True):
    label = f"{model_type}_{size_label(train_size)}_seed{seed}"
    ckpt_path = os.path.join(CKPT_DIR, f"{label}.pt")
    log_path = os.path.join(LOG_DIR, f"{label}.csv")
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    if os.path.exists(ckpt_path):
        if verbose:
            print(f"Checkpoint already exists, skipping: {ckpt_path}")
        return None, None

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

    # ── Model (via registry) ─────────────────────────────────────────────────
    spec = MODEL_REGISTRY[model_type]
    width = MODEL_WIDTHS[model_type]
    model = build_model(model_type, width, NUM_LAYERS).to(DEVICE)
    n_params = get_param_count(model)

    if verbose:
        print(f"Model:      {spec.label}")
        print(f"Parameters: {n_params:,}")
        if spec.color_augment:
            print("            [color augmentation enabled]")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR_INIT)

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

            # Apply color augmentation if the model spec requests it
            if spec.color_augment:
                x_batch = apply_color_aug(x_batch, rng)

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
                    print(f"Early stopping at epoch {epoch+1}")
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
    parser = argparse.ArgumentParser(description="Train cube value networks.")
    parser.add_argument("--model",
                        choices=ALL_MODEL_TYPES + ["all"],
                        default="all",
                        help="Model type(s) to train")
    parser.add_argument("--size", choices=["50k", "200k", "1m", "all"], default="all")
    parser.add_argument("--seed", type=int, choices=[0, 1, 2, -1], default=-1,
                        help="-1 means all seeds")
    args = parser.parse_args()

    model_types = ALL_MODEL_TYPES if args.model == "all" else [args.model]
    size_map = {"50k": 50_000, "200k": 200_000, "1m": 1_000_000}
    train_sizes = TRAIN_SIZES if args.size == "all" else [size_map[args.size]]
    seeds = SEEDS if args.seed == -1 else [args.seed]

    results = {}
    for model_type in model_types:
        for train_size in train_sizes:
            for seed in seeds:
                config_key = f"{model_type}_{size_label(train_size)}_seed{seed}"
                best_val_mse, _ = train_one(model_type, train_size, seed, verbose=True)
                if best_val_mse is not None:
                    results[config_key] = best_val_mse

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    for config, val_mse in sorted(results.items()):
        print(f"  {config:<38s}  val_mse={val_mse:.4f}")


if __name__ == "__main__":
    main()
