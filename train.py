"""
Training loop for cube value networks.
PyTorch backend — works on Kaggle without JAX version conflicts.

Usage:
    python train.py                                         # train all configs
    python train.py --model emlp_rot  --size 200k --seed 42
    python train.py --model emlp_both --size 200k --seed 42
    python train.py --model emlp_col  --size 200k --seed 42
    python train.py --model mlp     --size 200k --seed 42
    python train.py --model mlp_aug --size 200k --seed 42
    python train.py --model all     --size all  --seed -1  # all runs
"""

import os
import csv
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from models import MODEL_REGISTRY, build_model, get_param_count, save_checkpoint, DEVICE
from cube_symmetry import ALL_ROTATIONS, ALL_COLOR_PERMS
from dataset import load_dataset, DATA_DIR, MAX_DISTANCE

# Precomputed array form of rotations for vectorised augmentation
_ALL_ROTATIONS_ARR = np.stack(ALL_ROTATIONS).astype(np.int64)  # (24, 24)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
CKPT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
LOG_DIR = os.path.join(RESULTS_DIR, "logs")

# ─── Hyperparameters ──────────────────────────────────────────────────────────
BATCH_SIZE = 512
MAX_EPOCHS = 200
EARLY_STOP_PATIENCE = 15
LR_INIT = 1e-3
LR_MIN = 1e-5
WEIGHT_DECAY = 1e-4
HUBER_DELTA = 0.1   # in normalised-distance units ([0, 1] range)
NUM_LAYERS = 3

# Per-model width.  emlp_rot/mlp widths give comparable hidden features;
# emlp_both/emlp_col k=20 → 120 channels/position = 2880 total features.
MODEL_WIDTHS = {
    "emlp_rot":     28,  # c_hidden = 28  (spatial only, ~43K params)
    "emlp_both":    20,  # k_hidden = 20  → ~40K params
    "emlp_col":     60,  # h_hidden = 60  → ~38K params  (color-matching MLP, S6-invariant)
    "mlp":         256,  # hidden_dim = 256, ~169K params  (large unconstrained baseline)
    "mlp_aug":     110,  # hidden_dim = 110, ~40K params   (matched to equivariant)
    "mlp_matched": 110,  # hidden_dim = 110, ~40K params   (parameter-matched to emlp_rot)
}

TRAIN_SIZES = [50_000, 200_000, 1_000_000]
SEEDS = [42, 43, 44]

ALL_MODEL_TYPES = list(MODEL_REGISTRY.keys())


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


# ─── Data augmentation ────────────────────────────────────────────────────────

def apply_augmentation(x_batch: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Apply an independent random (spatial rotation, S6 color perm) to each sample.

    Draws a fresh transform per sample rather than one per batch, giving the
    full 17,280-fold diversity within every mini-batch.

    x_batch : (N, 144) float32 one-hot
    Returns  : (N, 144) float32 copy with both symmetries applied.
    """
    N = x_batch.shape[0]

    rot_idx   = rng.randint(0, 24,  size=N)                         # (N,)
    color_idx = rng.randint(0, 720, size=N)                         # (N,)

    rots = _ALL_ROTATIONS_ARR[rot_idx]                              # (N, 24) forward perms
    cols = ALL_COLOR_PERMS[color_idx].astype(np.int64)              # (N, 6)

    rots_inv = np.argsort(rots, axis=1)                             # (N, 24) gather indices

    x = x_batch.reshape(N, 24, 6)

    # Apply per-sample spatial rotation via fancy indexing (always copies)
    x = x[np.arange(N)[:, None], rots_inv, :]                      # (N, 24, 6)

    # Apply per-sample color permutation
    x = x[np.arange(N)[:, None, None],
           np.arange(24)[None, :, None],
           cols[:, None, :]]                                         # (N, 24, 6)

    return x.reshape(N, 144).copy()


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

    # Normalize distances to [0, 1]
    y_train = y_train.astype(np.float32) / MAX_DISTANCE
    y_val   = y_val.astype(np.float32)   / MAX_DISTANCE

    if verbose:
        print(f"Train: {X_train.shape}, Val: {X_val.shape}")
        print(f"Target range: [{y_train.min():.3f}, {y_train.max():.3f}]")

    # ── Model (via registry) ─────────────────────────────────────────────────
    spec = MODEL_REGISTRY[model_type]
    width = MODEL_WIDTHS[model_type]
    model = build_model(model_type, width, NUM_LAYERS).to(DEVICE)
    n_params = get_param_count(model)

    if verbose:
        print(f"Model:      {spec.label}")
        print(f"Parameters: {n_params:,}")
        if spec.color_augment:
            print("            [spatial + color augmentation enabled]")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_INIT, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=LR_MIN)

    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=DEVICE)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=DEVICE)

    # ── Training loop ────────────────────────────────────────────────────────
    best_val_mse = float("inf")
    best_epoch = 0
    patience_counter = 0
    rng = np.random.RandomState(seed + 1000)
    log_rows = []

    for epoch in range(MAX_EPOCHS):
        t0 = time.time()
        model.train()
        train_losses = []

        for x_batch, y_batch in make_batches(X_train, y_train, BATCH_SIZE, rng):
            if spec.color_augment:
                x_batch = apply_augmentation(x_batch, rng)

            x_t = torch.tensor(x_batch, dtype=torch.float32, device=DEVICE)
            y_t = torch.tensor(y_batch, dtype=torch.float32, device=DEVICE)

            optimizer.zero_grad()
            pred = model(x_t).squeeze()
            loss = F.huber_loss(pred, y_t, delta=HUBER_DELTA)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]

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
        val_mse   = float(np.mean(val_losses))
        wall_time = time.time() - t0

        log_rows.append({
            "epoch": epoch,
            "train_mse": train_mse,
            "val_mse": val_mse,
            "lr": lr_now,
            "epoch_seconds": wall_time,
        })

        if verbose:
            print(f"Epoch {epoch+1:3d} | train_mse={train_mse:.5f} "
                  f"val_mse={val_mse:.5f} lr={lr_now:.2e} t={wall_time:.1f}s")

        # ── Early stopping + checkpointing ────────────────────────────────────
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(model, ckpt_path,
                            config={
                                'model_type': model_type,
                                'train_size': train_size,
                                'seed': seed,
                                'width': width,
                                'num_layers': NUM_LAYERS,
                                'max_distance': MAX_DISTANCE,
                            },
                            best_val_mse=best_val_mse,
                            best_epoch=best_epoch)
            if verbose:
                print(f"  → New best val_mse={best_val_mse:.5f}, checkpoint saved.")
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
        print(f"\nBest val MSE (normalized): {best_val_mse:.5f}")
        print(f"Best epoch: {best_epoch+1}")
        print(f"Checkpoint: {ckpt_path}")
        print(f"Log: {log_path}")

    return best_val_mse, log_rows


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train cube value networks.")
    parser.add_argument("--model", choices=ALL_MODEL_TYPES + ["all"], default="all")
    parser.add_argument("--size", choices=["50k", "200k", "1m", "all"], default="all")
    parser.add_argument("--seed", type=int, default=-1,
                        help="Specific seed (42/43/44), or -1 for all")
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
    print("TRAINING SUMMARY (val MSE on normalized targets)")
    print("=" * 60)
    for config, val_mse in sorted(results.items()):
        print(f"  {config:<42s}  val_mse={val_mse:.5f}")


if __name__ == "__main__":
    main()
