"""
Evaluation, metrics, and plots for the EMLP vs MLP cube experiment.

Metrics:
  1. Equivariance error  — |f(g·x) - f(x)| for invariant value head
  2. Value prediction    — MAE, per-depth MAE, correlation, rounded accuracy
  3. Greedy solve rate   — success rate solving random scrambles
  4. Data efficiency     — val MSE vs training set size
  5. Parameter efficiency — test MAE per parameter

Plots (saved to results/plots/):
  - learning_curves.png
  - data_efficiency.png
  - per_depth_accuracy.png
  - equivariance_error.png
  - solve_rate_table.txt
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from collections import defaultdict

from cube_env import (
    SOLVED_STATE, MOVES, MOVE_NAMES, INVERSE_MOVE, NUM_MOVES,
    apply_move, encode_state, is_solved, state_to_tuple, tuple_to_state, scramble,
)
from cube_group import enumerate_group_elements, X_ROT_PERM, Y_ROT_PERM, apply_rotation
from models import (
    build_emlp_model, build_mlp_model, load_model,
    get_param_count, EMLP_AVAILABLE,
)
from dataset import load_dataset, DATA_DIR, load_bfs_tables, bfs_tables_exist
from train import CKPT_DIR, LOG_DIR, TRAIN_SIZES, SIZE_LABELS, SEEDS

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
PLOT_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

MAX_DISTANCE = 14  # QTM with R,R',U,U',F,F' quarter turns


# ─── Load model checkpoint ────────────────────────────────────────────────────

def load_checkpoint(model_type, train_size, seed):
    """Load a model from its best checkpoint."""
    label = f"{model_type}_{SIZE_LABELS[train_size]}_seed{seed}"
    ckpt_path = os.path.join(CKPT_DIR, f"{label}.npz")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if model_type == "emlp":
        model, G, _, _ = build_emlp_model()
    else:
        model, G, _, _ = build_mlp_model()

    load_model(model, ckpt_path)
    return model


def model_predict(model, X_np):
    """Run forward pass on numpy array X, return numpy predictions."""
    import jax.numpy as jnp
    import objax

    @objax.Function.with_vars(model.vars())
    def predict_fn(x):
        return model(x, training=False).squeeze()

    predict_fn = objax.Jit(predict_fn, model.vars())

    batch_size = 512
    preds = []
    for start in range(0, len(X_np), batch_size):
        x_batch = jnp.array(X_np[start:start + batch_size])
        preds.append(np.array(predict_fn(x_batch)))
    return np.concatenate(preds)


# ─── Metric 1: Equivariance Error ─────────────────────────────────────────────

def equivariance_error(model, X_test, n_rotations=10, n_states=500):
    """Compute mean |f(g·x) - f(x)| over random states and rotations.

    For a truly invariant model, this should be ~1e-7 (float32 precision).
    """
    import jax.numpy as jnp
    import objax

    @objax.Function.with_vars(model.vars())
    def predict_fn(x):
        return model(x, training=False).squeeze()

    predict_fn = objax.Jit(predict_fn, model.vars())

    all_rotations = enumerate_group_elements(X_ROT_PERM, Y_ROT_PERM)

    # Sample random states from test set
    idx = np.random.choice(len(X_test), size=n_states, replace=False)
    X_sample = X_test[idx]  # (n_states, 144)

    # Decode states (reverse one-hot)
    # X has shape (n_states, 144), each row is a 24x6 one-hot flattened
    states_sample = X_sample.reshape(n_states, 24, 6).argmax(axis=2)  # (n_states, 24)

    # Sample n_rotations random rotations
    rot_indices = np.random.choice(len(all_rotations), size=n_rotations, replace=False)
    rotations = [all_rotations[i] for i in rot_indices]

    errors = []
    f_original = np.array(predict_fn(jnp.array(X_sample)))  # (n_states,)

    for rot_perm in rotations:
        # Apply rotation to each state
        rotated_states = np.array([apply_rotation(rot_perm, s) for s in states_sample],
                                  dtype=np.int8)
        # Re-encode
        X_rotated = np.array([encode_state(s) for s in rotated_states], dtype=np.float32)
        f_rotated = np.array(predict_fn(jnp.array(X_rotated)))  # (n_states,)
        errors.append(np.abs(f_rotated - f_original))

    mean_err = float(np.mean(errors))
    return mean_err, np.array(errors)


# ─── Metric 2: Value Prediction Accuracy ─────────────────────────────────────

def value_prediction_metrics(model, X_test, y_test):
    """Compute value prediction metrics on the test set.

    Returns dict with: mae, per_depth_mae, correlation, rounded_accuracy
    """
    preds = model_predict(model, X_test)
    y_true = y_test.astype(np.float32)

    mae = float(np.mean(np.abs(preds - y_true)))
    rounded_acc = float(np.mean(np.round(preds) == y_true))

    # Pearson correlation
    corr = float(np.corrcoef(preds, y_true)[0, 1])

    # Per-depth MAE
    per_depth_mae = {}
    for d in range(MAX_DISTANCE + 1):
        mask = y_test == d
        if mask.sum() > 0:
            per_depth_mae[d] = float(np.mean(np.abs(preds[mask] - y_true[mask])))
        else:
            per_depth_mae[d] = None

    return {
        "mae": mae,
        "rounded_accuracy": rounded_acc,
        "correlation": corr,
        "per_depth_mae": per_depth_mae,
    }


# ─── Metric 3: Greedy Solve Rate ──────────────────────────────────────────────

def greedy_solve(model, initial_state, max_steps=50):
    """Greedily solve the cube by picking the move with lowest predicted distance.

    Returns (solved: bool, n_moves: int, move_sequence: list)
    """
    import jax.numpy as jnp
    import objax

    @objax.Function.with_vars(model.vars())
    def predict_fn(x):
        return model(x, training=False).squeeze()

    predict_fn = objax.Jit(predict_fn, model.vars())

    state = initial_state.copy()
    move_sequence = []

    for step_i in range(max_steps):
        if is_solved(state):
            return True, step_i, move_sequence

        # Evaluate all 6 moves
        next_states = np.array([
            encode_state(apply_move(state, MOVES[m])) for m in range(NUM_MOVES)
        ], dtype=np.float32)

        preds = np.array(predict_fn(jnp.array(next_states)))  # (6,)
        best_move = int(np.argmin(preds))

        state = apply_move(state, MOVES[best_move])
        move_sequence.append(best_move)

    return is_solved(state), len(move_sequence), move_sequence


def evaluate_solve_rate(model, n_trials=1000, scramble_depths=None, verbose=True):
    """Evaluate greedy solve rate at various scramble depths.

    Returns dict: depth -> {solve_rate, mean_moves, mean_excess}
    """
    if scramble_depths is None:
        scramble_depths = [4, 7, 10, 14]

    results = {}
    rng = np.random.RandomState(42)

    for depth in scramble_depths:
        solved_count = 0
        move_lengths = []

        for _ in range(n_trials):
            state, _ = scramble(SOLVED_STATE, depth,
                                rng=type('R', (), {'choice': lambda self, x: rng.choice(x)})())
            # Use numpy for scramble
            s = SOLVED_STATE.copy()
            last = -1
            for _k in range(depth):
                cands = [m for m in range(NUM_MOVES) if m != INVERSE_MOVE[last]] \
                    if last >= 0 else list(range(NUM_MOVES))
                m = rng.choice(cands)
                s = apply_move(s, MOVES[m])
                last = m

            solved, n_moves, _ = greedy_solve(model, s)
            if solved:
                solved_count += 1
                move_lengths.append(n_moves)

        solve_rate = solved_count / n_trials
        mean_moves = float(np.mean(move_lengths)) if move_lengths else float("nan")
        mean_excess = mean_moves - depth if move_lengths else float("nan")

        results[depth] = {
            "solve_rate": solve_rate,
            "mean_moves": mean_moves,
            "mean_excess": mean_excess,
        }

        if verbose:
            print(f"  depth={depth:2d}: solve_rate={solve_rate:.1%}  "
                  f"mean_moves={mean_moves:.1f}  excess={mean_excess:+.1f}")

    return results


# ─── Load logs ────────────────────────────────────────────────────────────────

def load_log(model_type, train_size, seed):
    label = f"{model_type}_{SIZE_LABELS[train_size]}_seed{seed}"
    log_path = os.path.join(LOG_DIR, f"{label}.csv")
    if not os.path.exists(log_path):
        return None
    rows = []
    with open(log_path) as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})
    return rows


# ─── Plots ────────────────────────────────────────────────────────────────────

COLORS = {"emlp": "#2196F3", "mlp": "#FF5722"}
SIZE_COLORS = {50_000: "#E91E63", 200_000: "#9C27B0", 1_000_000: "#3F51B5"}


def plot_learning_curves(train_size=200_000):
    """Plot train/val MSE vs epoch for EMLP and MLP (mean ± std over 3 seeds)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Learning Curves (train size = {SIZE_LABELS[train_size]})", fontsize=14)

    for ax_idx, metric in enumerate(["train_mse", "val_mse"]):
        ax = axes[ax_idx]
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE")

        for model_type in ["emlp", "mlp"]:
            all_curves = []
            for seed in SEEDS:
                log = load_log(model_type, train_size, seed)
                if log is not None:
                    all_curves.append([row[metric] for row in log])

            if not all_curves:
                continue

            max_len = max(len(c) for c in all_curves)
            # Pad shorter runs with last value
            padded = np.array([
                c + [c[-1]] * (max_len - len(c)) for c in all_curves
            ])
            mean = padded.mean(axis=0)
            std = padded.std(axis=0)
            epochs = np.arange(1, max_len + 1)

            ax.plot(epochs, mean, label=model_type.upper(), color=COLORS[model_type])
            ax.fill_between(epochs, mean - std, mean + std,
                            alpha=0.2, color=COLORS[model_type])

        ax.legend()
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "learning_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_data_efficiency(all_val_mses):
    """Plot val MSE vs dataset size for EMLP and MLP.

    all_val_mses: dict[(model_type, train_size, seed)] -> val_mse
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("Data Efficiency: Val MSE vs Training Set Size")
    ax.set_xlabel("Training set size")
    ax.set_ylabel("Val MSE")
    ax.set_xscale("log")
    ax.set_yscale("log")

    for model_type in ["emlp", "mlp"]:
        means, stds, sizes = [], [], []
        for train_size in TRAIN_SIZES:
            mses = [all_val_mses.get((model_type, train_size, s)) for s in SEEDS]
            mses = [m for m in mses if m is not None]
            if mses:
                means.append(np.mean(mses))
                stds.append(np.std(mses))
                sizes.append(train_size)

        if means:
            ax.errorbar(sizes, means, yerr=stds, marker="o",
                        label=model_type.upper(), color=COLORS[model_type],
                        capsize=5, linewidth=2, markersize=8)

    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "data_efficiency.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_per_depth_accuracy(metrics_emlp, metrics_mlp):
    """Bar chart: per-depth MAE for EMLP vs MLP."""
    depths = list(range(MAX_DISTANCE + 1))
    emlp_mae = [metrics_emlp["per_depth_mae"].get(d) or 0 for d in depths]
    mlp_mae = [metrics_mlp["per_depth_mae"].get(d) or 0 for d in depths]

    x = np.arange(len(depths))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width/2, emlp_mae, width, label="EMLP", color=COLORS["emlp"], alpha=0.8)
    ax.bar(x + width/2, mlp_mae, width, label="MLP", color=COLORS["mlp"], alpha=0.8)

    ax.set_title("Per-Depth MAE: EMLP vs MLP")
    ax.set_xlabel("Optimal Distance (BFS depth)")
    ax.set_ylabel("Mean Absolute Error")
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in depths])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "per_depth_accuracy.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_equivariance_error(eq_errors):
    """Bar chart: equivariance error for EMLP vs MLP (log scale)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    models = ["EMLP", "MLP"]
    errors = [eq_errors.get("emlp", 0), eq_errors.get("mlp", 0)]
    colors = [COLORS["emlp"], COLORS["mlp"]]

    bars = ax.bar(models, errors, color=colors, alpha=0.8, width=0.4)
    ax.set_yscale("log")
    ax.set_ylabel("Mean |f(g·x) - f(x)|")
    ax.set_title("Equivariance Error\n(lower is better; EMLP should be ~1e-7)")
    ax.grid(True, alpha=0.3, axis="y")

    for bar, err in zip(bars, errors):
        ax.text(bar.get_x() + bar.get_width()/2, err * 1.5,
                f"{err:.2e}", ha="center", va="bottom", fontsize=11)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "equivariance_error.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def save_solve_rate_table(solve_results, path=None):
    """Save solve rate results to a text table."""
    if path is None:
        path = os.path.join(PLOT_DIR, "solve_rate_table.txt")

    lines = [
        "Greedy Solve Rate",
        "=" * 70,
        f"{'Depth':>6}  {'Model':>6}  {'Solve Rate':>12}  {'Mean Moves':>12}  {'Excess':>10}",
        "-" * 70,
    ]

    for depth in sorted(solve_results.keys()):
        for model_type in ["emlp", "mlp"]:
            if model_type in solve_results[depth]:
                r = solve_results[depth][model_type]
                lines.append(
                    f"{depth:>6}  {model_type.upper():>6}  "
                    f"{r['solve_rate']:>11.1%}  "
                    f"{r['mean_moves']:>12.1f}  "
                    f"{r['mean_excess']:>+10.1f}"
                )

    text = "\n".join(lines)
    with open(path, "w") as f:
        f.write(text)
    print(text)
    print(f"\nSaved {path}")


# ─── Full evaluation run ──────────────────────────────────────────────────────

def run_full_evaluation(train_size_for_detail=200_000):
    """Run all evaluation metrics and generate all plots."""
    if not EMLP_AVAILABLE:
        print("emlp not available. Cannot run evaluation.")
        return

    X_test, y_test, _ = load_dataset("test")
    print(f"Test set: {X_test.shape}")

    all_val_mses = {}
    eq_errors = {}
    val_metrics = {}
    solve_results = defaultdict(dict)

    for model_type in ["emlp", "mlp"]:
        for train_size in TRAIN_SIZES:
            for seed in SEEDS:
                try:
                    model = load_checkpoint(model_type, train_size, seed)
                except FileNotFoundError as e:
                    print(f"  Skip (no checkpoint): {e}")
                    continue

                val_mse_list = []
                for s in SEEDS:
                    try:
                        m = load_checkpoint(model_type, train_size, s)
                        X_val, y_val, _ = load_dataset("val")
                        preds_val = model_predict(m, X_val)
                        val_mse = float(np.mean((preds_val - y_val.astype(np.float32))**2))
                        val_mse_list.append(val_mse)
                        all_val_mses[(model_type, train_size, s)] = val_mse
                    except Exception:
                        pass
                break  # Only need one model per (type, size) for detailed metrics

            # Detailed metrics using 200K model, seed 0
            if train_size == train_size_for_detail:
                try:
                    model = load_checkpoint(model_type, train_size, 0)

                    print(f"\n── {model_type.upper()} ({SIZE_LABELS[train_size]}) ──")

                    print("  Equivariance error...")
                    eq_err, _ = equivariance_error(model, X_test)
                    eq_errors[model_type] = eq_err
                    print(f"  Equivariance error: {eq_err:.2e}")

                    print("  Value prediction metrics...")
                    metrics = value_prediction_metrics(model, X_test, y_test)
                    val_metrics[model_type] = metrics
                    print(f"  MAE: {metrics['mae']:.4f}")
                    print(f"  Rounded accuracy: {metrics['rounded_accuracy']:.1%}")
                    print(f"  Pearson r: {metrics['correlation']:.4f}")

                    print("  Greedy solve rate...")
                    solve_res = evaluate_solve_rate(model, n_trials=500, verbose=True)
                    for depth, res in solve_res.items():
                        solve_results[depth][model_type] = res

                except FileNotFoundError as e:
                    print(f"  Skip: {e}")

    # ── Generate plots ────────────────────────────────────────────────────────
    print("\nGenerating plots...")

    plot_learning_curves(train_size=train_size_for_detail)

    if all_val_mses:
        plot_data_efficiency(all_val_mses)

    if "emlp" in val_metrics and "mlp" in val_metrics:
        plot_per_depth_accuracy(val_metrics["emlp"], val_metrics["mlp"])

    if "emlp" in eq_errors and "mlp" in eq_errors:
        plot_equivariance_error(eq_errors)

    if solve_results:
        save_solve_rate_table(solve_results)

    # ── Parameter efficiency ──────────────────────────────────────────────────
    print("\nParameter efficiency:")
    for model_type in ["emlp", "mlp"]:
        try:
            model = load_checkpoint(model_type, train_size_for_detail, 0)
            n_params = get_param_count(model)
            mae = val_metrics.get(model_type, {}).get("mae", float("nan"))
            if n_params > 0:
                efficiency = mae / n_params * 1e6
                print(f"  {model_type.upper()}: {n_params:,} params, "
                      f"MAE={mae:.4f}, MAE per 1M params={efficiency:.4f}")
        except Exception:
            pass

    print("\nEvaluation complete. Plots saved to:", PLOT_DIR)


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=int, default=200_000,
                        choices=[50_000, 200_000, 1_000_000],
                        help="Which training size to use for detailed metrics")
    args = parser.parse_args()
    run_full_evaluation(train_size_for_detail=args.train_size)
