"""
Evaluation, metrics, and plots for the EMLP vs MLP cube experiment.
PyTorch backend.
"""

import os
import csv
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

from cube_env import (
    SOLVED_STATE, MOVES, MOVE_NAMES, INVERSE_MOVE, NUM_MOVES,
    apply_move, encode_state, is_solved, state_to_tuple, tuple_to_state, scramble,
)
from cube_group import ALL_ROTATIONS, apply_rotation
from models import (
    build_emlp_model, build_mlp_model, load_model,
    get_param_count, EMLP_AVAILABLE, DEVICE,
)
from dataset import load_dataset, DATA_DIR, load_bfs_tables, bfs_tables_exist, _CANDIDATES
from train import CKPT_DIR, LOG_DIR, TRAIN_SIZES, SEEDS, size_label

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
PLOT_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

MAX_DISTANCE = 14


# ─── Load model checkpoint ────────────────────────────────────────────────────

def load_checkpoint(model_type, train_size, seed):
    label = f"{model_type}_{size_label(train_size)}_seed{seed}"
    ckpt_path = os.path.join(CKPT_DIR, f"{label}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model = load_model(model_type, ckpt_path)
    model = model.to(DEVICE)
    model.eval()
    return model


def make_predict_fn(model):
    """Return a predict function: numpy (N,144) -> numpy (N,) predictions."""
    def _predict(X_np):
        with torch.no_grad():
            x = torch.tensor(X_np, dtype=torch.float32, device=DEVICE)
            return model(x).squeeze().cpu().numpy()
    return _predict


def model_predict(predict_fn, X_np, batch_size=512):
    preds = []
    for start in range(0, len(X_np), batch_size):
        preds.append(predict_fn(X_np[start:start + batch_size]))
    return np.concatenate(preds)


# ─── Metric 1: Equivariance Error ─────────────────────────────────────────────

def equivariance_error(predict_fn, X_test, n_rotations=10, n_states=500):
    actual_states = min(n_states, len(X_test))
    idx = np.random.choice(len(X_test), size=actual_states, replace=False)
    X_sample = X_test[idx]

    states_sample = X_sample.reshape(actual_states, 24, 6).argmax(axis=2)

    actual_rotations = min(n_rotations, len(ALL_ROTATIONS))
    rot_indices = np.random.choice(len(ALL_ROTATIONS), size=actual_rotations, replace=False)
    rotations = [ALL_ROTATIONS[i] for i in rot_indices]

    f_original = predict_fn(X_sample)
    errors = []

    for rot_perm in rotations:
        rotated_states = np.array([apply_rotation(rot_perm, s) for s in states_sample],
                                  dtype=np.int8)
        X_rotated = np.array([encode_state(s) for s in rotated_states], dtype=np.float32)
        f_rotated = predict_fn(X_rotated)
        errors.append(np.abs(f_rotated - f_original))

    mean_err = float(np.mean(errors))
    return mean_err, np.array(errors)


# ─── Metric 2: Value Prediction Accuracy ─────────────────────────────────────

def value_prediction_metrics(predict_fn, X_test, y_test):
    preds = model_predict(predict_fn, X_test)
    y_true = y_test.astype(np.float32)

    mae = float(np.mean(np.abs(preds - y_true)))
    rounded_acc = float(np.mean(np.round(preds) == y_true))
    corr = float(np.corrcoef(preds, y_true)[0, 1])

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

def greedy_solve(predict_fn, initial_state, max_steps=50):
    state = initial_state.copy()
    move_sequence = []

    for step_i in range(max_steps):
        if is_solved(state):
            return True, step_i, move_sequence

        next_states = np.array([
            encode_state(apply_move(state, MOVES[m])) for m in range(NUM_MOVES)
        ], dtype=np.float32)

        preds = predict_fn(next_states)
        best_move = int(np.argmin(preds))
        state = apply_move(state, MOVES[best_move])
        move_sequence.append(best_move)

    return is_solved(state), len(move_sequence), move_sequence


def _beam_search_batch(predict_fn, initial_states, beam_width=5, max_steps=50):
    n = len(initial_states)
    solved = np.zeros(n, dtype=bool)
    n_moves_out = np.full(n, max_steps, dtype=int)
    beams = [[(s.copy(), 0)] for s in initial_states]

    for step in range(max_steps):
        for i in range(n):
            if solved[i]:
                continue
            for state, depth in beams[i]:
                if is_solved(state):
                    solved[i] = True
                    n_moves_out[i] = depth
                    break

        active = np.where(~solved)[0]
        if len(active) == 0:
            break

        expanded = []
        all_encoded = []

        for i in active:
            trial_cands = []
            for state, depth in beams[i]:
                for m in range(NUM_MOVES):
                    ns = apply_move(state, MOVES[m])
                    trial_cands.append((ns, depth + 1))
                    all_encoded.append(encode_state(ns))
            expanded.append(trial_cands)

        preds = model_predict(predict_fn, np.array(all_encoded, dtype=np.float32))

        offset = 0
        for idx, i in enumerate(active):
            cands = expanded[idx]
            count = len(cands)
            trial_preds = preds[offset:offset + count]
            offset += count
            top_k = min(beam_width, count)
            top_indices = np.argpartition(trial_preds, top_k - 1)[:top_k]
            beams[i] = [cands[j] for j in top_indices]

    for i in range(n):
        if not solved[i]:
            for state, depth in beams[i]:
                if is_solved(state):
                    solved[i] = True
                    n_moves_out[i] = depth
                    break

    return solved, n_moves_out


def evaluate_solve_rate(predict_fn, n_trials=500, scramble_depths=None,
                        beam_width=5, verbose=True):
    if scramble_depths is None:
        scramble_depths = [4, 7, 10, 14]

    results = {}
    rng = np.random.RandomState(42)

    for depth in scramble_depths:
        start_states = []
        for _ in range(n_trials):
            s = SOLVED_STATE.copy()
            last = -1
            for _k in range(depth):
                m = rng.choice(_CANDIDATES[last])
                s = apply_move(s, MOVES[m])
                last = m
            start_states.append(s)

        solved_arr, n_moves_arr = _beam_search_batch(
            predict_fn, start_states, beam_width=beam_width)

        solve_rate = float(solved_arr.mean())
        move_lengths = n_moves_arr[solved_arr].tolist()
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
    label = f"{model_type}_{size_label(train_size)}_seed{seed}"
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
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Learning Curves (train size = {size_label(train_size)})", fontsize=14)

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
            padded = np.array([c + [c[-1]] * (max_len - len(c)) for c in all_curves])
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

def run_full_evaluation(train_size_for_detail=200_000, train_sizes=None):
    if not EMLP_AVAILABLE:
        print("emlp not available.")
        return

    if train_sizes is None:
        train_sizes = TRAIN_SIZES
    if train_size_for_detail not in train_sizes:
        train_size_for_detail = train_sizes[-1]

    X_test, y_test = load_dataset("test")
    X_val, y_val = load_dataset("val")
    print(f"Test set: {X_test.shape}")

    all_val_mses = {}
    eq_errors = {}
    val_metrics = {}
    solve_results = defaultdict(dict)

    for model_type in ["emlp", "mlp"]:
        for train_size in train_sizes:
            for seed in SEEDS:
                try:
                    m = load_checkpoint(model_type, train_size, seed)
                    predict_fn_m = make_predict_fn(m)
                    preds_val = model_predict(predict_fn_m, X_val)
                    val_mse = float(np.mean((preds_val - y_val.astype(np.float32))**2))
                    all_val_mses[(model_type, train_size, seed)] = val_mse
                except Exception as e:
                    print(f"  Skip (no checkpoint): {e}")

            if train_size == train_size_for_detail:
                try:
                    model = load_checkpoint(model_type, train_size, 0)
                    predict_fn = make_predict_fn(model)

                    print(f"\n── {model_type.upper()} ({size_label(train_size)}) ──")

                    print("  Equivariance error...")
                    eq_err, _ = equivariance_error(predict_fn, X_test)
                    eq_errors[model_type] = eq_err
                    print(f"  Equivariance error: {eq_err:.2e}")

                    print("  Value prediction metrics...")
                    metrics = value_prediction_metrics(predict_fn, X_test, y_test)
                    val_metrics[model_type] = metrics
                    print(f"  MAE: {metrics['mae']:.4f}")
                    print(f"  Rounded accuracy: {metrics['rounded_accuracy']:.1%}")
                    print(f"  Pearson r: {metrics['correlation']:.4f}")

                    print("  Greedy solve rate...")
                    solve_res = evaluate_solve_rate(
                        predict_fn, n_trials=500, beam_width=5, verbose=True)
                    for depth, res in solve_res.items():
                        solve_results[depth][model_type] = res

                except FileNotFoundError as e:
                    print(f"  Skip: {e}")

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
                        choices=[50_000, 200_000, 1_000_000])
    args = parser.parse_args()
    run_full_evaluation(train_size_for_detail=args.train_size)
