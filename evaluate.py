"""
Evaluation, metrics, and plots for the cube value network experiment.
PyTorch backend.

Supports any subset of model types registered in MODEL_REGISTRY.
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
    apply_move, encode_state, is_solved, state_to_tuple, tuple_to_state,
)
from cube_symmetry import ALL_ROTATIONS, ALL_COLOR_PERMS, apply_color_perm_batch, apply_rotation
from models import MODEL_REGISTRY, load_model, get_param_count, DEVICE
from dataset import load_dataset, DATA_DIR, load_bfs_tables, bfs_tables_exist, _CANDIDATES, MAX_DISTANCE
from train import CKPT_DIR, LOG_DIR, TRAIN_SIZES, SEEDS, size_label

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
PLOT_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

MAX_BFS_DISTANCE = MAX_DISTANCE  # 14
BEAM_WIDTH = 20

# ─── Plot styling ─────────────────────────────────────────────────────────────

MODEL_COLORS = {
    "emlp":     "#2196F3",   # blue
    "emlp_col": "#4CAF50",   # green
    "mlp":      "#FF5722",   # orange-red
    "mlp_aug":  "#9C27B0",   # purple
}

def _model_color(key):
    return MODEL_COLORS.get(key, "#888888")

def _model_label(key):
    return MODEL_REGISTRY[key].label if key in MODEL_REGISTRY else key


# ─── Load model checkpoint ────────────────────────────────────────────────────

def load_checkpoint(model_type, train_size, seed, verbose=True):
    label = f"{model_type}_{size_label(train_size)}_seed{seed}"
    ckpt_path = os.path.join(CKPT_DIR, f"{label}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model = load_model(model_type, ckpt_path)
    model = model.to(DEVICE)
    model.eval()
    if verbose:
        n_params = get_param_count(model)
        print(f"  Loaded {label}  ({n_params:,} params)")
    return model


def make_predict_fn(model):
    """Return a predict function: numpy (N, 144) -> numpy (N,) predictions in move units.

    Models are trained on targets normalized to [0, 1].  This function scales
    predictions back to the original distance scale (0 – MAX_DISTANCE moves)
    so that all downstream evaluation code works in interpretable units.
    """
    def _predict(X_np):
        with torch.no_grad():
            x = torch.tensor(X_np, dtype=torch.float32, device=DEVICE)
            pred = model(x).squeeze().cpu().numpy()
            return pred * MAX_BFS_DISTANCE   # scale [0,1] → [0, MAX_DISTANCE]
    return _predict


def model_predict(predict_fn, X_np, batch_size=512):
    preds = []
    for start in range(0, len(X_np), batch_size):
        preds.append(predict_fn(X_np[start:start + batch_size]))
    return np.concatenate(preds)


# ─── Metric 1: Equivariance Error ─────────────────────────────────────────────

def equivariance_error(predict_fn, X_test, symmetries=("spatial",),
                       n_samples=10, n_states=500):
    """Measure equivariance error for each requested symmetry type.

    Parameters
    ----------
    predict_fn  : callable (N, 144) -> (N,)  [in move units]
    X_test      : (N, 144) float32 one-hot states
    symmetries  : tuple of: "spatial", "color", "combined"
    n_samples   : number of random group elements per state
    n_states    : number of test states to sample

    Returns
    -------
    dict[str, float]  — mean absolute equivariance error per symmetry type.
    Error is in the same units as predict_fn output (moves).
    """
    actual_states = min(n_states, len(X_test))
    idx = np.random.choice(len(X_test), size=actual_states, replace=False)
    X_sample = X_test[idx]
    f_original = predict_fn(X_sample)

    results = {}

    if "spatial" in symmetries:
        states_24 = X_sample.reshape(actual_states, 24, 6).argmax(axis=2)
        n_rots = min(n_samples, len(ALL_ROTATIONS))
        rot_indices = np.random.choice(len(ALL_ROTATIONS), size=n_rots, replace=False)
        errors = []
        for ri in rot_indices:
            rot_perm = ALL_ROTATIONS[ri]
            rotated = np.array([apply_rotation(rot_perm, s) for s in states_24], dtype=np.int8)
            X_rot = np.array([encode_state(s) for s in rotated], dtype=np.float32)
            errors.append(np.abs(predict_fn(X_rot) - f_original))
        results["spatial"] = float(np.mean(errors))

    if "color" in symmetries:
        n_perms = min(n_samples, len(ALL_COLOR_PERMS))
        perm_indices = np.random.choice(len(ALL_COLOR_PERMS), size=n_perms, replace=False)
        errors = []
        for pi in perm_indices:
            perm = ALL_COLOR_PERMS[pi]
            X_perm = apply_color_perm_batch(X_sample, perm)
            errors.append(np.abs(predict_fn(X_perm) - f_original))
        results["color"] = float(np.mean(errors))

    if "combined" in symmetries:
        # Apply a random spatial rotation AND a random color permutation together
        states_24 = X_sample.reshape(actual_states, 24, 6).argmax(axis=2)
        rng = np.random.RandomState(7)
        errors = []
        for _ in range(n_samples):
            rot_perm = ALL_ROTATIONS[rng.randint(len(ALL_ROTATIONS))]
            color_perm = ALL_COLOR_PERMS[rng.randint(len(ALL_COLOR_PERMS))]
            rotated = np.array([apply_rotation(rot_perm, s) for s in states_24], dtype=np.int8)
            X_both = np.array([encode_state(s) for s in rotated], dtype=np.float32)
            X_both = apply_color_perm_batch(X_both, color_perm)
            errors.append(np.abs(predict_fn(X_both) - f_original))
        results["combined"] = float(np.mean(errors))

    return results


# ─── Metric 2: Value Prediction Accuracy ─────────────────────────────────────

def value_prediction_metrics(predict_fn, X_test, y_test):
    """Compute prediction accuracy metrics.

    predict_fn returns predictions already scaled to move units.
    y_test contains raw BFS distances (integers 0–MAX_DISTANCE).
    """
    preds = model_predict(predict_fn, X_test)   # in move units
    y_true = y_test.astype(np.float32)           # raw distances

    mae = float(np.mean(np.abs(preds - y_true)))
    rounded_acc = float(np.mean(np.round(preds) == y_true))
    corr = float(np.corrcoef(preds, y_true)[0, 1])

    per_depth_mae = {}
    for d in range(MAX_BFS_DISTANCE + 1):
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
                        beam_width=20, verbose=True):
    if scramble_depths is None:
        scramble_depths = [4, 7, 10, 14]

    if not bfs_tables_exist():
        if verbose:
            print("  BFS table not found — skipping solve rate evaluation.")
        return {}

    if verbose:
        print("  Loading BFS table for exact-distance sampling...")
    dist_table, _ = load_bfs_tables()

    states_by_depth = defaultdict(list)
    for state_t, d in dist_table.items():
        states_by_depth[d].append(state_t)

    results = {}
    rng = np.random.RandomState(42)

    for depth in scramble_depths:
        pool = states_by_depth.get(depth, [])
        if not pool:
            continue
        n = min(n_trials, len(pool))
        indices = rng.choice(len(pool), size=n, replace=False)
        start_states = [tuple_to_state(pool[i]) for i in indices]
        solved_arr, n_moves_arr = _beam_search_batch(predict_fn, start_states, beam_width=beam_width)
        solve_rate = float(solved_arr.mean())
        move_lengths = n_moves_arr[solved_arr].tolist()
        mean_moves = float(np.mean(move_lengths)) if move_lengths else float("nan")
        mean_excess = mean_moves - depth if move_lengths else float("nan")
        results[depth] = {"solve_rate": solve_rate, "mean_moves": mean_moves, "mean_excess": mean_excess}
        if verbose:
            print(f"  depth={depth:2d}: solve_rate={solve_rate:.1%}  "
                  f"mean_moves={mean_moves:.1f}  excess={mean_excess:+.1f}  (n={n})")

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

def plot_learning_curves(model_types, train_size=200_000):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Learning Curves (train size = {size_label(train_size)})", fontsize=14)

    for ax_idx, metric in enumerate(["train_mse", "val_mse"]):
        ax = axes[ax_idx]
        ax.set_title(metric.replace("_", " ").title() + " (normalized)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE")

        for key in model_types:
            all_curves = []
            for seed in SEEDS:
                log = load_log(key, train_size, seed)
                if log is not None:
                    all_curves.append([row[metric] for row in log])
            if not all_curves:
                continue
            max_len = max(len(c) for c in all_curves)
            padded = np.array([c + [c[-1]] * (max_len - len(c)) for c in all_curves])
            mean = padded.mean(axis=0)
            std = padded.std(axis=0)
            epochs = np.arange(1, max_len + 1)
            ax.plot(epochs, mean, label=_model_label(key), color=_model_color(key))
            ax.fill_between(epochs, mean - std, mean + std,
                            alpha=0.2, color=_model_color(key))
        ax.legend()
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "learning_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_data_efficiency(all_val_mses, model_types):
    """Val MSE vs training set size.  all_val_mses is in raw move² units."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("Data Efficiency: Val MSE vs Training Set Size")
    ax.set_xlabel("Training set size")
    ax.set_ylabel("Val MSE (moves²)")
    ax.set_xscale("log")
    ax.set_yscale("log")

    for key in model_types:
        means, stds, sizes = [], [], []
        for train_size in TRAIN_SIZES:
            mses = [all_val_mses.get((key, train_size, s)) for s in SEEDS]
            mses = [m for m in mses if m is not None]
            if mses:
                means.append(np.mean(mses))
                stds.append(np.std(mses))
                sizes.append(train_size)
        if means:
            ax.errorbar(sizes, means, yerr=stds, marker="o",
                        label=_model_label(key), color=_model_color(key),
                        capsize=5, linewidth=2, markersize=8)

    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "data_efficiency.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_per_depth_accuracy(metrics_by_model):
    depths = list(range(MAX_BFS_DISTANCE + 1))
    keys = list(metrics_by_model.keys())
    n_models = len(keys)
    width = 0.8 / n_models
    x = np.arange(len(depths))

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, key in enumerate(keys):
        mae_vals = [metrics_by_model[key]["per_depth_mae"].get(d) or 0 for d in depths]
        offset = (i - (n_models - 1) / 2) * width
        ax.bar(x + offset, mae_vals, width, label=_model_label(key),
               color=_model_color(key), alpha=0.8)

    ax.set_title("Per-Depth MAE (moves)")
    ax.set_xlabel("Optimal Distance (BFS depth)")
    ax.set_ylabel("Mean Absolute Error (moves)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in depths])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "per_depth_accuracy.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_equivariance_error(eq_errors_by_model):
    """eq_errors_by_model: dict[model_key -> dict[symmetry_type -> float]]"""
    symmetry_types = ["spatial", "color", "combined"]
    has_sym = {sym: any(sym in v for v in eq_errors_by_model.values())
               for sym in symmetry_types}
    active_syms = [s for s in symmetry_types if has_sym[s]]

    if not active_syms:
        return

    n_syms = len(active_syms)
    fig, axes = plt.subplots(1, n_syms, figsize=(5 * n_syms, 5))
    if n_syms == 1:
        axes = [axes]

    for ax, sym in zip(axes, active_syms):
        keys = [k for k, v in eq_errors_by_model.items() if sym in v]
        errors = [eq_errors_by_model[k][sym] for k in keys]
        colors = [_model_color(k) for k in keys]
        labels = [_model_label(k) for k in keys]

        bars = ax.bar(labels, errors, color=colors, alpha=0.8, width=0.5)
        ax.set_yscale("log")
        ax.set_ylabel("Mean |f(g·x) − f(x)| (moves)")
        ax.set_title(f"{sym.capitalize()} equivariance error")
        ax.grid(True, alpha=0.3, axis="y")
        for bar, err in zip(bars, errors):
            ax.text(bar.get_x() + bar.get_width() / 2, err * 1.5,
                    f"{err:.2e}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "equivariance_error.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_solve_rates(solve_results, model_types):
    """Grouped bar chart: solve rate by scramble depth for each model."""
    if not solve_results:
        return

    depths = sorted(solve_results.keys())
    n_models = len(model_types)
    x = np.arange(len(depths))
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, key in enumerate(model_types):
        rates = [solve_results[d].get(key, {}).get("solve_rate", 0) * 100
                 for d in depths]
        offset = (i - (n_models - 1) / 2) * width
        ax.bar(x + offset, rates, width, label=_model_label(key),
               color=_model_color(key), alpha=0.8)

    ax.set_title("Greedy Solve Rate by Scramble Depth")
    ax.set_xlabel("BFS Distance (scramble depth)")
    ax.set_ylabel("Solve Rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in depths])
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "solve_rates.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_param_comparison(model_types, train_size, seed=SEEDS[0]):
    """Bar chart comparing total parameter counts."""
    param_counts = {}
    for key in model_types:
        try:
            m = load_checkpoint(key, train_size, seed, verbose=False)
            param_counts[key] = get_param_count(m)
        except FileNotFoundError:
            pass

    if not param_counts:
        return

    keys = list(param_counts.keys())
    counts = [param_counts[k] for k in keys]
    labels = [_model_label(k) for k in keys]
    colors = [_model_color(k) for k in keys]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, counts, color=colors, alpha=0.8, width=0.5)
    ax.set_yscale("log")
    ax.set_ylabel("Number of parameters")
    ax.set_title("Parameter Count Comparison")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.1,
                f"{count:,}", ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "param_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def save_solve_rate_table(solve_results, model_types, path=None):
    if path is None:
        path = os.path.join(PLOT_DIR, "solve_rate_table.txt")

    lines = [
        "Greedy Solve Rate",
        "=" * 80,
        f"{'Depth':>6}  {'Model':<25}  {'Solve Rate':>12}  {'Mean Moves':>12}  {'Excess':>10}",
        "-" * 80,
    ]
    for depth in sorted(solve_results.keys()):
        for key in model_types:
            if key in solve_results[depth]:
                r = solve_results[depth][key]
                lines.append(
                    f"{depth:>6}  {_model_label(key):<25}  "
                    f"{r['solve_rate']:>11.1%}  "
                    f"{r['mean_moves']:>12.1f}  "
                    f"{r['mean_excess']:>+10.1f}"
                )

    text = "\n".join(lines)
    with open(path, "w") as f:
        f.write(text)
    print(text)
    print(f"\nSaved {path}")


# ─── Summary table ────────────────────────────────────────────────────────────

def print_summary_table(model_types, all_val_mses, val_metrics,
                        eq_errors, solve_results, train_size):
    """Print a formatted summary table across all evaluated models.

    all_val_mses : dict[(key, train_size, seed) -> float]  (in move² units)
    val_metrics  : dict[key -> {mae, rounded_accuracy, correlation, ...}]
    eq_errors    : dict[key -> {spatial/color/combined -> float}]
    solve_results: dict[depth -> {key -> {solve_rate, ...}}]
    """
    print(f"\n{'='*80}")
    print(f"  RESULTS SUMMARY  ({size_label(train_size)} training set, mean ± std over {len(SEEDS)} seeds)")
    print(f"{'='*80}")

    col_w = max(20, *(len(_model_label(k)) + 2 for k in model_types))
    header = f"{'Metric':<28}" + "".join(f"  {_model_label(k):<{col_w}}" for k in model_types)
    print(header)
    print("-" * len(header))

    def row(name, fmt_fn):
        line = f"  {name:<26}"
        for k in model_types:
            try:
                line += "  " + fmt_fn(k).ljust(col_w)
            except Exception:
                line += "  " + "N/A".ljust(col_w)
        print(line)

    # Parameter count
    def param_str(k):
        try:
            m = load_checkpoint(k, train_size, SEEDS[0], verbose=False)
            return f"{get_param_count(m):,}"
        except Exception:
            return "N/A"
    row("Parameters", param_str)

    # Val MSE (mean ± std across seeds, in move² units)
    def val_mse_str(k):
        vals = [all_val_mses.get((k, train_size, s)) for s in SEEDS]
        vals = [v for v in vals if v is not None]
        if not vals:
            return "N/A"
        return f"{np.mean(vals):.4f}±{np.std(vals):.4f}"
    row("Val MSE (moves²)", val_mse_str)

    # Test MAE
    def mae_str(k):
        m = val_metrics.get(k)
        return f"{m['mae']:.3f}" if m else "N/A"
    row("Test MAE (moves)", mae_str)

    # Rounded accuracy
    def racc_str(k):
        m = val_metrics.get(k)
        return f"{m['rounded_accuracy']:.1%}" if m else "N/A"
    row("Rounded accuracy", racc_str)

    # Pearson r
    def corr_str(k):
        m = val_metrics.get(k)
        return f"{m['correlation']:.4f}" if m else "N/A"
    row("Pearson r", corr_str)

    # Equivariance errors
    for sym in ("spatial", "color", "combined"):
        def eq_str(k, _sym=sym):
            eq = eq_errors.get(k, {})
            return f"{eq[_sym]:.2e}" if _sym in eq else "—"
        row(f"Equiv err ({sym})", eq_str)

    # Inference time
    def infer_str(k):
        m = val_metrics.get(k)
        ms = m.get('ms_per_sample') if m else None
        return f"{ms:.4f} ms" if ms is not None else "N/A"
    row("Inference (ms/sample)", infer_str)

    # Solve rates
    for depth in sorted(solve_results.keys()):
        def sr_str(k, _d=depth):
            r = solve_results[_d].get(k)
            return f"{r['solve_rate']:.1%}" if r else "N/A"
        row(f"Solve rate (d={depth})", sr_str)

    print(f"{'='*80}\n")


# ─── Full evaluation run ──────────────────────────────────────────────────────

def run_full_evaluation(train_size_for_detail=200_000, train_sizes=None,
                        model_types=None, beam_width=BEAM_WIDTH):
    if train_sizes is None:
        train_sizes = TRAIN_SIZES
    if train_size_for_detail not in train_sizes:
        train_size_for_detail = train_sizes[-1]
    if model_types is None:
        model_types = ["emlp", "mlp"]

    X_test, y_test = load_dataset("test")
    X_val, y_val = load_dataset("val")
    print(f"Test set: {X_test.shape}, y range: {y_test.min()}–{y_test.max()}")

    # all_val_mses: in move² units (predictions scaled by MAX_DISTANCE)
    all_val_mses = {}
    eq_errors = {}
    val_metrics = {}
    solve_results = defaultdict(dict)

    for key in model_types:
        spec = MODEL_REGISTRY.get(key)

        # Data efficiency: val MSE across all train sizes and seeds
        for train_size in train_sizes:
            for seed in SEEDS:
                try:
                    m = load_checkpoint(key, train_size, seed, verbose=False)
                    predict_fn_m = make_predict_fn(m)
                    preds_val = model_predict(predict_fn_m, X_val)   # in move units
                    val_mse = float(np.mean((preds_val - y_val.astype(np.float32))**2))
                    all_val_mses[(key, train_size, seed)] = val_mse
                except Exception as e:
                    print(f"  Skip (no checkpoint): {e}")

        # Detailed metrics at the chosen train_size
        try:
            model = load_checkpoint(key, train_size_for_detail, SEEDS[0])
            predict_fn = make_predict_fn(model)

            print(f"\n── {_model_label(key)} ({size_label(train_size_for_detail)}) ──")

            # Which symmetries to test
            test_syms = set(spec.symmetries if spec else ())
            test_syms.add("spatial")  # always test spatial
            if "spatial" in test_syms and "color" in test_syms:
                test_syms.add("combined")

            print("  Equivariance error...")
            eq_result = equivariance_error(predict_fn, X_test,
                                           symmetries=tuple(test_syms))
            eq_errors[key] = eq_result
            for sym, err in sorted(eq_result.items()):
                print(f"    {sym}: {err:.2e} moves")

            print("  Value prediction metrics...")
            metrics = value_prediction_metrics(predict_fn, X_test, y_test)
            val_metrics[key] = metrics
            print(f"  MAE: {metrics['mae']:.3f} moves")
            print(f"  Rounded accuracy: {metrics['rounded_accuracy']:.1%}")
            print(f"  Pearson r: {metrics['correlation']:.4f}")

            import time as _time
            _warmup = model_predict(predict_fn, X_test[:32])
            _t0 = _time.perf_counter()
            _preds = model_predict(predict_fn, X_test)
            _elapsed = _time.perf_counter() - _t0
            _ms = _elapsed / len(X_test) * 1000
            metrics['ms_per_sample'] = _ms
            print(f"  Inference: {_ms:.4f} ms/sample ({len(X_test)} samples)")

            print(f"  Greedy solve rate (beam_width={beam_width})...")
            solve_res = evaluate_solve_rate(predict_fn, n_trials=500,
                                            beam_width=beam_width, verbose=True)
            for depth, res in solve_res.items():
                solve_results[depth][key] = res

        except FileNotFoundError as e:
            print(f"  Skip: {e}")

    print("\nGenerating plots...")
    plot_learning_curves(model_types, train_size=train_size_for_detail)
    if all_val_mses:
        plot_data_efficiency(all_val_mses, model_types)
    if val_metrics:
        plot_per_depth_accuracy(val_metrics)
    if eq_errors:
        plot_equivariance_error(eq_errors)
    if solve_results:
        plot_solve_rates(solve_results, model_types)
        save_solve_rate_table(solve_results, model_types)
    plot_param_comparison(model_types, train_size_for_detail)

    print_summary_table(model_types, all_val_mses, val_metrics,
                        eq_errors, solve_results, train_size_for_detail)

    print("Evaluation complete. Plots saved to:", PLOT_DIR)


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from train import ALL_MODEL_TYPES

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=int, default=200_000,
                        choices=[50_000, 200_000, 1_000_000])
    parser.add_argument("--models", nargs="+",
                        choices=ALL_MODEL_TYPES,
                        default=["emlp", "mlp"],
                        help="Model types to evaluate")
    parser.add_argument("--beam-width", type=int, default=BEAM_WIDTH,
                        help=f"Beam width for solve-rate evaluation (default: {BEAM_WIDTH})")
    args = parser.parse_args()
    run_full_evaluation(train_size_for_detail=args.train_size,
                        model_types=args.models,
                        beam_width=args.beam_width)
