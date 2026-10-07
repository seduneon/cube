"""Exact-fact wrappers on existing checkpoints (no retraining): docs/IMPROVEMENTS.md, Tier 0.

    python -m eqcube.experiments.posthoc --out results/posthoc

For every checkpoint, each wrapper chain is evaluated exactly as
``runner._evaluate_backup`` evaluates the base model (same suites, same
subsets, temperature and backup weights fit on val), so the "base" row must
reproduce the saved result.json numbers; the script checks this first.

    base       TemperedPredictor(net, T)                      (as saved)
    bound      + LowerBoundPredictor   (pattern databases + parity)
    inv        + InversionPredictor    (pool s and s^-1, geometric mean)
    bound+inv  both

``--search-widths 1 5`` adds beam search (greedy = width 1) for the base and
bound+inv chains, with and without the backup, plus a regression check of
the raw net at width 20 against the saved search numbers. ``--rules`` compares
the ranking rules of ``inference.ranking`` (best-child) on the same chains.

Each wrapped chain gets a second temperature refit on val, and its backup
must be used (beta >= 0.25): with the bound, val NLL alone can pick beta = 0
although the backup still improves ranking. Reads checkpoints
only; writes one JSON per checkpoint under --out.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from eqcube.data.table import DistanceTable
from eqcube.evaluation.metrics import rank_children, sibling_ranking, solve_metrics, value_metrics
from eqcube.experiments.runner import EvalConfig, Workspace, per_depth_subset
from eqcube.inference import (
    BackupPredictor,
    InversionPredictor,
    LowerBoundPredictor,
    Predictor,
    PredictorHeuristic,
    TemperedPredictor,
    fit_pooling,
    fit_temperature,
)
from eqcube.inference.ranking import RULES, child_costs
from eqcube.models.checkpoint import load_checkpoint
from eqcube.models.heuristic import NetHeuristic, NetPredictor
from eqcube.puzzles.pocket_coords import pocket_coords
from eqcube.search.beam import BeamSearch
from eqcube.training.config import TrainConfig, resolve_device

DEFAULT_RUNS = (
    "results/bellman/task-categorical/n200000_a0.5_random_s42",
    "results/bellman/task-ordinal+bell/n200000_a0.5_random_s42",
    "results/hard_band/task-n1m/n1000000_a0.5_random_s42",
)
BAND = (7, 8, 9, 10)
SEARCH_CHAINS = ("base", "bound+inv")  # also the chains for --rules
IntArray = npt.NDArray[np.int64]

Wrap = Callable[[Predictor], Predictor]
COORDS = pocket_coords()
CHAINS: dict[str, Wrap] = {
    "base": lambda p: p,
    "bound": lambda p: LowerBoundPredictor(p, COORDS),
    "inv": lambda p: InversionPredictor(p, COORDS),
    "bound+inv": lambda p: InversionPredictor(LowerBoundPredictor(p, COORDS), COORDS),
}


def band_mean(per_depth: dict[Any, float]) -> float:
    return float(np.mean([v for k, v in per_depth.items() if int(k) in BAND]))


def rule_ranking(predict: Predictor, table: DistanceTable, index: IntArray) -> dict[str, Any]:
    """best-child of every ranking rule, from one evaluation of parents and children."""
    index = index[table.dist[index] >= 1]
    children = table.indexer.expand(index)
    n, m = children.shape
    own = predict(table.indexer.unrank(index))
    child_lp = predict(table.indexer.unrank(children.reshape(-1))).reshape(n, m, -1)
    out: dict[str, Any] = {}
    for rule in RULES:
        r = rank_children(table, index, child_costs(rule, child_lp, own))
        out[rule] = {
            "best_child": r["best_child"],
            "band_best_child": band_mean(r["per_depth"]["best_child"]),
        }
    return out


def evaluate_chains(
    run_dir: Path,
    ws: Workspace,
    chains: list[str],
    widths: list[int] | None = None,
    rules: bool = False,
) -> dict[str, Any]:
    saved = json.loads((run_dir / "result.json").read_text())
    partition = TrainConfig.from_dict(saved["train_config"]).partition
    model, _, _ = load_checkpoint(run_dir / "best.pt")
    net = NetPredictor(model, resolve_device("auto"))
    table, suites = ws.table, ws.suites(partition)
    puzzle, k = table.indexer.puzzle, EvalConfig().backup_per_depth

    val = suites["val/per_depth"]
    val_states = table.indexer.unrank(val.index)
    t_raw = fit_temperature(net(val_states), val.distance)
    val_index = per_depth_subset(val.index, val.distance, k)
    test = suites["test/per_depth"]
    test_index = per_depth_subset(test.index, test.distance, k)
    states, labels = table.indexer.unrank(test_index), table.labels(test_index)

    out: dict[str, Any] = {
        "run": str(run_dir),
        "temperature_raw": t_raw,
        "chains": {},
        "search": {},
        "rules": {},
    }
    for name in chains:
        chain = CHAINS[name](TemperedPredictor(net, t_raw))
        t_refit = 1.0
        if name != "base":
            t_refit = fit_temperature(chain(val_states), val.distance)
            chain = TemperedPredictor(chain, t_refit)
        fit = fit_pooling(
            chain,
            puzzle,
            table.indexer.unrank(val_index),
            table.labels(val_index),
            min_beta=0.0 if name == "base" else 0.25,  # base: exactly as saved
        )
        backup = BackupPredictor(chain, puzzle, fit.w, fit.beta)
        own_rank = sibling_ranking(PredictorHeuristic(chain), table, test_index)
        backup_rank = sibling_ranking(PredictorHeuristic(backup), table, test_index)
        values = value_metrics(chain(states), labels)
        out["chains"][name] = {
            "temperature_refit": t_refit,
            "w": fit.w,
            "beta": fit.beta,
            "nll": values["ce"],
            "accuracy": values["accuracy"],
            "band_accuracy": band_mean(values["per_depth"]["accuracy"]),
            "best_child": own_rank["best_child"],
            "band_best_child": band_mean(own_rank["per_depth"]["best_child"]),
            "backup_best_child": backup_rank["best_child"],
            "backup_band_best_child": band_mean(backup_rank["per_depth"]["best_child"]),
            "siblings": own_rank,
            "siblings_backup": backup_rank,
        }
        if rules and name in SEARCH_CHAINS:
            out["rules"][name] = rule_ranking(chain, table, test_index)
            out["rules"][f"{name}+bk"] = rule_ranking(backup, table, test_index)
        if widths and name in SEARCH_CHAINS:
            search_index = suites["test/search"].index
            for width in widths:
                for label, predictor in (("", chain), ("+bk", backup)):
                    searcher = BeamSearch(
                        PredictorHeuristic(predictor),
                        width=width,
                        max_steps=EvalConfig().beam_max_steps,
                    )
                    out["search"][f"{name}{label}/w{width}"] = solve_metrics(
                        searcher, table, search_index
                    )

    if widths:  # regression check: the raw net as saved (no temperature, width 20)
        cfg = EvalConfig()
        heuristic = NetHeuristic(model, "expected", net.device)
        searcher = BeamSearch(heuristic, width=cfg.beam_width, max_steps=cfg.beam_max_steps)
        raw = solve_metrics(searcher, table, suites["test/search"].index)
        saved_search = saved["eval"]["search/expected"]
        out["search_regression"] = {
            key: [raw[key], saved_search[key]] for key in ("solve_rate", "mean_excess")
        }

    if "base" in out["chains"]:  # regression check against the saved evaluation
        b, s = out["chains"]["base"], saved["eval"]["backup"]
        out["regression"] = {
            "best_child": [b["best_child"], s["siblings/base"]["best_child"]],
            "backup_best_child": [b["backup_best_child"], s["siblings/backup"]["best_child"]],
            "w_beta": [[b["w"], b["beta"]], [s["w"], s["beta"]]],
        }
    return out


def table_rows(result: dict[str, Any]) -> str:
    name = Path(result["run"]).parent.name
    lines = [f"\n{name}  (T_raw = {result['temperature_raw']:.2f})"]
    lines.append(
        f"  {'chain':10s} {'nll':>6s} {'acc':>6s} {'band acc':>8s} {'best':>6s} "
        f"{'band best':>9s} {'bk best':>7s} {'bk band':>7s}"
    )
    for chain, r in result["chains"].items():
        lines.append(
            f"  {chain:10s} {r['nll']:6.3f} {r['accuracy']:6.3f} {r['band_accuracy']:8.3f} "
            f"{r['best_child']:6.3f} {r['band_best_child']:9.3f} "
            f"{r['backup_best_child']:7.3f} {r['backup_band_best_child']:7.3f}"
        )
    if "regression" in result:
        reg = result["regression"]
        ok = all(abs(a - b) < 1e-9 for a, b in (reg["best_child"], reg["backup_best_child"]))
        lines.append(f"  regression vs saved result.json: {'OK' if ok else 'MISMATCH'} {reg}")
    if result["search"]:
        lines.append(f"  {'search':22s} {'solved':>7s} {'excess':>7s}  band(d=10,12) solved")
        for key, r in result["search"].items():
            per = r["per_depth"]["solve_rate"]
            band = np.mean([v for d, v in per.items() if int(d) in (10, 12)])
            lines.append(f"  {key:22s} {r['solve_rate']:7.3f} {r['mean_excess']:7.3f}  {band:.3f}")
    if result["rules"]:
        chains = list(result["rules"])
        lines.append("  rule        " + " ".join(f"{c:>17s}" for c in chains))
        for rule in RULES:
            cells = [result["rules"][c][rule] for c in chains]
            lines.append(
                f"  {rule:10s}  "
                + " ".join(f"{x['best_child']:8.3f} ({x['band_best_child']:.3f})" for x in cells)
            )
    if "search_regression" in result:
        reg = result["search_regression"]
        ok = all(abs(a - b) < 1e-9 for a, b in reg.values())
        lines.append(f"  search regression (raw, width 20): {'OK' if ok else 'MISMATCH'} {reg}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=list(DEFAULT_RUNS))
    ap.add_argument("--chains", nargs="+", default=list(CHAINS), choices=list(CHAINS))
    ap.add_argument(
        "--search-widths", nargs="*", type=int, default=[], help="e.g. 1 5 (beam search)"
    )
    ap.add_argument("--rules", action="store_true", help="compare ranking rules (best-child)")
    ap.add_argument("--out", default="results/posthoc")
    ap.add_argument("--cache", default="results/cache")
    args = ap.parse_args(argv)

    ws = Workspace(args.cache)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for run in args.runs:
        result = evaluate_chains(Path(run), ws, args.chains, args.search_widths, args.rules)
        name = "_".join(Path(run).parts[-2:])
        (out_dir / f"{name}.json").write_text(json.dumps(result, indent=1, default=float))
        print(table_rows(result), flush=True)


if __name__ == "__main__":
    main()
