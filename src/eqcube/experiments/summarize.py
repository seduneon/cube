"""Collect result.json files into one table (mean ± std over seeds).

python -m eqcube.experiments.summarize results/exp1
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

COLUMNS: list[tuple[str, tuple[str, ...]]] = [
    ("params", ("n_params",)),
    ("acc(depth)", ("eval", "test/per_depth", "accuracy")),
    ("acc(uniform)", ("eval", "test/uniform", "accuracy")),
    ("mae(depth)", ("eval", "test/per_depth", "mae")),
    ("best_child", ("eval", "siblings/expected", "best_child")),
    ("solve(argmax)", ("eval", "search/argmax", "solve_rate")),
    ("solve(expected)", ("eval", "search/expected", "solve_rate")),
    ("excess", ("eval", "search/expected", "mean_excess")),
    ("eq_err(task)", ("eval", "equivariance", "task", "max")),
    ("nll", ("eval", "calibration", "raw", "nll")),
    ("T", ("eval", "calibration", "temperature")),
    ("ece(T)", ("eval", "calibration", "tempered", "ece")),
    ("backup_acc", ("eval", "backup", "backup", "accuracy")),
    ("backup_best", ("eval", "backup", "siblings/backup", "best_child")),
    ("train_s", ("train_seconds",)),
]


def _get(d: Any, path: tuple[str, ...]) -> float | None:
    for key in path:
        if not isinstance(d, dict) or key not in d:
            return None
        d = d[key]
    return float(d) if isinstance(d, (int, float)) else None


def load_results(root: str | Path) -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted(Path(root).rglob("result.json"))]


def summarize(root: str | Path) -> str:
    """Markdown table: one row per (preset, size, sampler, policy), seeds aggregated."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in load_results(root):
        groups[r["run_id"].rsplit("_s", 1)[0]].append(r)
    if not groups:
        return "(no results)"

    header = ["run", "seeds"] + [name for name, _ in COLUMNS]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for key in sorted(groups):
        rows = groups[key]
        cells = [key, str(len(rows))]
        for _, path in COLUMNS:
            values = [v for v in (_get(r, path) for r in rows) if v is not None]
            if not values:
                cells.append("—")
            elif len(values) == 1:
                cells.append(f"{values[0]:.4g}")
            else:
                cells.append(f"{np.mean(values):.4g} ± {np.std(values):.2g}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summarize(sys.argv[1] if len(sys.argv) > 1 else "results/runs"))


BAND = (7, 8, 9, 10)


def _band_mean(d: dict[str, float | None] | None, depths: tuple[int, ...]) -> float:
    if not d:
        return float("nan")
    values = [v for k in depths if (v := d.get(str(k))) is not None]
    return float(np.mean(values)) if values else float("nan")


def band_summary(root: str | Path, band: tuple[int, ...] = BAND) -> str:
    """One row per run: hard-band metrics, calibration, backup gain, search."""
    head = [
        "run", "params", "band acc", "band best_child", "acc(depth)", "nll", "T", "ece(T)",
        "backup band acc", "backup best_child", "solve", "excess", "train_s",
    ]  # fmt: skip
    lines = ["| " + " | ".join(head) + " |", "|" + "---:|" * len(head)]
    runs = load_results(root)
    for r in sorted(
        runs, key=lambda r: (r["train_config"]["train_size"], r.get("name", r["preset"]))
    ):
        e = r["eval"]
        backup = e.get("backup", {})
        cal = e.get("calibration", {})
        values = [
            _band_mean(e["test/per_depth"]["per_depth"]["accuracy"], band),
            _band_mean(e["siblings/expected"]["per_depth"]["best_child"], band),
            e["test/per_depth"]["accuracy"],
            _get(cal, ("raw", "nll")),
            _get(cal, ("temperature",)),
            _get(cal, ("tempered", "ece")),
            _band_mean(backup.get("backup", {}).get("per_depth", {}).get("accuracy"), band),
            _band_mean(
                backup.get("siblings/backup", {}).get("per_depth", {}).get("best_child"), band
            ),
            _get(e, ("search/expected", "solve_rate")),
            _get(e, ("search/expected", "mean_excess")),
        ]
        label = f"{r.get('name', r['preset'])} n={r['train_config']['train_size'] // 1000}k"
        cells = [label, f"{r['n_params']:,}"]
        cells += ["—" if v is None or v != v else f"{v:.3f}" for v in values]
        cells.append(f"{r['train_seconds']:.0f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
