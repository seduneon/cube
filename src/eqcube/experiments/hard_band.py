"""Diagnose the hard band (depths 7-10): which change helps it, and why?

    python -m eqcube.experiments.hard_band --out results/hard_band

Every variant is the ``task`` preset at 200k states with one change:

    baseline   alpha = 0.5, plain cross-entropy               (reference)
    balanced   every depth weighs the same in the loss         -> class prior
    alpha0     uniform over depths, depths <= 5 taken whole    -> class prior (via data)
    band3x     3x sampling weight on depths 7-10, same n       -> coverage
    n1m        1M states (5x data everywhere)                  -> coverage, upper bound
    parity     wrong-parity distances masked out               -> parity knowledge
    wide       width 24 instead of 12 (~4x params)             -> capacity
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Any

from eqcube.experiments.runner import EvalConfig, RunSpec, Workspace, run_one
from eqcube.experiments.summarize import BAND, band_summary
from eqcube.models.config import PRESETS
from eqcube.training.config import TrainConfig


def variants(seed: int = 42, base: TrainConfig | None = None) -> list[RunSpec]:
    base = base or TrainConfig(train_size=200_000, alpha=0.5, seed=seed)
    model = PRESETS["task"]
    boost = tuple(3.0 if d in BAND else 1.0 for d in range(15))
    changes: list[tuple[str, Any, TrainConfig]] = [
        ("baseline", model, base),
        ("balanced", model, replace(base, class_balance=True)),
        ("alpha0", model, replace(base, alpha=0.0, full_depth=5)),
        ("band3x", model, replace(base, depth_weights=boost)),
        ("n1m", model, replace(base, train_size=1_000_000)),
        ("parity", replace(model, parity_mask=True), base),
        ("wide", replace(model, width=24), base),
    ]
    return [RunSpec("task", m, t, EvalConfig(), tag) for tag, m, t in changes]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", default="results/hard_band")
    ap.add_argument("--cache", default="results/cache")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", nargs="*", help="run only these variants")
    args = ap.parse_args(argv)

    ws = Workspace(args.cache)
    specs = [s for s in variants(args.seed) if not args.only or s.tag in args.only]
    for i, spec in enumerate(specs, 1):
        print(f"=== {i}/{len(specs)}: {spec.run_id}", flush=True)
        run_one(spec, ws, args.out)
    print(band_summary(args.out))


if __name__ == "__main__":
    main()
