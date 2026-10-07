"""Sibling ranking loss vs plain cross-entropy (docs/IMPROVEMENTS.md, Tier 2.1).

    python -m eqcube.experiments.ranking --out results/ranking --sizes 20000

The ``task`` preset, categorical head, alpha = 0.5; one change per variant:

    categorical        cross-entropy only (reference; must match results/bellman)
    categorical+rank   + ranking_weight * sibling ranking loss (training/ranking.py)

Every variant has its own tag, so run IDs never collide with older runs
(``RunSpec.run_id`` does not encode the loss or its weights).
"""

from __future__ import annotations

import argparse
from dataclasses import replace

from eqcube.experiments.runner import EvalConfig, RunSpec, Workspace, run_one
from eqcube.experiments.summarize import band_summary
from eqcube.models.config import PRESETS
from eqcube.training.config import TrainConfig


def variants(
    sizes: tuple[int, ...] = (20_000,), seeds: tuple[int, ...] = (42,), weight: float = 1.0
) -> list[RunSpec]:
    model = PRESETS["task"]
    specs: list[RunSpec] = []
    for n in sizes:
        for seed in seeds:
            base = TrainConfig(train_size=n, alpha=0.5, seed=seed)
            for tag, t in [
                ("categorical", base),
                ("categorical+rank", replace(base, ranking_weight=weight)),
            ]:
                specs.append(RunSpec("task", model, t, EvalConfig(), tag))
    return specs


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", default="results/ranking")
    ap.add_argument("--cache", default="results/cache")
    ap.add_argument("--sizes", nargs="+", type=int, default=[20_000])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--weight", type=float, default=1.0)
    ap.add_argument("--only", nargs="*", help="run only these variant tags")
    args = ap.parse_args(argv)

    ws = Workspace(args.cache)
    specs = [
        s
        for s in variants(tuple(args.sizes), tuple(args.seeds), args.weight)
        if not args.only or s.tag in args.only
    ]
    for i, spec in enumerate(specs, 1):
        print(f"=== {i}/{len(specs)}: {spec.run_id}", flush=True)
        run_one(spec, ws, args.out)
    print(band_summary(args.out))


if __name__ == "__main__":
    main()
