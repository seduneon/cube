"""Ordinal heads and Bellman consistency at two label budgets.

    python -m eqcube.experiments.bellman --out results/bellman

The ``task`` preset with one change per variant (alpha = 0.5, seed 42):

    categorical       log-softmax head, cross-entropy                  (reference)
    ordinal           CORN survival head, its NLL (= the CORN loss)
    ordinal_bce       CORN head, sum of BCEs over "d > k" (far errors cost more)
    categorical+bell  + Bellman consistency on unlabeled train-split states
    ordinal+bell      CORN head + Bellman consistency

Every run is also evaluated with temperature scaling and a one-step test-time
backup (both fit on validation), so all of Q2/Q3 appear in one table.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

from eqcube.experiments.runner import EvalConfig, RunSpec, Workspace, run_one
from eqcube.experiments.summarize import band_summary
from eqcube.models.config import PRESETS
from eqcube.training.config import TrainConfig


def variants(sizes: tuple[int, ...] = (20_000, 200_000), seed: int = 42) -> list[RunSpec]:
    model = PRESETS["task"]
    ordinal = replace(model, head="ordinal")
    specs: list[RunSpec] = []
    for n in sizes:
        base = TrainConfig(train_size=n, alpha=0.5, seed=seed)
        bell = replace(base, consistency_weight=1.0)
        for tag, m, t in [
            ("categorical", model, base),
            ("ordinal", ordinal, base),
            ("ordinal_bce", ordinal, replace(base, loss="ordinal_bce")),
            ("categorical+bell", model, bell),
            ("ordinal+bell", ordinal, bell),
        ]:
            specs.append(RunSpec("task", m, t, EvalConfig(), tag))
    return specs


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", default="results/bellman")
    ap.add_argument("--cache", default="results/cache")
    ap.add_argument("--sizes", nargs="+", type=int, default=[20_000, 200_000])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", nargs="*", help="run only these variant tags")
    args = ap.parse_args(argv)

    ws = Workspace(args.cache)
    specs = [
        s for s in variants(tuple(args.sizes), args.seed) if not args.only or s.tag in args.only
    ]
    for i, spec in enumerate(specs, 1):
        print(f"=== {i}/{len(specs)}: {spec.run_id}", flush=True)
        run_one(spec, ws, args.out)
    print(band_summary(args.out))


if __name__ == "__main__":
    main()
