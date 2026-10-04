"""CLI: run an experiment grid (resumable).

    python -m eqcube.experiments.run --presets task mlp_small --sizes 50000 200000 \
        --alphas 0.5 --seeds 42 43 44 --out results/exp1 --cache results/cache

    --quick   tiny smoke run (2k states, 3 epochs, small eval) to check a machine
"""

from __future__ import annotations

import argparse
from dataclasses import replace

from eqcube.data.partition import PartitionConfig
from eqcube.experiments.runner import EvalConfig, Workspace, grid, run_one
from eqcube.experiments.summarize import summarize
from eqcube.models.config import PRESETS
from eqcube.training.config import TrainConfig


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--presets", nargs="+", default=sorted(PRESETS), choices=sorted(PRESETS))
    ap.add_argument("--sizes", nargs="+", type=int, default=[50_000, 200_000, 1_000_000])
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.5])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--policy", choices=["random", "orbit"], default="random")
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-search", action="store_true", help="skip beam-search evaluation")
    ap.add_argument("--out", default="results/runs")
    ap.add_argument("--cache", default="results/cache")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--reevaluate", action="store_true", help="recompute eval of finished runs")
    args = ap.parse_args(argv)

    base = TrainConfig(
        partition=PartitionConfig(policy=args.policy),
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        device=args.device,
    )
    eval_cfg = EvalConfig(search=not args.no_search)
    sizes = args.sizes
    if args.quick:
        base = replace(base, max_epochs=3, patience=3)
        eval_cfg = EvalConfig(sibling_per_depth=20, equivariance_states=50, beam_max_steps=16)
        sizes = [2000]

    ws = Workspace(args.cache)
    specs = grid(args.presets, sizes, args.alphas, args.seeds, base, eval_cfg)
    for i, spec in enumerate(specs, 1):
        print(f"=== run {i}/{len(specs)}: {spec.run_id}", flush=True)
        run_one(spec, ws, args.out, reevaluate=args.reevaluate)
    print(summarize(args.out))


if __name__ == "__main__":
    main()
