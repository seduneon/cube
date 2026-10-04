"""Experiment grid: presets x train sizes x alphas x seeds -> one result.json per run.

Runs are resumable: a run whose ``result.json`` exists is skipped, so a grid
can be stopped and restarted on any machine (laptop, Kaggle, cluster).

    python -m eqcube.experiments.run --presets task mlp_small --sizes 50000 \
        --alphas 0.5 --seeds 42 --out results/exp1 --cache results/cache
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from eqcube.data.partition import Partition, PartitionConfig
from eqcube.data.suites import EvalSuite, default_suites
from eqcube.data.symmetry import orbit_representatives
from eqcube.data.table import DistanceTable
from eqcube.evaluation.metrics import (
    equivariance_error,
    sibling_ranking,
    solve_metrics,
    value_metrics,
)
from eqcube.models.checkpoint import load_checkpoint
from eqcube.models.config import PRESETS, ModelConfig
from eqcube.models.heuristic import NetHeuristic, NetPredictor
from eqcube.puzzles import PUZZLES
from eqcube.puzzles.pocket_coords import pocket_coords
from eqcube.search.beam import BeamSearch
from eqcube.training.config import TrainConfig, resolve_device
from eqcube.training.trainer import Trainer


@dataclass(frozen=True)
class EvalConfig:
    sibling_per_depth: int = 300
    equivariance_states: int = 500
    beam_width: int = 20
    beam_max_steps: int = 30
    search: bool = True


@dataclass(frozen=True)
class RunSpec:
    preset: str
    model: ModelConfig
    train: TrainConfig
    eval: EvalConfig = field(default_factory=EvalConfig)

    @property
    def run_id(self) -> str:
        t = self.train
        sampler = f"a{t.alpha:g}" if t.sampler == "tempered" else f"walk{t.walk_length}"
        return f"{self.preset}/n{t.train_size}_{sampler}_{t.partition.policy}_s{t.seed}"


def grid(
    presets: list[str],
    sizes: list[int],
    alphas: list[float],
    seeds: list[int],
    base: TrainConfig | None = None,
    eval_cfg: EvalConfig | None = None,
) -> list[RunSpec]:
    base = base or TrainConfig()
    return [
        RunSpec(
            p, PRESETS[p], replace(base, train_size=n, alpha=a, seed=s), eval_cfg or EvalConfig()
        )
        for p in presets
        for n in sizes
        for a in alphas
        for s in seeds
    ]


class Workspace:
    """Shared, cached data for all runs (table, partitions, suites)."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.table = DistanceTable.build(pocket_coords(), self.cache_dir)
        self._partitions: dict[PartitionConfig, Partition] = {}
        self._suites: dict[PartitionConfig, dict[str, EvalSuite]] = {}

    def partition(self, cfg: PartitionConfig) -> Partition:
        if cfg not in self._partitions:
            reps = None
            if cfg.policy == "orbit":
                group = PUZZLES["pocket_cube"].symmetries["task"]()
                reps = orbit_representatives(
                    self.table.indexer, group, cache_path=self.cache_dir / "task_orbit_reps.npy"
                )
            self._partitions[cfg] = Partition(self.table, cfg, orbit_reps=reps)
        return self._partitions[cfg]

    def suites(self, cfg: PartitionConfig) -> dict[str, EvalSuite]:
        if cfg not in self._suites:
            self._suites[cfg] = default_suites(self.partition(cfg))
        return self._suites[cfg]


def evaluate(spec: RunSpec, ws: Workspace, checkpoint: Path) -> dict[str, Any]:
    model, model_cfg, _ = load_checkpoint(checkpoint)
    device = resolve_device(spec.train.device)
    predict = NetPredictor(model, device)
    suites = ws.suites(spec.train.partition)
    table, rng = ws.table, np.random.default_rng(0)
    report: dict[str, Any] = {}

    for name in ("test/per_depth", "test/uniform"):
        suite = suites[name]
        report[name] = value_metrics(predict(table.indexer.unrank(suite.index)), suite.distance)

    per_depth = suites["test/per_depth"]
    sib = np.concatenate(
        [
            per_depth.index[per_depth.distance == d][: spec.eval.sibling_per_depth]
            for d in np.unique(per_depth.distance)
        ]
    )
    for reduce in ("argmax", "expected"):
        report[f"siblings/{reduce}"] = sibling_ranking(
            NetHeuristic(model, reduce, device), table, sib
        )

    sample = rng.choice(per_depth.index, size=spec.eval.equivariance_states, replace=False)
    states = table.indexer.unrank(sample)
    symmetries = PUZZLES[model_cfg.puzzle].symmetries
    report["equivariance"] = {
        "task": equivariance_error(predict, states, symmetries["task"]()),
        "own": equivariance_error(predict, states, symmetries[model_cfg.symmetry]())
        if model_cfg.symmetry in ("task", "trivial", "rotation")
        else None,  # large groups: skip (17k+ elements)
    }

    if spec.eval.search:
        for reduce in ("argmax", "expected"):
            searcher = BeamSearch(
                NetHeuristic(model, reduce, device),
                width=spec.eval.beam_width,
                max_steps=spec.eval.beam_max_steps,
            )
            report[f"search/{reduce}"] = solve_metrics(searcher, table, suites["test/search"].index)
    return report


def run_one(
    spec: RunSpec, ws: Workspace, out_root: str | Path, verbose: bool = True
) -> dict[str, Any]:
    out = Path(out_root) / spec.run_id
    result_path = out / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())

    def log(row: dict[str, Any]) -> None:
        if verbose:
            print(
                f"[{spec.run_id}] epoch {row['epoch']:3d}  train_ce {row['train_ce']:.4f}  "
                f"val_ce {row['val_ce']:.4f}  val_acc {row['val_accuracy']:.3f}  "
                f"{row['seconds']:.1f}s",
                flush=True,
            )

    t0 = time.time()
    trainer = Trainer(spec.model, spec.train, ws.table, ws.partition(spec.train.partition), log)
    train = trainer.fit(out)
    report = evaluate(spec, ws, out / "best.pt")
    result = {
        "run_id": spec.run_id,
        "preset": spec.preset,
        "model_config": spec.model.to_dict(),
        "train_config": spec.train.to_dict(),
        "device": str(trainer.device),
        "n_params": train.n_params,
        "best_epoch": train.best_epoch,
        "epochs_run": train.epochs_run,
        "train_seconds": train.seconds,
        "total_seconds": time.time() - t0,
        "best_val_ce": train.best_val_ce,
        "eval": report,
    }
    result_path.write_text(json.dumps(result, indent=1, default=float))
    return result
