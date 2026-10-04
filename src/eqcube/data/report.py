"""Per-depth view of the state space and the data built from it.

    python -m eqcube.data.report [--cache results/cache] [--results results/pilot]

Three tables, one row per depth:
  structure  how many states, how many moves go closer, local maxima,
             symmetric states, how often random walks land there
  data       split pools, and how many states each training recipe takes
  results    (optional) per-depth metrics of trained runs, side by side
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from eqcube.data.partition import Partition, PartitionConfig
from eqcube.data.samplers import RandomWalkSampler, TemperedSampler
from eqcube.data.symmetry import orbit_representatives
from eqcube.data.table import DistanceTable
from eqcube.puzzles import PUZZLES

IntArray = npt.NDArray[np.int64]
Row = dict[str, Any]


def structure_rows(
    table: DistanceTable,
    orbit_reps: IntArray | None = None,
    walk_samples: int = 200_000,
    walk_length: int = 20,
    seed: int = 0,
) -> list[Row]:
    """Shape of the state space at each depth."""
    dist = table.dist
    n = len(dist)
    down = np.zeros(n, dtype=np.int64)
    for start in range(0, n, 1_000_000):  # chunked: (chunk, 6) children at a time
        index = np.arange(start, min(start + 1_000_000, n))
        down[index] = (dist[table.indexer.expand(index)] < dist[index, None]).sum(axis=1)
    n_moves = table.indexer.puzzle.n_moves

    walk_depth = np.zeros(table.max_distance + 1)
    if walk_samples:
        everything = Partition(table, PartitionConfig(fractions=(1.0, 0.0, 0.0)))
        walks = RandomWalkSampler(everything, walk_length).sample(
            walk_samples, np.random.default_rng(seed)
        )
        walk_depth = np.bincount(dist[walks], minlength=table.max_distance + 1) / walk_samples

    symmetric = None
    if orbit_reps is not None:
        orbit_size = np.bincount(orbit_reps, minlength=n)[orbit_reps]
        symmetric = orbit_size < orbit_size.max()  # fixed by some non-identity symmetry

    rows: list[Row] = []
    for d in range(table.max_distance + 1):
        at = dist == d
        row: Row = {
            "depth": d,
            "states": int(at.sum()),
            "share": float(at.mean()),
            "cumulative": float((dist <= d).mean()),
            "moves_closer": float(down[at].mean()),
            "local_max": float((down[at] == n_moves).mean()),
            "walk_share": float(walk_depth[d]),
        }
        if symmetric is not None:
            row["symmetric"] = float(symmetric[at].mean())
        rows.append(row)
    return rows


def data_rows(partition: Partition, sizes: Sequence[int], alphas: Sequence[float]) -> list[Row]:
    """Split pools and per-depth training quotas for every (size, alpha) recipe."""
    pools = {s: partition.pool_sizes(s) for s in ("train", "val", "test")}
    quotas = {(n, a): TemperedSampler(partition, a).depth_quota(n) for n in sizes for a in alphas}
    rows: list[Row] = []
    for d in range(partition.table.max_distance + 1):
        row: Row = {
            "depth": d,
            "shared": d <= partition.config.shared_max_depth,
            **{f"{s}_pool": int(p[d]) for s, p in pools.items()},
        }
        for (n, a), q in quotas.items():
            row[f"n{n // 1000}k_a{a:g}"] = int(q[d])
            row[f"n{n // 1000}k_a{a:g}_cover"] = float(q[d] / max(pools["train"][d], 1))
        rows.append(row)
    return rows


def result_rows(results_dir: str | Path) -> list[Row]:
    """Per-depth metrics of every run under ``results_dir``, one column group per run."""
    runs = [json.loads(p.read_text()) for p in sorted(Path(results_dir).rglob("result.json"))]
    depths = sorted(
        {int(d) for r in runs for d in r["eval"]["test/per_depth"]["per_depth"]["accuracy"]}
    )
    rows: list[Row] = []
    for d in depths:
        row: Row = {"depth": d}
        for r in runs:
            name = r.get("name", r["preset"])
            e = r["eval"]
            row[f"{name}:acc"] = e["test/per_depth"]["per_depth"]["accuracy"].get(str(d))
            row[f"{name}:mae"] = e["test/per_depth"]["per_depth"]["mae"].get(str(d))
            row[f"{name}:best_child"] = e["siblings/expected"]["per_depth"]["best_child"].get(
                str(d)
            )
            row[f"{name}:solve"] = e["search/expected"]["per_depth"]["solve_rate"].get(str(d))
        rows.append(row)
    return rows


def markdown(rows: list[Row], percent: Sequence[str] = ()) -> str:
    """Render rows as a markdown table; columns in ``percent`` are shown as %."""
    if not rows:
        return "(empty)"
    cols = list(rows[0])

    def fmt(key: str, v: Any) -> str:
        if v is None:
            return "—"
        if isinstance(v, bool):
            return "yes" if v else ""
        if isinstance(v, float):
            if key.split(":")[-1] in percent or key.endswith(
                ("_cover", "share", "local_max", "symmetric", "cumulative")
            ):
                return f"{100 * v:.1f}%" if v >= 0.001 or v == 0.0 else f"{100 * v:.3f}%"
            return f"{v:.2f}"
        if isinstance(v, int):
            return f"{v:,}"
        return str(v)

    lines = ["| " + " | ".join(cols) + " |", "|" + "---:|" * len(cols)]
    lines += ["| " + " | ".join(fmt(c, r.get(c)) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--cache", default="results/cache")
    ap.add_argument("--results", default=None, help="directory of result.json files")
    ap.add_argument("--sizes", nargs="+", type=int, default=[50_000, 200_000, 1_000_000])
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.5, 1.0])
    args = ap.parse_args(argv)

    spec = PUZZLES["pocket_cube"]
    from eqcube.puzzles.pocket_coords import pocket_coords

    table = DistanceTable.build(pocket_coords(), args.cache)
    reps = orbit_representatives(
        table.indexer,
        spec.symmetries["task"](),
        cache_path=Path(args.cache) / "task_orbit_reps.npy",
    )
    partition = Partition(table)

    print("## State space by depth\n")
    print(markdown(structure_rows(table, reps)))
    print("\n## Splits and training quotas by depth\n")
    print(markdown(data_rows(partition, args.sizes, args.alphas)))
    if args.results:
        print(f"\n## Results by depth ({args.results})\n")
        print(markdown(result_rows(args.results), percent=["acc", "best_child", "solve"]))


if __name__ == "__main__":
    main()
