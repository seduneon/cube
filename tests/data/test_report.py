"""Per-depth report tables are consistent with the table and samplers."""

import json
from pathlib import Path

import numpy as np

from eqcube.data import DistanceTable, Partition, TemperedSampler
from eqcube.data.report import data_rows, markdown, result_rows, structure_rows


def test_structure_rows(table: DistanceTable) -> None:
    rows = structure_rows(table, walk_samples=5000)
    assert [r["depth"] for r in rows] == list(range(15))
    assert sum(r["states"] for r in rows) == table.indexer.n_states
    assert rows[-1]["cumulative"] == 1.0
    assert rows[14]["local_max"] == 1.0 and rows[1]["local_max"] == 0.0
    assert rows[1]["moves_closer"] == 1.0  # only the inverse move solves a 1-move scramble
    assert abs(sum(r["walk_share"] for r in rows) - 1.0) < 1e-9


def test_data_rows_match_sampler(partition: Partition) -> None:
    rows = data_rows(partition, [50_000], [0.5])
    quota = TemperedSampler(partition, 0.5).depth_quota(50_000)
    assert [r["n50k_a0.5"] for r in rows] == quota.tolist()
    assert sum(r["n50k_a0.5"] for r in rows) == 50_000
    assert all(r["shared"] == (r["depth"] <= 4) for r in rows)
    np.testing.assert_array_equal([r["train_pool"] for r in rows], partition.pool_sizes("train"))


def test_result_rows_and_markdown(tmp_path: Path) -> None:
    per_depth = {"accuracy": {"3": 0.5}, "mae": {"3": 1.0}}
    run = {
        "preset": "task",
        "eval": {
            "test/per_depth": {"per_depth": per_depth},
            "siblings/expected": {"per_depth": {"best_child": {"3": 0.9}}},
            "search/expected": {"per_depth": {"solve_rate": {}}},
        },
    }
    (tmp_path / "r").mkdir()
    (tmp_path / "r" / "result.json").write_text(json.dumps(run))
    rows = result_rows(tmp_path)
    assert rows == [{"depth": 3, "task:acc": 0.5, "task:mae": 1.0, "task:best_child": 0.9,
                     "task:solve": None}]  # fmt: skip
    text = markdown(rows, percent=["acc", "best_child"])
    assert "50.0%" in text and "90.0%" in text and "—" in text
