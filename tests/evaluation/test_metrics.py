"""Metrics against oracles with known answers."""

import numpy as np
import pytest
import torch

from eqcube.data import DistanceTable, Partition, per_depth_suite
from eqcube.evaluation import equivariance_error, sibling_ranking, solve_metrics, value_metrics
from eqcube.models import PRESETS, NetPredictor, build_model
from eqcube.puzzles.pocket_coords import pocket_coords
from eqcube.puzzles.pocket_cube import task_group
from eqcube.search import greedy


@pytest.fixture(scope="module")
def table() -> DistanceTable:
    return DistanceTable.build(pocket_coords())


@pytest.fixture(scope="module")
def index(table: DistanceTable) -> np.ndarray:
    suite = per_depth_suite(Partition(table), 40, np.random.default_rng(0))
    return suite.index


def onehot_logits(labels: np.ndarray, scale: float = 10.0) -> np.ndarray:
    logits = np.zeros((len(labels), 15))
    logits[np.arange(len(labels)), labels] = scale
    return logits


def test_value_metrics_perfect_and_uniform() -> None:
    labels = np.array([0, 3, 7, 14, 14])
    perfect = value_metrics(onehot_logits(labels), labels)
    assert perfect["accuracy"] == 1.0 and perfect["mae"] == 0.0
    assert perfect["ce"] < 1e-3
    uniform = value_metrics(np.zeros((5, 15)), labels)
    assert uniform["ce"] == pytest.approx(np.log(15))
    assert uniform["mae_expected"] == pytest.approx(np.mean(np.abs(7 - labels)))
    assert set(perfect["per_depth"]["accuracy"]) == {0, 3, 7, 14}


def test_sibling_ranking_oracle(table: DistanceTable, index: np.ndarray) -> None:
    oracle = sibling_ranking(lambda s: table.distance(s).astype(float), table, index)
    assert oracle["pair_accuracy"] == 1.0 and oracle["best_child"] == 1.0
    anti = sibling_ranking(lambda s: -table.distance(s).astype(float), table, index)
    assert anti["pair_accuracy"] == 0.0
    # The anti-oracle picks an up child whenever one exists. Local maxima (every
    # child closer to solved) occur at depths 11-14, and there any pick is correct.
    parents = index[table.dist[index] >= 1]
    children = table.indexer.expand(parents)
    local_max = (table.dist[children] < table.dist[parents, None]).all(axis=1)
    assert local_max.any()
    assert anti["best_child"] == pytest.approx(local_max.mean())


def test_sibling_ranking_constant_is_chance(table: DistanceTable, index: np.ndarray) -> None:
    flat = sibling_ranking(lambda s: np.zeros(len(s)), table, index)
    assert flat["pair_accuracy"] == 0.5
    assert 0.0 < flat["best_child"] < 1.0  # = fraction of down children


def test_equivariance_error(table: DistanceTable, index: np.ndarray) -> None:
    torch.manual_seed(0)
    states = table.indexer.unrank(index[:100])
    task = equivariance_error(NetPredictor(build_model(PRESETS["task"])), states, task_group())
    mlp = equivariance_error(NetPredictor(build_model(PRESETS["mlp_small"])), states, task_group())
    assert task["max"] < 1e-4
    assert mlp["max"] > 1e-3


def test_solve_metrics_with_oracle(table: DistanceTable, index: np.ndarray) -> None:
    oracle = greedy(lambda s: table.distance(s).astype(float), max_steps=20)
    report = solve_metrics(oracle, table, index)
    assert report["solve_rate"] == 1.0
    assert report["mean_excess"] == 0.0
    assert report["per_depth"]["solve_rate"][14] == 1.0


def test_solve_metrics_reports_failures(table: DistanceTable, index: np.ndarray) -> None:
    hopeless = greedy(lambda s: np.zeros(len(s)), max_steps=2)
    report = solve_metrics(hopeless, table, index[table.dist[index] >= 5])
    assert report["solve_rate"] == 0.0
    assert np.isnan(report["mean_excess"])
