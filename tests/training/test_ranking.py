"""Sibling ranking loss: hand values, invariances, and the table oracle."""

import math

import numpy as np
import pytest
import torch

from eqcube.data import DistanceTable
from eqcube.models import PRESETS, build_model
from eqcube.puzzles.pocket_coords import pocket_coords
from eqcube.training.ranking import SiblingRanking, listwise_loss, sibling_ranking_loss

C = pocket_coords()


@pytest.fixture(scope="module")
def table() -> DistanceTable:
    return DistanceTable.build(C)


def test_listwise_by_hand() -> None:
    scores = torch.tensor([[1.0, 3.0]])
    down = torch.tensor([[True, False]])
    expected = math.log(1 + math.exp(-2.0))  # -log(e^-1 / (e^-1 + e^-3))
    assert listwise_loss(scores, down).item() == pytest.approx(expected)
    assert listwise_loss(scores, down, tau=2.0).item() == pytest.approx(math.log(1 + math.exp(-1)))


def test_all_down_costs_nothing() -> None:
    scores = torch.randn(4, 6)
    assert listwise_loss(scores, torch.ones(4, 6, dtype=torch.bool)).item() == pytest.approx(0.0)


def test_shared_shift_gets_no_gradient() -> None:
    """Adding a constant to every sibling's score changes nothing; gradients sum to 0."""
    scores = torch.randn(32, 6, requires_grad=True)
    down = torch.rand(32, 6) < 0.4
    down[:, 0] = True
    loss = listwise_loss(scores, down)
    shifted = listwise_loss(scores + torch.randn(32, 1) * 5, down)
    assert loss.item() == pytest.approx(shifted.item(), abs=1e-5)
    loss.backward()
    assert scores.grad is not None
    torch.testing.assert_close(scores.grad.sum(dim=1), torch.zeros(32), atol=1e-6, rtol=0)


def test_pushes_down_children_lower() -> None:
    scores = torch.zeros(1, 6, requires_grad=True)
    down = torch.tensor([[True, True, False, False, False, False]])
    listwise_loss(scores, down).backward()
    assert scores.grad is not None
    assert (scores.grad[0, :2] > 0).all() and (scores.grad[0, 2:] < 0).all()  # descent lowers down


def test_oracle_distributions_give_small_loss(table: DistanceTable) -> None:
    index = np.random.default_rng(0).choice(np.flatnonzero(table.dist >= 1), 300)
    children = C.expand(index)
    labels = torch.as_tensor(table.labels(children.reshape(-1)))
    lp = torch.full((len(labels), 15), -30.0)
    lp[torch.arange(len(labels)), labels] = 0.0
    lp = lp.log_softmax(-1).view(300, 6, 15)
    down = torch.as_tensor(table.dist[children] < table.dist[index, None])
    assert sibling_ranking_loss(lp, down, tau=0.1).item() < 1e-6


def test_sampler_matches_the_table(table: DistanceTable) -> None:
    train = np.random.default_rng(1).choice(C.n_states, 5000, replace=False)
    train[0] = C.solved_index  # must never be used as a parent
    term = SiblingRanking(table, train, 128, 1.0, torch.device("cpu"), 0)
    assert C.solved_index not in term.parents
    states, down = term.sample()
    assert states.shape == (128 * 6, 24) and down.shape == (128, 6)
    assert down.any(dim=1).all()  # every parent has a down child
    child_d = table.distance(states.numpy()).reshape(128, 6)
    parent_d = child_d.min(axis=1) + 1  # some child is at d - 1
    np.testing.assert_array_equal(down.numpy(), child_d < parent_d[:, None])


def test_loss_backpropagates_into_the_model(table: DistanceTable) -> None:
    torch.manual_seed(0)
    model = build_model(PRESETS["task"])
    train = np.random.default_rng(2).choice(C.n_states, 2000, replace=False)
    term = SiblingRanking(table, train, 32, 1.0, torch.device("cpu"), 0)
    loss = term.loss(model)
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
