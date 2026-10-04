"""Output heads: valid distributions, CORN structure, unchanged checkpoint keys."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from eqcube.models import PRESETS, build_model
from eqcube.models.heads import CategoricalHead, OrdinalHead, make_head
from eqcube.puzzles.pocket_coords import pocket_coords
from eqcube.puzzles.pocket_cube import task_group
from eqcube.symmetry.action import FunctionAction

torch.manual_seed(0)


def test_ordinal_head_is_a_distribution_with_decreasing_survival() -> None:
    head = OrdinalHead(8, 15)
    lp = head(torch.randn(32, 8))
    assert lp.shape == (32, 15)
    torch.testing.assert_close(lp.exp().sum(dim=1), torch.ones(32))
    survival = 1 - lp.exp().cumsum(dim=1)[:, :-1]
    assert (survival[:, 1:] <= survival[:, :-1] + 1e-6).all()


def test_ordinal_head_matches_corn_formula() -> None:
    head = OrdinalHead(4, 5)
    x = torch.randn(3, 4)
    z = torch.nn.functional.linear(x, head.weight, head.bias)  # conditional logits
    cond = torch.sigmoid(z)  # P(d > k | d > k-1)
    s = torch.cumprod(cond, dim=1)  # S(0..3)
    s_prev = torch.cat([torch.ones(3, 1), s[:, :-1]], dim=1)
    expected = torch.cat([s_prev - s, s[:, -1:]], dim=1)
    torch.testing.assert_close(head(x).exp(), expected)


def test_corn_nll_equals_sum_of_conditional_bces() -> None:
    """NLL of the ordinal head is exactly the CORN loss for the true class."""
    head = OrdinalHead(4, 6)
    x, label = torch.randn(1, 4), 3
    z = torch.nn.functional.linear(x, head.weight, head.bias)[0]
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    corn = sum(bce(z[j], torch.tensor(1.0)) for j in range(label)) + bce(
        z[label], torch.tensor(0.0)
    )
    torch.testing.assert_close(-head(x)[0, label], corn)


def test_categorical_head_keeps_checkpoint_keys() -> None:
    model = build_model(PRESETS["task"])
    keys = set(model.state_dict())
    assert "body.head.weight" in keys and "body.head.bias" in keys
    assert isinstance(model.body.head, CategoricalHead)  # type: ignore[union-attr]


def test_unknown_head_rejected() -> None:
    with pytest.raises(ValueError):
        make_head("nope", 4, 5)  # type: ignore[arg-type]


@pytest.mark.parametrize("parity", [False, True])
def test_ordinal_task_model_is_invariant(parity: bool) -> None:
    model = build_model(replace(PRESETS["task"], head="ordinal", parity_mask=parity)).eval()
    c = pocket_coords()
    states = c.unrank(np.random.default_rng(0).integers(0, c.n_states, 32))
    action = FunctionAction(task_group(), n_domain=24)
    with torch.no_grad():
        base = model(torch.as_tensor(states))
        assert torch.allclose(base.exp().sum(1), torch.ones(32), atol=1e-4)
        for g in task_group().elements:
            moved = np.array([action(g, tuple(int(v) for v in s)) for s in states], dtype=np.int8)
            torch.testing.assert_close(model(torch.as_tensor(moved)), base, atol=1e-3, rtol=1e-4)
