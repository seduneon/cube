"""Every layer commutes with the group: L(g·x) == g·L(x)."""

import numpy as np
import pytest
import torch

from eqcube.nn import EquivariantLinear, FeatureType, InvariantPool, PointwiseNorm, ResidualBlock
from eqcube.puzzles.pocket_cube import SYMMETRIES
from eqcube.symmetry.gset import GSet, pair_orbit_labels
from eqcube.symmetry.perm import Perm, PermGroup

torch.manual_seed(0)
GROUPS = ["rotation", "color", "rotation_color", "task"]


def gsets(name: str) -> dict[str, GSet]:
    g = SYMMETRIES[name]()
    pos = GSet.block(g, 0, 24)
    return {"positions": pos, "stickers": pos.product(GSet.block(g, 24, 6))}


def some_elements(group: PermGroup, n: int = 6) -> list[Perm]:
    rng = np.random.default_rng(0)
    elements = sorted(group.elements, key=lambda g: g.arr)
    return [
        elements[i] for i in rng.choice(len(elements), size=min(n, len(elements)), replace=False)
    ]


def act(gset: GSet, g: Perm, x: torch.Tensor) -> torch.Tensor:
    return x[:, torch.as_tensor(gset.gather_index(g))]


@pytest.mark.parametrize("name", GROUPS)
@pytest.mark.parametrize(
    ("src", "dst"),
    [
        ("stickers", "positions"),
        ("positions", "positions"),
        ("stickers", "stickers"),
        ("positions", "stickers"),
    ],
)
def test_linear_is_equivariant(name: str, src: str, dst: str) -> None:
    gs = gsets(name)
    layer = EquivariantLinear(FeatureType(gs[src], 3), FeatureType(gs[dst], 4)).double()
    with torch.no_grad():
        layer.bias.normal_()  # type: ignore[union-attr]
    x = torch.randn(5, gs[src].size, 3, dtype=torch.float64)
    for g in some_elements(SYMMETRIES[name]()):
        torch.testing.assert_close(layer(act(gs[src], g, x)), act(gs[dst], g, layer(x)))


@pytest.mark.parametrize("name", GROUPS)
def test_block_norm_and_pool(name: str) -> None:
    gs = gsets(name)
    t = FeatureType(gs["positions"], 4)
    block, norm, pool = ResidualBlock(t).double(), PointwiseNorm(t).double(), InvariantPool(t)
    x = torch.randn(5, 24, 4, dtype=torch.float64)
    for g in some_elements(SYMMETRIES[name]()):
        gx = act(gs["positions"], g, x)
        torch.testing.assert_close(block(gx), act(gs["positions"], g, block(x)))
        torch.testing.assert_close(norm(gx), act(gs["positions"], g, norm(x)))
        torch.testing.assert_close(pool(gx.float()), pool(x.float()))


def test_parameter_count_is_orbit_count() -> None:
    gs = gsets("rotation_color")
    layer = EquivariantLinear(FeatureType(gs["stickers"], 3), FeatureType(gs["stickers"], 5))
    _, n_pairs = pair_orbit_labels(gs["stickers"], gs["stickers"])
    assert n_pairs == 48  # 24 spatial pair orbits x {same color, different color}
    assert layer.weight.numel() == 5 * 3 * 48


def test_trivial_group_linear_is_dense() -> None:
    gs = gsets("trivial") if "trivial" in SYMMETRIES else None
    assert gs is not None
    layer = EquivariantLinear(FeatureType(gs["positions"], 2), FeatureType(gs["positions"], 2))
    assert layer.weight.numel() == 2 * 2 * 24 * 24


def test_pool_out_dim_counts_orbits() -> None:
    assert InvariantPool(FeatureType(gsets("rotation")["positions"], 7)).out_dim == 7
    # mirrors fix the stickers lying on their planes, so S3 has 6 (not 24/6) orbits
    assert InvariantPool(FeatureType(gsets("task")["positions"], 7)).out_dim == 6 * 7


def test_mismatched_groups_rejected() -> None:
    a, b = gsets("rotation"), gsets("task")
    with pytest.raises(ValueError):
        EquivariantLinear(FeatureType(a["positions"], 2), FeatureType(b["positions"], 2))


def test_norm_rejects_single_channel() -> None:
    with pytest.raises(ValueError):
        PointwiseNorm(FeatureType(gsets("rotation")["positions"], 1))
