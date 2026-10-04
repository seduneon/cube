"""Every layer commutes with the group: L(g·x) == g·L(x)."""

import numpy as np
import pytest
import torch

from eqcube.nn import EquivariantLinear, FeatureType, InvariantPool, PointwiseNorm, ResidualBlock
from eqcube.nn.layers import DenseOrbitMap
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


# ── factored strategy ─────────────────────────────────────────────────────

FACTORABLE = ["trivial", "rotation", "color", "rotation_color", "isometry_color"]


def test_factorization_criterion() -> None:
    from eqcube.symmetry.gset import factored_pair_orbits

    st = gsets("rotation_color")["stickers"]
    orbits = factored_pair_orbits(st, st)
    assert orbits is not None and (orbits.n_a, orbits.n_b) == (24, 2)
    task = gsets("task")["stickers"]
    assert factored_pair_orbits(task, task) is None  # diagonal S3 couples position and color
    pos = gsets("rotation_color")["positions"]
    assert factored_pair_orbits(pos, pos) is None  # not built as a product


@pytest.mark.parametrize(
    ("name", "expected"),
    [("rotation_color", "FactoredOrbitMap"), ("color", "FactoredOrbitMap"),
     ("rotation", "DenseOrbitMap"), ("trivial", "DenseOrbitMap"), ("task", "DenseOrbitMap")],
)  # fmt: skip
def test_auto_picks_cheaper_strategy(name: str, expected: str) -> None:
    t = FeatureType(gsets(name)["stickers"], 2)
    assert type(EquivariantLinear(t, t).map).__name__ == expected


def test_forcing_factored_on_non_factoring_group_fails() -> None:
    t = FeatureType(gsets("task")["stickers"], 2)
    with pytest.raises(ValueError, match="do not factor"):
        EquivariantLinear(t, t, impl="factored")


@pytest.mark.parametrize("name", FACTORABLE)
def test_factored_matches_dense_exactly(name: str) -> None:
    """Same function space: copy factored weights into a dense layer, same outputs."""
    st = gsets(name)["stickers"]
    t_in, t_out = FeatureType(st, 3), FeatureType(st, 2)
    fact = EquivariantLinear(t_in, t_out, impl="factored").double()
    dense = EquivariantLinear(t_in, t_out, impl="dense").double()
    assert fact.weight.numel() == dense.weight.numel()

    kernel = fact.kernel()
    assert isinstance(dense.map, DenseOrbitMap)
    labels = dense.map.pair_orbit
    with torch.no_grad():
        flat = kernel.reshape(*kernel.shape[:2], -1)
        dense.weight.zero_()
        dense.weight[:, :, labels.reshape(-1)] = flat  # every orbit entry is equal
        assert torch.equal(dense.map.kernel(), kernel)  # kernel is constant on dense orbits
        fact.bias.normal_()  # type: ignore[union-attr]
        dense.bias.copy_(fact.bias)  # type: ignore[arg-type]
    x = torch.randn(4, 144, 3, dtype=torch.float64)
    torch.testing.assert_close(fact(x), dense(x))


@pytest.mark.parametrize("name", FACTORABLE)
def test_factored_is_equivariant(name: str) -> None:
    gs = gsets(name)
    t = FeatureType(gs["stickers"], 3)
    layer = EquivariantLinear(t, t, impl="factored").double()
    x = torch.randn(4, 144, 3, dtype=torch.float64)
    for g in some_elements(SYMMETRIES[name]()):
        torch.testing.assert_close(
            layer(act(gs["stickers"], g, x)), act(gs["stickers"], g, layer(x))
        )


def test_same_or_different_fast_path_only_for_two_transitive_colors() -> None:
    def fast(name: str) -> bool:
        t = FeatureType(gsets(name)["stickers"], 2)
        return EquivariantLinear(t, t, impl="factored").map.same_or_different  # type: ignore[union-attr]

    assert fast("rotation_color") and fast("color") and fast("isometry_color")
    assert not fast("rotation") and not fast("trivial")
