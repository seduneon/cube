"""Model configuration, architecture registry, and presets.

A ``ModelConfig`` fully determines a model. It is stored in every
checkpoint, so loading never has to guess the architecture from weights.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Literal

from torch import nn

from eqcube.models.value_net import EquivariantBody, Layout, MLPBody, OneHotEncoder, ValueNet
from eqcube.nn.layers import FeatureType
from eqcube.puzzles import PUZZLES
from eqcube.symmetry.gset import GSet

Arch = Literal["equivariant", "mlp"]


@dataclass(frozen=True)
class ModelConfig:
    arch: Arch = "equivariant"
    puzzle: str = "pocket_cube"
    symmetry: str = "task"  # key into the puzzle's symmetry registry
    input_layout: Layout = "stickers"
    hidden_layout: Layout = "positions"
    width: int = 32  # hidden channels (equivariant) or hidden units (mlp)
    depth: int = 3  # number of hidden layers

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelConfig:
        return cls(**d)


# ── architecture registry ─────────────────────────────────────────────────

Builder = Callable[[ModelConfig], nn.Module]
ARCHITECTURES: dict[str, Builder] = {}


def register_arch(name: str) -> Callable[[Builder], Builder]:
    def wrap(fn: Builder) -> Builder:
        if name in ARCHITECTURES:
            raise ValueError(f"architecture {name!r} already registered")
        ARCHITECTURES[name] = fn
        return fn

    return wrap


def build_model(cfg: ModelConfig) -> nn.Module:
    if cfg.arch not in ARCHITECTURES:
        raise ValueError(f"unknown arch {cfg.arch!r}; known: {sorted(ARCHITECTURES)}")
    return ARCHITECTURES[cfg.arch](cfg)


@register_arch("equivariant")
def _build_equivariant(cfg: ModelConfig) -> nn.Module:
    spec = PUZZLES[cfg.puzzle]
    puzzle = spec.make()
    group = spec.symmetries[cfg.symmetry]()
    p, c = puzzle.n_positions, puzzle.n_colors

    positions = GSet.block(group, 0, p)
    colors = GSet.block(group, p, c)
    gsets = {"positions": positions, "stickers": positions.product(colors)}

    if cfg.input_layout == "positions" and not colors.is_trivial():
        raise ValueError(
            f"symmetry {cfg.symmetry!r} permutes colors, so colors cannot be channels; "
            "use input_layout='stickers'"
        )
    in_channels = c if cfg.input_layout == "positions" else 1
    in_type = FeatureType(gsets[cfg.input_layout], in_channels)
    hidden_type = FeatureType(gsets[cfg.hidden_layout], cfg.width)

    body = EquivariantBody(in_type, hidden_type, cfg.depth, spec.max_distance + 1)
    return ValueNet(OneHotEncoder(c, cfg.input_layout), body)


@register_arch("mlp")
def _build_mlp(cfg: ModelConfig) -> nn.Module:
    spec = PUZZLES[cfg.puzzle]
    puzzle = spec.make()
    in_dim = puzzle.n_positions * puzzle.n_colors
    body = MLPBody(in_dim, cfg.width, cfg.depth, spec.max_distance + 1)
    return ValueNet(OneHotEncoder(puzzle.n_colors, "stickers"), body)


# ── presets ───────────────────────────────────────────────────────────────

# All presets except "mlp" have roughly 40k parameters, so they compare
# symmetry priors at equal capacity. Fewer symmetries -> fewer tied weights,
# so the narrower widths of "color" and "task" are expected.
PRESETS: dict[str, ModelConfig] = {
    # v1 RotValueNet: O on positions, colors as channels.
    "rotation": ModelConfig(symmetry="rotation", input_layout="positions", width=28),
    # v1 BothValueNet: O x S6 on (position, color) points, k=20 blocks.
    "rotation_color": ModelConfig(
        symmetry="rotation_color", input_layout="stickers", hidden_layout="stickers", width=20
    ),
    "color": ModelConfig(symmetry="color", width=5),
    # The computed task symmetry (S3: rotations + mirrors about DBL-UFR, with recoloring).
    "task": ModelConfig(symmetry="task", width=12),
    "mlp": ModelConfig(arch="mlp", symmetry="trivial", width=256),
    "mlp_small": ModelConfig(arch="mlp", symmetry="trivial", width=110),
}
