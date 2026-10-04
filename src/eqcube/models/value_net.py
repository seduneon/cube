"""Value networks: states (B, P) int -> distance-class logits (B, num_classes).

Every model is ``ValueNet(encoder, body)`` (template: encode, then body), so
search and training never care which architecture they hold.
"""

from __future__ import annotations

from typing import Literal

import torch.nn.functional as F
from torch import Tensor, nn

from eqcube.nn.layers import EquivariantLinear, FeatureType, InvariantPool, ResidualBlock

Layout = Literal["stickers", "positions"]


class OneHotEncoder(nn.Module):
    """One-hot colors, laid out for a FeatureType.

    stickers:  (B, P * C, 1)  point = (position, color), index p * C + c
    positions: (B, P, C)      point = position, colors are channels
    """

    def __init__(self, n_colors: int, layout: Layout) -> None:
        super().__init__()
        self.n_colors, self.layout = n_colors, layout

    def forward(self, states: Tensor) -> Tensor:
        x = F.one_hot(states.long(), self.n_colors).float()  # (B, P, C)
        if self.layout == "stickers":
            return x.reshape(x.shape[0], -1, 1)
        return x


class EquivariantBody(nn.Module):
    """lift -> residual blocks -> invariant pool -> linear head."""

    def __init__(
        self, in_type: FeatureType, hidden_type: FeatureType, depth: int, num_classes: int
    ) -> None:
        super().__init__()
        self.in_type = in_type
        self.lift = EquivariantLinear(in_type, hidden_type)
        self.blocks = nn.Sequential(*(ResidualBlock(hidden_type) for _ in range(depth - 1)))
        self.pool = InvariantPool(hidden_type)
        self.head = nn.Linear(self.pool.out_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        self.in_type.check(x)
        x = F.relu(self.lift(x))
        x = self.blocks(x)
        return self.head(self.pool(x))


class MLPBody(nn.Module):
    """Unconstrained baseline: flatten -> (linear, relu) x depth -> linear."""

    def __init__(self, in_dim: int, width: int, depth: int, num_classes: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.ReLU()]
            d = width
        layers.append(nn.Linear(d, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x.flatten(1))


class ValueNet(nn.Module):
    def __init__(self, encoder: nn.Module, body: nn.Module) -> None:
        super().__init__()
        self.encoder, self.body = encoder, body

    def forward(self, states: Tensor) -> Tensor:
        return self.body(self.encoder(states))
