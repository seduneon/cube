"""Value networks: states (B, P) int -> distance-class log-probabilities (B, num_classes).

Every model is ``ValueNet(encoder, body)`` (template: encode, then body), so
search and training never care which architecture they hold.
"""

from __future__ import annotations

from typing import Literal

import torch.nn.functional as F
from torch import Tensor, nn

from eqcube.models.heads import HeadName, make_head
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
    """lift -> residual blocks -> invariant pool -> head (log-probabilities)."""

    def __init__(
        self,
        in_type: FeatureType,
        hidden_type: FeatureType,
        depth: int,
        num_classes: int,
        head: HeadName = "categorical",
    ) -> None:
        super().__init__()
        self.in_type = in_type
        self.lift = EquivariantLinear(in_type, hidden_type)
        self.blocks = nn.Sequential(*(ResidualBlock(hidden_type) for _ in range(depth - 1)))
        self.pool = InvariantPool(hidden_type)
        self.head = make_head(head, self.pool.out_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        self.in_type.check(x)
        x = F.relu(self.lift(x))
        x = self.blocks(x)
        return self.head(self.pool(x))


class MLPBody(nn.Module):
    """Unconstrained baseline: flatten -> (linear, relu) x depth -> head."""

    def __init__(
        self, in_dim: int, width: int, depth: int, num_classes: int, head: HeadName = "categorical"
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.ReLU()]
            d = width
        layers.append(make_head(head, d, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x.flatten(1))


class ValueNet(nn.Module):
    """log_probs = output_filter(states, body(encoder(states))); the filter is optional."""

    def __init__(
        self, encoder: nn.Module, body: nn.Module, output_filter: nn.Module | None = None
    ) -> None:
        super().__init__()
        self.encoder, self.body, self.output_filter = encoder, body, output_filter

    def forward(self, states: Tensor) -> Tensor:
        log_probs = self.body(self.encoder(states))
        if self.output_filter is not None:
            log_probs = self.output_filter(states, log_probs)
        return log_probs
