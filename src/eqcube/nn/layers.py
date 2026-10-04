"""Equivariant layers, generic over any finite group.

A feature tensor has shape ``(batch, n_points, channels)``. Its ``FeatureType``
says which G-set the points form; channels are plain multiplicity (G never
mixes them). Every layer maps a FeatureType to a FeatureType:

    EquivariantLinear   in_type  -> out_type   weights tied on pair orbits
    PointwiseNorm       t        -> t          LayerNorm over channels, per point
    ResidualBlock       t        -> t          x + relu(linear(norm(x)))
    InvariantPool       t        -> R^d        mean over each point orbit

Why each is equivariant (g acts by permuting the point axis):
  * linear:   W[j, i] depends only on the orbit of (j, i), so W commutes
              with the permutation (it is a G-invariant kernel);
  * relu/norm: act on each point independently with shared parameters;
  * pool:     averaging over an orbit is unchanged when g permutes it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from eqcube.symmetry.gset import GSet, pair_orbit_labels


@dataclass(frozen=True, eq=False)
class FeatureType:
    """Features on the points of ``gset``, with ``channels`` copies each."""

    gset: GSet
    channels: int

    @property
    def size(self) -> int:
        return self.gset.size

    def check(self, x: Tensor) -> None:
        if x.shape[1:] != (self.size, self.channels):
            raise ValueError(f"expected (B, {self.size}, {self.channels}), got {tuple(x.shape)}")


class EquivariantLinear(nn.Module):
    """out[b, j, o] = sum_{i, c} theta[o, c, orbit(j, i)] * x[b, i, c] + bias[o, orbit(j)].

    Free parameters: c_out * c_in * (#pair orbits) + c_out * (#point orbits of out).
    """

    pair_orbit: Tensor
    point_orbit: Tensor

    def __init__(self, in_type: FeatureType, out_type: FeatureType, bias: bool = True) -> None:
        super().__init__()
        if in_type.gset.group is not out_type.gset.group:
            raise ValueError("in and out types must share one group object")
        self.in_type, self.out_type = in_type, out_type

        pair_labels, n_pairs = pair_orbit_labels(out_type.gset, in_type.gset)
        point_labels, n_points = out_type.gset.orbit_labels
        self.register_buffer("pair_orbit", torch.as_tensor(pair_labels))
        self.register_buffer("point_orbit", torch.as_tensor(point_labels))

        c_in, c_out = in_type.channels, out_type.channels
        bound = 1.0 / math.sqrt(c_in * in_type.size)  # fan-in of one output entry
        self.weight = nn.Parameter(torch.empty(c_out, c_in, n_pairs).uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.zeros(c_out, n_points)) if bias else None

    def kernel(self) -> Tensor:
        """The full (c_out, c_in, n_out, n_in) weight, materialised from orbit weights."""
        return self.weight[:, :, self.pair_orbit]

    def forward(self, x: Tensor) -> Tensor:
        out = torch.einsum("bic,ocji->bjo", x, self.kernel())
        if self.bias is not None:
            out = out + self.bias[:, self.point_orbit].T
        return out


class PointwiseNorm(nn.Module):
    """LayerNorm over the channels of each point; affine shared by all points."""

    def __init__(self, ftype: FeatureType) -> None:
        super().__init__()
        if ftype.channels < 2:
            raise ValueError("normalising over a single channel would zero it out")
        self.norm = nn.LayerNorm(ftype.channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x)


class ResidualBlock(nn.Module):
    """Pre-norm residual block: x + relu(linear(norm(x)))."""

    def __init__(self, ftype: FeatureType) -> None:
        super().__init__()
        self.norm = PointwiseNorm(ftype)
        self.linear = EquivariantLinear(ftype, ftype)

    def forward(self, x: Tensor) -> Tensor:
        return x + F.relu(self.linear(self.norm(x)))


class InvariantPool(nn.Module):
    """Mean over every point orbit: (B, n, c) -> (B, n_orbits * c), G-invariant.

    For a transitive action this is a plain mean over points. For the trivial
    group every point is its own orbit, so nothing is pooled.
    """

    avg: Tensor

    def __init__(self, ftype: FeatureType) -> None:
        super().__init__()
        labels, n_orbits = ftype.gset.orbit_labels
        avg = torch.zeros(n_orbits, ftype.size)
        avg[torch.as_tensor(labels), torch.arange(ftype.size)] = 1.0
        avg = avg / avg.sum(dim=1, keepdim=True)
        self.register_buffer("avg", avg)
        self.out_dim = n_orbits * ftype.channels

    def forward(self, x: Tensor) -> Tensor:
        pooled = torch.einsum("kn,bnc->bkc", self.avg, x)
        return pooled.flatten(1)
