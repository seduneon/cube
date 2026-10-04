"""Equivariant layers, generic over any finite group.

A feature tensor has shape ``(batch, n_points, channels)``. Its ``FeatureType``
says which G-set the points form; channels are plain multiplicity (G never
mixes them). Every layer maps a FeatureType to a FeatureType:

    EquivariantLinear   in_type  -> out_type   weights tied on pair orbits
                                               (strategy: DenseOrbitMap | FactoredOrbitMap)
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
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from eqcube.symmetry.gset import FactoredPairOrbits, GSet, factored_pair_orbits, pair_orbit_labels


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


class DenseOrbitMap(nn.Module):
    """Generic strategy: materialise the full kernel from one weight per pair orbit.

    Works for any group. Cost per sample and channel pair: n_out * n_in.
    """

    pair_orbit: Tensor

    def __init__(self, in_type: FeatureType, out_type: FeatureType) -> None:
        super().__init__()
        labels, n_pairs = pair_orbit_labels(out_type.gset, in_type.gset)
        self.register_buffer("pair_orbit", torch.as_tensor(labels))
        self.weight = nn.Parameter(torch.empty(out_type.channels, in_type.channels, n_pairs))

    def kernel(self) -> Tensor:
        """The full (c_out, c_in, n_out, n_in) weight."""
        return self.weight[:, :, self.pair_orbit]

    def forward(self, x: Tensor) -> Tensor:
        return torch.einsum("bic,ocji->bjo", x, self.kernel())


class FactoredOrbitMap(nn.Module):
    """Fast strategy for product G-sets whose pair orbits factor (see GSet docs).

    Points are pairs (a, b) (e.g. position, color) and the kernel is
        K = sum_{a, b} theta[a, b] * (P_a kron Q_b),
    so we never build it: first mix the second factor with each Q_b (a tiny
    |B'| x |B| matrix), then apply one spatial kernel per b over the first
    factor. Cost per sample and channel pair: |A'| |B'| |A| n_b instead of
    |A'| |B'| |A| |B| -- cheaper whenever n_b < |B| (S6 on colors: 2 < 6).

    Special case (v1's alpha/beta trick): if the second-factor orbits are
    just "same" and "different", rewrite I, J - I as I, J. Then the J term
    is "sum over the second factor, one spatial product, broadcast".
    """

    a_orbit: Tensor
    q: Tensor

    def __init__(
        self, in_type: FeatureType, out_type: FeatureType, orbits: FactoredPairOrbits
    ) -> None:
        super().__init__()
        assert in_type.gset.factors is not None and out_type.gset.factors is not None
        self.a_in = in_type.gset.factors[0].size
        self.b_in = in_type.gset.factors[1].size
        self.a_out = out_type.gset.factors[0].size
        self.b_out = out_type.gset.factors[1].size
        self.register_buffer("a_orbit", torch.as_tensor(orbits.a_labels))
        b_labels = torch.as_tensor(orbits.b_labels)
        self.register_buffer("q", F.one_hot(b_labels, orbits.n_b).permute(2, 0, 1).float())
        self.weight = nn.Parameter(
            torch.empty(out_type.channels, in_type.channels, orbits.n_a, orbits.n_b)
        )
        # Second-factor orbits exactly {b' == b} and {b' != b} (e.g. S6 on colors)?
        eye = torch.eye(self.b_in)
        self.same_or_different = (
            orbits.n_b == 2
            and self.b_in == self.b_out
            and bool(torch.equal(self.q[0], eye) and torch.equal(self.q[1], 1 - eye))
        )

    def kernel(self) -> Tensor:
        """The full (c_out, c_in, n_out, n_in) weight (for tests; forward never builds it)."""
        b_orbit = self.q.argmax(dim=0)  # (|B'|, |B|)
        k = self.weight[:, :, self.a_orbit[:, None, :, None], b_orbit[None, :, None, :]]
        c_out, c_in = self.weight.shape[:2]
        return k.reshape(c_out, c_in, self.a_out * self.b_out, self.a_in * self.b_in)

    def forward(self, x: Tensor) -> Tensor:
        batch, _, c_in = x.shape
        x = x.view(batch, self.a_in, self.b_in, c_in)
        spatial = self.weight[:, :, self.a_orbit]  # (c_out, c_in, |A'|, |A|, n_b)
        if self.same_or_different:
            # theta_s * I + theta_d * (J - I) = (theta_s - theta_d) * I + theta_d * J,
            # and J x is a sum over the second factor: one spatial product, not |B|.
            alpha = spatial[..., 0] - spatial[..., 1]
            per_b = torch.einsum("bpkc,ocqp->bqko", x, alpha)
            summed = torch.einsum("bpc,ocqp->bqo", x.sum(dim=2), spatial[..., 1])
            out = per_b + summed[:, :, None, :]
        else:
            xq = torch.einsum("nlk,bpkc->bnplc", self.q, x)  # mix the second factor
            out = torch.einsum("bnplc,ocqpn->bqlo", xq, spatial)
        return out.reshape(batch, self.a_out * self.b_out, -1)


Impl = Literal["auto", "dense", "factored"]


class EquivariantLinear(nn.Module):
    """out[b, j, o] = sum_{i, c} theta[o, c, orbit(j, i)] * x[b, i, c] + bias[o, orbit(j)].

    Free parameters: c_out * c_in * (#pair orbits) + c_out * (#point orbits of out).
    The computation is delegated to a strategy with identical parameters:
      factored  when the pair orbits factor *and* that is cheaper (n_b < |B|)
      dense     otherwise (any group)
    """

    point_orbit: Tensor

    def __init__(
        self, in_type: FeatureType, out_type: FeatureType, bias: bool = True, impl: Impl = "auto"
    ) -> None:
        super().__init__()
        if in_type.gset.group is not out_type.gset.group:
            raise ValueError("in and out types must share one group object")
        self.in_type, self.out_type = in_type, out_type

        orbits = factored_pair_orbits(out_type.gset, in_type.gset)
        if impl == "factored" and orbits is None:
            raise ValueError("pair orbits do not factor for these feature types")
        use_factored = orbits is not None and (
            impl == "factored" or (impl == "auto" and _factoring_is_cheaper(in_type, orbits))
        )
        self.map: DenseOrbitMap | FactoredOrbitMap = (
            FactoredOrbitMap(in_type, out_type, orbits)  # type: ignore[arg-type]
            if use_factored
            else DenseOrbitMap(in_type, out_type)
        )

        bound = 1.0 / math.sqrt(in_type.channels * in_type.size)  # fan-in of one output entry
        nn.init.uniform_(self.map.weight, -bound, bound)
        point_labels, n_points = out_type.gset.orbit_labels
        self.register_buffer("point_orbit", torch.as_tensor(point_labels))
        self.bias = nn.Parameter(torch.zeros(out_type.channels, n_points)) if bias else None

    @property
    def weight(self) -> nn.Parameter:
        return self.map.weight

    def kernel(self) -> Tensor:
        return self.map.kernel()

    def forward(self, x: Tensor) -> Tensor:
        out = self.map(x)
        if self.bias is not None:
            out = out + self.bias[:, self.point_orbit].T
        return out


def _factoring_is_cheaper(in_type: FeatureType, orbits: FactoredPairOrbits) -> bool:
    assert in_type.gset.factors is not None
    return orbits.n_b < in_type.gset.factors[1].size


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
