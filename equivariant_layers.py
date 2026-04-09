"""
Pure PyTorch equivariant layers for the 2x2x2 Pocket Cube.

Spatial-only (emlp_rot):
  GroupConvLayer    (batch, 24, c_in) -> (batch, 24, c_out)
  InvariantLinear   (batch, 24, c_in) -> (batch, 1)   [average over positions]

Spatial + color (emlp_both):
  BothConvLayer  (batch, 24, 6*k_in) -> (batch, 24, 6*k_out)
  InvariantHead  (batch, 24, 6*k_in) -> (batch, 1)

Color-only S6 (emlp_col):
  ColorConvLayer  (batch, 24, 6*k_in) -> (batch, 24, 6*k_out)
  InvariantHead   (batch, 24, 6*k_in) -> (batch, 1)   [shared with above]

Both color-equivariant families use InvariantHead and require hidden channel
counts to be multiples of 6 (one block per copy of the S6 rep).
"""

import numpy as np
import torch
import torch.nn as nn

from cube_symmetry import (
    ALL_ROTATIONS,
    compute_spatial_pair_orbits,
    compute_color_pair_orbits,
)


# ─── Orbit computation (kept for backward compatibility) ─────────────────────

def compute_pair_orbits():
    """Alias for compute_spatial_pair_orbits() — kept for backward compatibility."""
    return compute_spatial_pair_orbits()


# ─── Spatial-only layers ─────────────────────────────────────────────────────

class GroupConvLayer(nn.Module):
    """Equivariant linear layer: (batch, 24, c_in) -> (batch, 24, c_out).

    Equivariant to the 24-element spatial rotation group O acting on the 24
    sticker positions.  One scalar weight per pair-orbit (c_out × c_in ×
    n_orbits free parameters total).

        out[b, i, o] = Σ_{j,c}  weight[o, c, orbit(i,j)] * x[b, j, c]  +  bias[o]
    """

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        pair_orbit, n_orbits = compute_spatial_pair_orbits()
        self.n_orbits = n_orbits
        self.register_buffer('pair_orbit', torch.tensor(pair_orbit, dtype=torch.long))
        self.weight = nn.Parameter(torch.empty(c_out, c_in, n_orbits))
        self.bias = nn.Parameter(torch.zeros(c_out))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 24, c_in)
        kernel = self.weight[:, :, self.pair_orbit]   # (c_out, c_in, 24, 24)
        out = torch.einsum('bjc,ocij->bio', x, kernel)
        return out + self.bias


class InvariantLinear(nn.Module):
    """Invariant output: (batch, 24, c_in) -> (batch, 1).

    Averages over the 24 position dimension (invariant to any permutation of
    positions) then applies a scalar linear layer.
    """

    def __init__(self, c_in: int):
        super().__init__()
        self.linear = nn.Linear(c_in, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=1)       # (batch, c_in)
        return self.linear(pooled)   # (batch, 1)


# ─── Spatial + color layers ──────────────────────────────────────────────────

class BothConvLayer(nn.Module):
    """Equivariant linear layer: (batch, 24, 6*k_in) -> (batch, 24, 6*k_out).

    Equivariant to the combined group G = O × S6 (17,280 elements):
      - O (24 spatial rotations) acts on the 24 position axis
      - S6 (720 color permutations) acts on the 6-channel color axis within
        each block

    Hidden channels are organized as k blocks of 6 (one copy of the S6
    permutation representation per block).  The kernel is parameterized by
    triple-orbit indices:

        weight: (k_out, k_in, n_spatial_orbits, 2)

    where the last dimension indexes the 2 color-pair orbits under S6
    (same-color: a==b, and different-color: a≠b).

    Total free parameters: k_out × k_in × n_spatial_orbits × 2
    Compare to unconstrained: (6k_out) × (6k_in) × 24 × 24

        out[b, i, ko, do] = Σ_{j,ki,ci}
            weight[ko, ki, sp_orbit(i,j), co_orbit(do,ci)] * x[b, j, ki, ci]
          + bias[ko]
    """

    def __init__(self, k_in: int, k_out: int):
        super().__init__()
        self.k_in = k_in
        self.k_out = k_out

        sp_orbit, n_sp = compute_spatial_pair_orbits()
        co_orbit, n_co = compute_color_pair_orbits()    # n_co = 2

        self.register_buffer('sp_orbit', torch.tensor(sp_orbit, dtype=torch.long))
        self.register_buffer('co_orbit', torch.tensor(co_orbit, dtype=torch.long))

        self.weight = nn.Parameter(torch.empty(k_out, k_in, n_sp, n_co))
        # One bias scalar per output block, shared across positions and colors.
        self.bias = nn.Parameter(torch.zeros(k_out))

        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 24, 6*k_in)
        B = x.shape[0]
        x = x.view(B, 24, self.k_in, 6)  # (B, pos_j, block_in, color_in)

        # Build full kernel via orbit lookup:
        #   weight[:, :, sp_orbit, :]  ->  (k_out, k_in, 24, 24, 2)
        #   [..., co_orbit]            ->  (k_out, k_in, 24, 24, 6, 6)
        kernel = self.weight[:, :, self.sp_orbit, :]        # (k_out, k_in, 24, 24, 2)
        kernel_full = kernel[:, :, :, :, self.co_orbit]     # (k_out, k_in, 24, 24, 6, 6)
        #                                   o   k   i   j   d   c

        # Contract:
        #   x:           b j k c
        #   kernel_full: o k i j d c
        #   out:         b i o d
        out = torch.einsum('bjkc,okijdc->biod', x, kernel_full)

        # bias: (k_out,) -> broadcast over (B, 24, k_out, 6)
        out = out + self.bias[None, None, :, None]

        return out.reshape(B, 24, self.k_out * 6)


class InvariantHead(nn.Module):
    """Invariant output: (batch, 24, 6*k_in) -> (batch, 1).

    Achieves invariance to both spatial rotations and color permutations by:
      1. Averaging over 24 positions (spatial invariance)
      2. Averaging over 6 color channels within each block (color invariance)
      3. Linear on the k_in block-level scalars -> scalar output
    """

    def __init__(self, k_in: int):
        super().__init__()
        self.k_in = k_in
        self.linear = nn.Linear(k_in, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = x.view(B, 24, self.k_in, 6)
        pooled = x.mean(dim=(1, 3))      # (B, k_in) — avg over positions and colors
        return self.linear(pooled)        # (B, 1)


# ─── Color-only layers (S6, no spatial equivariance) ─────────────────────────

class ColorConvLayer(nn.Module):
    """Equivariant linear layer: (batch, 24, 6*k_in) -> (batch, 24, 6*k_out).

    Equivariant to S6 acting on the 6-channel color axis only.
    The 24 position axis has no group action — each position is processed
    independently with shared weights (weight-tying across positions).

    weight: (k_out, k_in, n_co)  where n_co = 2 (same/diff color pair orbits)

        out[b, i, ko, do] = Σ_{ki, ci}
            weight[ko, ki, co_orbit(do, ci)] * x[b, i, ki, ci]
          + bias[ko]

    Total free parameters: k_out × k_in × 2 + k_out
    Compare to BothConvLayer: k_out × k_in × n_spatial_orbits × 2 + k_out
    """

    def __init__(self, k_in: int, k_out: int):
        super().__init__()
        self.k_in = k_in
        self.k_out = k_out

        co_orbit, n_co = compute_color_pair_orbits()   # n_co = 2
        self.register_buffer('co_orbit', torch.tensor(co_orbit, dtype=torch.long))

        self.weight = nn.Parameter(torch.empty(k_out, k_in, n_co))
        self.bias = nn.Parameter(torch.zeros(k_out))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 24, 6*k_in)
        B = x.shape[0]
        x = x.view(B, 24, self.k_in, 6)  # (B, pos, block_in, color_in)

        # Build full kernel via color orbit lookup:
        #   weight[:, :, co_orbit]  ->  (k_out, k_in, 6, 6)
        kernel = self.weight[:, :, self.co_orbit]   # (k_out, k_in, 6, 6)
        #                               o    k   d  c

        # Contract over (block_in, color_in); positions are independent:
        #   x:      b i k c
        #   kernel: o k d c
        #   out:    b i o d
        out = torch.einsum('bikc,okdc->biod', x, kernel)

        out = out + self.bias[None, None, :, None]
        return out.reshape(B, 24, self.k_out * 6)
