"""
Pure PyTorch equivariant layers for the 2x2x2 Pocket Cube.

Spatial-only (emlp_rot):
  GroupConvLayer    (batch, 24, c_in) -> (batch, 24, c_out)
  InvariantLinear   (batch, 24, c_in) -> (batch, 1)   [average over positions]
  RotResBlock       (batch, 24, c)    -> (batch, 24, c)   [pre-norm residual]

Spatial + color (emlp_both):
  BothConvLayer   (batch, 24, 6*k_in) -> (batch, 24, 6*k_out)
  InvariantHead   (batch, 24, 6*k_in) -> (batch, 1)
  BothResBlock    (batch, 24, 6*k)    -> (batch, 24, 6*k)  [pre-norm residual]

Color-only S6 (emlp_col):
  ColorConvLayer  (batch, 24, 6*k_in) -> (batch, 24, 6*k_out)
  InvariantHead   (batch, 24, 6*k_in) -> (batch, 1)         [shared with above]
  ColorResBlock   (batch, 24, 6*k)    -> (batch, 24, 6*k)   [pre-norm residual]

Both color-equivariant families use InvariantHead and require hidden channel
counts to be multiples of 6 (one block per copy of the S6 rep).

BothConvLayer and ColorConvLayer use an α/β decomposition to avoid ever
materializing a 6D kernel, keeping peak memory linear in k rather than k².
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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


class RotResBlock(nn.Module):
    """Pre-norm residual block: (batch, 24, c) -> (batch, 24, c).

    LayerNorm over channels, then GroupConvLayer, then ReLU, plus skip.
    Equivariant to spatial rotations (LayerNorm over channels preserves
    equivariance; skip connection preserves it trivially).
    """

    def __init__(self, c_hidden: int):
        super().__init__()
        self.norm = nn.LayerNorm(c_hidden)
        self.conv = GroupConvLayer(c_hidden, c_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + F.relu(self.conv(self.norm(x)))


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

    Forward uses an α/β decomposition to avoid materializing the full
    (k_out, k_in, 24, 24, 6, 6) kernel:

        α = weight[..., 0] - weight[..., 1]   (same-color minus diff-color)
        β = weight[..., 1]                     (diff-color coefficient)

        out[b,i,ko,d] = Σ_{j,ki} α[ko,ki,sp(i,j)] * x[b,j,ki,d]      # per-color spatial conv
                      + Σ_{j,ki} β[ko,ki,sp(i,j)] * x_sum[b,j,ki]    # color-sum spatial conv
    """

    def __init__(self, k_in: int, k_out: int):
        super().__init__()
        self.k_in = k_in
        self.k_out = k_out

        sp_orbit, n_sp = compute_spatial_pair_orbits()
        _, n_co = compute_color_pair_orbits()    # n_co = 2

        self.register_buffer('sp_orbit', torch.tensor(sp_orbit, dtype=torch.long))

        self.weight = nn.Parameter(torch.empty(k_out, k_in, n_sp, n_co))
        self.bias = nn.Parameter(torch.zeros(k_out))

        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 24, 6*k_in)
        B = x.shape[0]
        x = x.view(B, 24, self.k_in, 6)  # (B, pos_j, block_in, color_in)

        # Decompose weight into per-color (α) and color-sum (β) components.
        # Avoids materializing the (k_out, k_in, 24, 24, 6, 6) kernel.
        alpha = self.weight[:, :, :, 0] - self.weight[:, :, :, 1]  # (k_out, k_in, n_sp)
        beta  = self.weight[:, :, :, 1]                              # (k_out, k_in, n_sp)

        alpha_kernel = alpha[:, :, self.sp_orbit]  # (k_out, k_in, 24, 24)
        beta_kernel  = beta[:, :, self.sp_orbit]   # (k_out, k_in, 24, 24)

        # Term 1: spatial group conv applied independently per color channel
        term1 = torch.einsum('bjkd,okij->biod', x, alpha_kernel)  # (B, 24, k_out, 6)

        # Term 2: spatial group conv on color sum, broadcast over output colors
        x_sum = x.sum(dim=3)                                        # (B, 24, k_in)
        term2 = torch.einsum('bjk,okij->bio', x_sum, beta_kernel)  # (B, 24, k_out)

        out = term1 + term2.unsqueeze(-1)
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


class ColorEquivariantLayerNorm(nn.Module):
    """LayerNorm over 6*k channels with γ/β shared within each block of 6.

    Standard nn.LayerNorm has an independent γ and β per channel, so it can
    learn different scales for each color slot and break S6 equivariance.
    This version has one γ and one β per block, shared across all 6 colors,
    which preserves S6 equivariance exactly.

    The normalization (mean/std over all 6*k channels) is already equivariant
    because permuting colors within blocks does not change the global statistics.
    Only the affine parameters need tying.
    """

    def __init__(self, k_hidden: int, eps: float = 1e-5):
        super().__init__()
        self.k_hidden = k_hidden
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(k_hidden))   # γ, one per block
        self.bias   = nn.Parameter(torch.zeros(k_hidden))  # β, one per block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., 6*k_hidden)
        shape = x.shape
        mean = x.mean(dim=-1, keepdim=True)
        var  = x.var(dim=-1, unbiased=False, keepdim=True)
        x_norm = (x - mean) / (var + self.eps).sqrt()
        # Reshape to (..., k_hidden, 6) so block-level γ/β broadcast over colors
        x_norm = x_norm.view(*shape[:-1], self.k_hidden, 6)
        x_norm = x_norm * self.weight[..., None] + self.bias[..., None]
        return x_norm.reshape(shape)


class BothResBlock(nn.Module):
    """Pre-norm residual block: (batch, 24, 6*k) -> (batch, 24, 6*k).

    Uses ColorEquivariantLayerNorm so that the affine parameters do not
    break S6 equivariance.
    """

    def __init__(self, k_hidden: int):
        super().__init__()
        self.norm = ColorEquivariantLayerNorm(k_hidden)
        self.conv = BothConvLayer(k_hidden, k_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + F.relu(self.conv(self.norm(x)))


# ─── Color-only layers (S6, no spatial equivariance) ─────────────────────────

class ColorConvLayer(nn.Module):
    """Equivariant linear layer: (batch, 24, 6*k_in) -> (batch, 24, 6*k_out).

    Equivariant to S6 acting on the 6-channel color axis only.
    The 24 position axis has no group action — each position is processed
    independently with shared weights (weight-tying across positions).

    weight: (k_out, k_in, 2)  — one scalar per (block pair, color orbit)

    Forward uses the same α/β decomposition as BothConvLayer:

        α = weight[..., 0] - weight[..., 1]
        β = weight[..., 1]

        out[b,i,ko,d] = Σ_{ki} α[ko,ki] * x[b,i,ki,d]        # per-color linear
                      + Σ_{ki} β[ko,ki] * x_sum[b,i,ki]       # color-sum linear
    """

    def __init__(self, k_in: int, k_out: int):
        super().__init__()
        self.k_in = k_in
        self.k_out = k_out

        _, n_co = compute_color_pair_orbits()   # n_co = 2

        self.weight = nn.Parameter(torch.empty(k_out, k_in, n_co))
        self.bias = nn.Parameter(torch.zeros(k_out))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 24, 6*k_in)
        B = x.shape[0]
        x = x.view(B, 24, self.k_in, 6)  # (B, pos, block_in, color_in)

        alpha = self.weight[:, :, 0] - self.weight[:, :, 1]  # (k_out, k_in)
        beta  = self.weight[:, :, 1]                           # (k_out, k_in)

        # Term 1: per-color linear (positions independent)
        term1 = torch.einsum('bikd,ok->biod', x, alpha)  # (B, 24, k_out, 6)

        # Term 2: color-sum linear, broadcast over output colors
        x_sum = x.sum(dim=3)                               # (B, 24, k_in)
        term2 = torch.einsum('bik,ok->bio', x_sum, beta)  # (B, 24, k_out)

        out = term1 + term2.unsqueeze(-1)
        out = out + self.bias[None, None, :, None]
        return out.reshape(B, 24, self.k_out * 6)


class ColorResBlock(nn.Module):
    """Pre-norm residual block: (batch, 24, 6*k) -> (batch, 24, 6*k).

    Uses ColorEquivariantLayerNorm for the same reason as BothResBlock.
    """

    def __init__(self, k_hidden: int):
        super().__init__()
        self.norm = ColorEquivariantLayerNorm(k_hidden)
        self.conv = ColorConvLayer(k_hidden, k_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + F.relu(self.conv(self.norm(x)))


class ColorOnlyInvariantHead(nn.Module):
    """Invariant output for ColorValueNet: (batch, 24, 6*k_in) -> (batch, 1).

    Averages over the 6 color channels within each block (S6 invariance) but
    keeps all 24 positions as separate features.  The (24 * k_in)-dimensional
    pooled representation is then mapped to a scalar via a learned linear layer.

    Contrast with InvariantHead which also averages over positions — that is
    correct for spatially equivariant models but discards all positional
    information for color-only models.
    """

    def __init__(self, k_in: int):
        super().__init__()
        self.k_in = k_in
        self.linear = nn.Linear(24 * k_in, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = x.view(B, 24, self.k_in, 6)
        pooled = x.mean(dim=3)                      # (B, 24, k_in) — avg over colors only
        return self.linear(pooled.reshape(B, -1))   # (B, 1)
