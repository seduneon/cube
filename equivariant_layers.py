"""
Pure PyTorch equivariant layers for the 2x2x2 Pocket Cube.

GroupConvLayer: equivariant linear layer via pair-orbit weight sharing.
InvariantLinear: average-pool over 24 positions then a scalar linear layer.

The 24-element rotation group G acts on 24 sticker positions by permutation.
Two ordered pairs (i,j) and (i',j') are in the same orbit iff there exists
g in G with g(i)=i' and g(j)=j'. GroupConvLayer shares one scalar weight per
orbit — dramatically fewer parameters than a general linear map.
"""

import numpy as np
import torch
import torch.nn as nn

from cube_group import ALL_ROTATIONS


def compute_pair_orbits():
    """Compute orbit labels for all position pairs (i, j) under G.

    Returns
    -------
    pair_orbit : int64 ndarray of shape (24, 24)
        orbit_id for each pair; orbit_id is in [0, n_orbits)
    n_orbits   : int
        number of distinct orbits
    """
    pair_orbit = np.full((24, 24), -1, dtype=np.int64)
    orbit_id = 0

    for i in range(24):
        for j in range(24):
            if pair_orbit[i, j] == -1:
                # First time we see this pair — assign a new orbit and label
                # every image of (i, j) under the group.
                for g in ALL_ROTATIONS:
                    gi = int(g[i])
                    gj = int(g[j])
                    if pair_orbit[gi, gj] == -1:
                        pair_orbit[gi, gj] = orbit_id
                orbit_id += 1

    assert (pair_orbit >= 0).all(), "Some pairs were not assigned an orbit."
    return pair_orbit, orbit_id


class GroupConvLayer(nn.Module):
    """Equivariant linear layer: (batch, 24, c_in) -> (batch, 24, c_out).

    The weight tensor has shape (c_out, c_in, n_orbits).  For each output
    position i, the layer computes:

        out[b, i, o] = sum_{j, c}  weight[o, c, pair_orbit[i, j]] * x[b, j, c]
                     + bias[o]

    Because pair_orbit is G-equivariant by construction, applying any rotation
    g to x and then applying this layer gives the same result as applying the
    layer first and then applying g to the output.
    """

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        pair_orbit, n_orbits = compute_pair_orbits()
        self.n_orbits = n_orbits

        # pair_orbit must be a buffer so it moves to GPU with model.to(device)
        self.register_buffer('pair_orbit', torch.tensor(pair_orbit, dtype=torch.long))

        self.weight = nn.Parameter(torch.empty(c_out, c_in, n_orbits))
        self.bias = nn.Parameter(torch.zeros(c_out))

        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 24, c_in)
        # kernel[o, c, i, j] = weight[o, c, pair_orbit[i, j]]
        kernel = self.weight[:, :, self.pair_orbit]          # (c_out, c_in, 24, 24)
        out = torch.einsum('bjc,ocij->bio', x, kernel)       # (batch, 24, c_out)
        return out + self.bias                                # broadcast (c_out,)


class InvariantLinear(nn.Module):
    """Invariant output layer: (batch, 24, c_in) -> (batch, 1).

    Averages over the 24 position dimension (provably invariant to any
    permutation of positions) then applies a scalar linear layer.
    """

    def __init__(self, c_in: int):
        super().__init__()
        self.linear = nn.Linear(c_in, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=1)       # (batch, c_in)
        return self.linear(pooled)   # (batch, 1)
