"""Distance parity as a model component (pocket cube).

The distance of a state has the same parity as its corner permutation, so a
state at even distance can never be at an odd one. ``ParityMask`` forbids
the impossible half of the distance classes. Parity is preserved by every
task symmetry (they preserve distance), so the mask keeps models invariant.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from eqcube.puzzles.pocket_coords import N_PIECES, PocketCubeCoords


class CornerParity(nn.Module):
    """(B, 24) sticker colors -> (B,) parity of the corner permutation, on any device."""

    slots: Tensor
    piece_of_code: Tensor
    ud: Tensor
    upper: Tensor

    def __init__(self, coords: PocketCubeCoords) -> None:
        super().__init__()
        self.register_buffer("slots", torch.as_tensor(coords.slot_stickers[:N_PIECES]))
        self.register_buffer("piece_of_code", torch.as_tensor(coords.piece_of_code))
        self.register_buffer("ud", torch.as_tensor(coords.ud_colors))
        self.register_buffer("upper", torch.ones(N_PIECES, N_PIECES).triu(1).bool())

    def forward(self, states: Tensor) -> Tensor:
        colors = states.long()[:, self.slots]  # (B, 7, 3)
        is_ud = (colors == self.ud[0]) | (colors == self.ud[1])
        twist = is_ud.to(torch.uint8).argmax(dim=2)  # position of the U/D sticker
        k = (twist[..., None] + torch.arange(3, device=states.device)) % 3
        canonical = torch.gather(colors, 2, k)  # read colors starting at U/D
        code = canonical[..., 0] * 36 + canonical[..., 1] * 6 + canonical[..., 2]
        piece = self.piece_of_code[code]  # (B, 7)
        inversions = ((piece[:, :, None] > piece[:, None, :]) & self.upper).sum(dim=(1, 2))
        return inversions % 2


class ParityMask(nn.Module):
    """Adds a large negative number to the logits of distances with the wrong parity."""

    classes: Tensor

    def __init__(self, coords: PocketCubeCoords, num_classes: int, penalty: float = 1e4) -> None:
        super().__init__()
        self.parity = CornerParity(coords)
        self.penalty = penalty
        self.register_buffer("classes", torch.arange(num_classes) % 2)

    def forward(self, states: Tensor, logits: Tensor) -> Tensor:
        wrong = self.classes[None, :] != self.parity(states)[:, None]
        return logits - self.penalty * wrong.to(logits.dtype)
