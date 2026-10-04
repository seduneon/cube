"""Bellman consistency: make a state's prediction agree with its children's.

For an unsolved state s,  d(s) > k  <=>  every child c has d(c) > k - 1, so
with S(k) = P(d > k) and children treated as independent:

    S_s(0) = 1,    S_s(k) = prod_c S_c(k - 1)      (k >= 1)

The solved state is exactly d = 0. ``backup_probs`` turns the children's
distributions into the parent's this way (the NumPy twin used at inference
is ``eqcube.inference.backup.backup_log_probs``; tests keep them equal).

Consistency loss (semi-supervised, in the spirit of DeepCubeA's value
iteration): on *unlabeled* states, KL(backup of the children || model(s)).
The children are scored by a slowly moving EMA copy of the model with no
gradient (a "target network", as in DQN and Mean Teacher), which keeps the
targets stable. With "solved = 0" as an anchor, the true distance is the only
exact fixed point of the Bellman equation.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import Tensor, nn

from eqcube.data.partition import Partition
from eqcube.data.samplers import TemperedSampler
from eqcube.data.table import DistanceTable

EPS = 1e-12


def backup_probs(child_log_probs: Tensor, solved: Tensor) -> Tensor:
    """(N, M, C) children log-probs, (N,) solved mask -> (N, C) parent probabilities."""
    n, _, c = child_log_probs.shape
    child_s = (1.0 - torch.cumsum(child_log_probs.exp(), dim=-1)[..., :-1]).clamp(EPS, 1.0)
    # parent S(k) for k = 1..C-2 uses children S(k-1) for k-1 = 0..C-3
    log_s = torch.log(child_s[..., :-1]).sum(dim=1)  # (N, C-2)
    s = torch.cat([torch.ones(n, 1, device=log_s.device), log_s.exp()], dim=-1)  # S(0..C-2)
    s_prev = torch.cat([torch.ones(n, 1, device=s.device), s[:, :-1]], dim=-1)  # S(-1..C-3)
    probs = torch.cat([s_prev - s, s[:, -1:]], dim=-1).clamp_min(0.0)  # P(0..C-1)
    delta0 = torch.zeros(n, c, device=probs.device)
    delta0[:, 0] = 1.0
    return torch.where(solved[:, None], delta0, probs)


class BellmanConsistency:
    """KL(backup of EMA-target children || model) on states sampled from the train split."""

    def __init__(
        self,
        model: nn.Module,
        table: DistanceTable,
        sampler: TemperedSampler,
        batch: int,
        ema_decay: float,
        device: torch.device,
        seed: int,
    ) -> None:
        self.target = copy.deepcopy(model).eval()
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.table, self.sampler, self.batch = table, sampler, batch
        self.decay, self.device = ema_decay, device
        self.rng = np.random.default_rng(seed + 1)

    @property
    def partition(self) -> Partition:
        return self.sampler.partition

    def loss(self, model: nn.Module) -> Tensor:
        indexer = self.table.indexer
        index = self.sampler.sample(self.batch, self.rng)  # labels are never read
        children = indexer.expand(index)  # (B, M)
        states = torch.as_tensor(indexer.unrank(index), device=self.device)
        child_states = torch.as_tensor(indexer.unrank(children.reshape(-1)), device=self.device)
        solved = torch.as_tensor(index == indexer.solved_index, device=self.device)
        with torch.no_grad():
            child_lp = self.target(child_states).view(len(index), children.shape[1], -1)
            p_target = backup_probs(child_lp, solved)
        log_p_model = model(states)
        kl = p_target * (torch.log(p_target.clamp_min(EPS)) - log_p_model)
        return kl.sum(dim=-1).mean()

    @torch.no_grad()
    def update_target(self, model: nn.Module) -> None:
        for t, p in zip(self.target.parameters(), model.parameters(), strict=True):
            t.lerp_(p, 1.0 - self.decay)
        for t, b in zip(self.target.buffers(), model.buffers(), strict=True):
            t.copy_(b)
