"""Sibling ranking loss: train what greedy search needs, not the distance itself.

Greedy search solves optimally iff, at every state it visits, the
lowest-scored child is a "down" child (d(child) = d(parent) - 1). With the
score h(c) = E[d(c)] under the model and a Boltzmann policy over siblings,

    pi(c) = softmax_c(-h(c) / tau),      L = -log sum_{c down} pi(c)

is the negative log-probability that the policy picks a down child (a
listwise ranking loss; Chrestien et al. 2023, "Optimize planning heuristics
to rank, not to estimate cost-to-goal"). It depends only on differences
between siblings, so a shift shared by all siblings (the pull toward the
most common depth, a parity offset) gets no gradient.

Parents are drawn from the labeled training set; the down set of each parent
is read from the table (6 more labels per parent). Combined with the main
loss as  L_main + ranking_weight * L  (training/trainer.py).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor, nn

from eqcube.data.table import DistanceTable

IntArray = npt.NDArray[np.int64]


def sibling_ranking_loss(child_log_probs: Tensor, down: Tensor, tau: float = 1.0) -> Tensor:
    """(B, M, C) children log-probs, (B, M) bool down mask -> mean -log P(pick a down child).

    Parents without a down child (only the solved state) must be excluded by the caller.
    """
    k = torch.arange(child_log_probs.shape[-1], device=child_log_probs.device)
    h = (child_log_probs.exp() * k).sum(dim=-1)  # E[d(c)], (B, M)
    return listwise_loss(h, down, tau)


def listwise_loss(scores: Tensor, down: Tensor, tau: float = 1.0) -> Tensor:
    """(B, M) scores (lower = closer), (B, M) down mask -> mean -log sum_{down} softmax(-s/tau)."""
    logits = -scores / tau
    picked = torch.logsumexp(logits.masked_fill(~down, float("-inf")), dim=1)
    return (torch.logsumexp(logits, dim=1) - picked).mean()


class SiblingRanking:
    """The ranking term on parents sampled from the (labeled) training indices."""

    def __init__(
        self,
        table: DistanceTable,
        train_index: IntArray,
        batch: int,
        tau: float,
        device: torch.device,
        seed: int,
    ) -> None:
        self.table, self.batch, self.tau, self.device = table, batch, tau, device
        self.parents = train_index[table.dist[train_index] >= 1]
        self.rng = np.random.default_rng(seed + 2)

    def sample(self) -> tuple[Tensor, Tensor]:
        """Child states (B*M, P) and the down mask (B, M) for a batch of parents."""
        index = self.rng.choice(self.parents, size=self.batch)
        children = self.table.indexer.expand(index)  # (B, M)
        down = self.table.dist[children] < self.table.dist[index, None]
        states = self.table.indexer.unrank(children.reshape(-1))
        return (
            torch.as_tensor(states, device=self.device),
            torch.as_tensor(down, device=self.device),
        )

    def loss(self, model: nn.Module) -> Tensor:
        states, down = self.sample()
        child_lp = model(states).view(*down.shape, -1)
        return sibling_ranking_loss(child_lp, down, self.tau)
