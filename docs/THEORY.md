# Theory notes: distributions, the Bellman structure, and search

Notation: s is an unsolved state with children c_1..c_M (M = 6 moves). The
model gives a distribution p_s over d(s) and q_c over d(c). Distances lie in
{0, ..., C-1} (C = 15).

## 1. What the puzzle guarantees (exact, verified on all states)

For every unsolved s:

    (A) every child is at d(s) - 1 or d(s) + 1        (one move changes d by 1;
                                                        parity flips every move)
    (B) at least one child is at d(s) - 1             (a shortest path exists)

Conversely, if the children's values are all in {k-1, k+1} and at least one
equals k-1, then d(s) = k. So (A) and (B) characterize the parent exactly.

## 2. The exact backup (two-sided)

Treat the children's model distributions as independent. Condition on the
children's values being *consistent* with some parent value, i.e. with (A)
and (B). Then

    P(d(s) = k | children, consistent)  ∝  L(k)
    L(k) = Π_c A_c(k) - Π_c U_c(k)
    A_c(k) = q_c(k-1) + q_c(k+1)        (child is a neighbour value of k)
    U_c(k) = q_c(k+1)                   (child is the "up" value)

Proof: the event {d(s) = k} is {all children in {k-1, k+1}} minus {all
children equal to k+1}. With independence, the first has probability Π A_c
and the second Π U_c. Different k give disjoint events (k = 1 + min), and
their union is "consistent", so normalizing gives the conditional law. ∎

**Comparison with the min-only backup** used so far,
P(d(s) = k) = Π_c q_c(≥ k-1) − Π_c q_c(≥ k). That formula counts
configurations where a child sits at k + 3, k + 5, ... and so violates (A);
it never renormalizes them away. The two formulas agree when the children's
predictions are exact, and differ exactly when the model is unsure, which
is when the backup matters. The two-sided version uses strictly more of the
known structure.

## 3. Which child is "down"? The joint posterior

The same model gives the posterior of each child's direction. Given the
parent at k, child c is down with probability a_c(k) = q_c(k-1) / A_c(k),
independently, conditioned on at least one being down:

    P(c down | k) = a_c(k) / (1 - Π_j (1 - a_j(k)))
    P(c down)     = Σ_k π(k) P(c down | k)
    E[d(c)]       = Σ_k π(k) (k + 1 - 2 P(c down | k))

where π is the parent posterior (Section 4). E[d(c)] is the child's expected
distance **given its siblings and its parent**. It is comparable across
parents, so beam search can rank by it.

**Decision rule.** Greedy search makes progress iff it picks a down child, so
the Bayes-optimal greedy choice is argmax_c P(c down). Ranking children by
their own E[d] (what search did so far) ignores the siblings: if all six
children look like "about 9", what matters is which one looks *least* like
the up value, relative to the others.

## 4. Combining with the parent's own prediction

Bayes with the parent's prediction as a prior and the children as evidence:

    π(k) ∝ p_s(k)^w · L(k)^β

w = β = 1 is exact Bayes if the errors at s and at its children are
independent. They are not (the same network scores neighbouring states), so
the evidence would be double-counted. Tempering the two factors with
exponents (a "power" or generalized-Bayes posterior; cf. Grünwald's safe
Bayes) is the standard correction; (w, β) are fit on validation NLL. The
old log opinion pool is this same form with the min-only likelihood.

Calibration matters for the same reason: L multiplies six children's
probabilities, so an overconfident q (we measured T ≈ 1.5–2.2) is amplified.
Temperature-scale first, then back up.

## 5. Depth > 1 and loops

Backing up children from their own children treats the neighbourhood as a
tree (the "computation tree" view of belief propagation). The Cayley graph
has short cycles (R⁴ = e gives 4-cycles), so deeper backups reuse the same
states and double-count; the tempering exponents absorb part of that. Exact
inference on the full graph is the BFS itself.

## 6. Why accuracy and ranking decouple (the diagnostic result)

Accuracy asks for d(s). Ranking asks only for the sign of d(c₁) − d(c₂) ∈
{−2, 0, 2} between siblings. Errors that shift every sibling equally (the
pull toward 11, or the parity offset) hurt accuracy but cancel in a ranking.
That is why the parity mask and depth-balanced sampling raised accuracy
without changing best_child, while more information (1M labels, Bellman
consistency) improved both.

## 7. Bellman consistency: why 20k labels can fail

The Bellman operator on exact values converges from the solved state outward
in at most diam = 14 sweeps (value iteration). With a learned function and
a target network, information moves about one hop per target update, and
only if the children are already predicted well (there is no contraction
guarantee: the "deadly triad" of bootstrapping, function approximation, and
off-policy sampling). With 20k labels, an epoch is 40 steps and early
stopping fires after ~2k steps, so the targets may never stabilize.
Theoretically motivated fixes: count training in steps rather than epochs,
use the two-sided target (Section 2, less biased), and use a depth
curriculum (consistency first on shallow states, as DeepCubeA grows scramble
depth).

## 8. Search guarantees (for later)

- A heuristic h is *admissible* if h ≤ d; A* with an admissible h returns
  optimal solutions. From a calibrated distribution, the ε-quantile is
  admissible with probability ≥ 1 − ε (a "likely-admissible" heuristic;
  Ernandes & Gori 2004).
- **Parity rounding:** d(s) has the parity of the corner permutation, which
  is known exactly. Rounding an admissible h up to the next integer of the
  right parity keeps it admissible and is never looser.
- **Pathmax:** since |d(s) − d(c)| = 1, h'(s) = max(h(s), max_c h(c) − 1)
  is admissible when h is, and is consistent.

## 9. Empirical check (Oct 2026, existing checkpoints, test split)

Best-child rate (band 7–10 in brackets), temperature and (w, β) fit on val:

| model | own E[d] | min backup | consistent backup | sibling posterior |
|---|---|---|---|---|
| categorical 200k | 0.821 (0.659) | **0.876 (0.780)** | 0.870 (0.764) | 0.823 (0.659) |
| ordinal+Bellman 200k | 0.864 (0.745) | **0.904 (0.847)** | 0.864 (0.745) | 0.858 (0.733) |
| categorical 1M | 0.901 (0.823) | **0.944 (0.918)** | 0.934 (0.897) | 0.897 (0.814) |

A contamination mixture (1 − ε)·consistent + ε·min, with ε fit on val,
chose ε = 1 (pure min) for two models and ε = 0.5 (same ranking) for one.

**Reading.** The formulas are exact (brute-force tests) under *independent*
child errors. The six siblings are scored by the same network and share
biases (e.g. the pull toward 11), so the extra constraint makes the posterior
confidently wrong. The min rule depends only on the lower tails
P(d ≥ k−1); the two-sided rule also needs the exact mass at k+1, the
worst-calibrated part of these distributions. This is a hypothesis.
The sibling posterior uses the same information as the children's own
predictions, so it cannot rank much better. The gains of the backup come
from the *extra* information (grandchildren), not from a sharper rule.

**Consequence.** Keep `rule="min"` as the default. The consistent rule is
kept, tested, and worth revisiting for models with less correlated errors
(ensembles, or models trained with the consistency loss on the two-sided
target). A principled next step is a likelihood with a shared sibling bias
(a latent shift common to all children).
