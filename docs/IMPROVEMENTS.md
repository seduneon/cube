# Improvement plan (October 2026)

Ideas for improving the models, ordered from easiest and safest to hardest.
"Safe" means the idea cannot change any result we already have.
Context: `docs/STATUS.md` (findings), `docs/THEORY.md` (derivations).

## Ground rules: how not to ruin existing experiments

1. **Opt-in only.** Every new feature is off by default. With the feature off,
   the code must reproduce the old numbers exactly.
2. **Regression check before each step.** Run `pytest -m "not slow"`, then
   re-evaluate one existing checkpoint with the feature off and compare the
   numbers with the saved results (they must be identical).
3. **New tag for every training variant.** `RunSpec.run_id` encodes only
   preset, train size, sampler, partition policy and seed. The loss,
   consistency weight etc. are **not** in it. A new variant without a new
   `tag` would collide with an old run in the resumable grid (skipped as
   "done", or overwritten).
4. **New results directory for post-hoc evaluation**, e.g. `results/posthoc/`.
   Never write into `results/pilot` or `results/bellman`.
5. **New `TrainConfig` fields get defaults equal to the old behaviour**, so old
   saved configs still load with `from_dict`.
6. **Oracle test first.** Every new "given" (a fact about the puzzle) gets an
   exhaustive test on all 3,674,160 states (marked `slow`) before any model
   uses it.

## Tier 0: no retraining, wrap existing checkpoints

These follow the existing Decorator pattern in `eqcube/inference`
(predictor in, predictor out), so evaluation and search accept them unchanged.
They run on saved checkpoints and need no new training. Fastest feedback.

| # | Idea | What to build | Effort | Exact or heuristic |
|---|---|---|---|---|
| 0.1 | **Parity rounding + pathmax** (THEORY §8) | heuristic wrappers for search | ~1 h | exact (keeps admissibility) |
| 0.2 | **Pattern-database lower bound** | BFS on permutation (5040) and twist (729) coordinates (move tables already exist in `pocket_coords.py`); `LowerBoundPredictor` removes mass below max(d_perm, d_twist) and on the wrong parity, then renormalizes | ~half day | exact bound; oracle test: bound ≤ d on all states |
| 0.3 | **Inversion averaging** d(s) = d(s⁻¹) | state inverse via `decode` → invert pieces → `encode`; `InversionPredictor` averages log-probs of s and s⁻¹ | ~half day | exact fact; oracle test: d(s) = d(s⁻¹) on all states |
| 0.4 | **Ranking-rule comparison** (planned experiment 3) | `rank_children` on existing checkpoints | ~half day | — |

### Tier 0 status (2026-10-07): 0.1–0.3 done, tested

Code: `PocketCubeCoords.lower_bound / invert` (puzzles/pocket_coords.py),
`LowerBoundPredictor`, `InversionPredictor` (inference/givens.py),
`LowerBoundHeuristic`, `ParityRounded`, `Pathmax` (search/heuristics.py),
`fit_pooling(min_beta=...)` (default unchanged), `experiments/posthoc.py`.
Exact checks: bound <= d and d(s^-1) = d(s) on every state (the latter `slow`).

Facts learned from the tests:
- The pattern-database bound is exact for d <= 4 but weak on hard states:
  mean bound ~6 for d = 8..14 (max 8).
- Pathmax changes only *inconsistent* heuristics. The bound and c·d are
  consistent, so pathmax leaves them unchanged; it matters for A* with a
  learned quantile.
- Parity rounding is for admissible heuristics only (siblings share a parity,
  so it would create ties when ranking).

Post-hoc results (`python -m eqcube.experiments.posthoc`, ~1 min; test split,
200 states per depth; seed 42 only; "base" reproduces the saved result.json
exactly). best = best-child rate, band = depths 7–10, bk = with min backup:

| model | chain | NLL | acc | band acc | best (band) | bk best (band) |
|---|---|---|---|---|---|---|
| categorical 200k | base | 1.236 | 0.520 | 0.330 | 0.821 (0.659) | 0.876 (0.780) |
| | bound+inv | 0.695 | 0.666 | 0.527 | 0.858 (0.738) | 0.893 (0.830) |
| ordinal+Bellman 200k | base | 1.077 | 0.553 | 0.542 | 0.864 (0.745) | 0.904 (0.847) |
| | bound+inv | 0.631 | 0.687 | 0.771 | 0.876 (0.769) | 0.923 (0.879) |
| categorical 1M | base | 0.875 | 0.646 | 0.485 | 0.901 (0.823) | 0.944 (0.918) |
| | bound+inv | 0.432 | 0.783 | 0.666 | 0.925 (0.881) | 0.959 (0.951) |

Reading (single seed; per-depth subsets give roughly ±0.01 on band numbers):
- **bound** mostly fixes accuracy and NLL (it removes wrong-parity and
  too-low mass) and helps ranking a little: the same accuracy/ranking split as
  finding 4.
- **inv** gives most of the ranking gain: s^-1 is a second view of the same
  distance with partly independent errors, i.e. more information.
- Gains add up, and they survive the backup (best result: 1M model with
  bound+inv+backup, band best-child 0.918 → 0.951).
- With the bound, val NLL chose beta = 0 (no backup) for ordinal+Bellman,
  although the backup still improves ranking (0.876 → 0.923). Posthoc
  therefore forces beta >= 0.25. Fitting (w, beta) by val best-child instead
  of NLL is the principled fix (not done).

Search (`--search-widths 1 5`; test/search suite, 200 starts at each of
depths 4, 7, 10, 12, 14; max 30 steps). The raw net at width 20 reproduces
the saved solve numbers exactly. Width 20 is saturated (all models solve
100%), so greedy (width 1) and width 5 are the informative settings:

| model | greedy | greedy+bk | w5 | w5+bk |
|---|---|---|---|---|
| categorical 200k, base | 0.525 | 0.725 | 0.923 | 0.957 |
| categorical 200k, bound+inv | 0.625 | 0.820 | 0.976 | 0.998 |
| ordinal+Bellman 200k, base | 0.627 | 0.805 | 0.961 | 0.983 |
| ordinal+Bellman 200k, bound+inv | 0.694 | 0.903 | 0.989 | 1.000 |
| categorical 1M, base | 0.698 | 0.910 | 0.999 | 0.995 |
| categorical 1M, bound+inv | 0.809 | 0.966 | 1.000 | 1.000 |

(solve rate; ±0.015 sampling error.) Extra moves are over *solved* starts
only, so a search that solves more hard starts can show more extra moves;
compare them only at equal solve rates (w5+bk: 0.305 → 0.203, 0.169 → 0.092,
0.019 → 0.010).

### 0.4 ranking rules (2026-10-07): done

`inference/ranking.py`: expected, argmax, median, q25, q75, p_closest
(P(child is the closest sibling), independent children), p_down (THEORY §3).
Run: `python -m eqcube.experiments.posthoc --rules` (log in
`results/posthoc_rules/`). Best-child (band 7–10), 1M model:

| rule | base | base+bk | bound+inv | bound+inv+bk |
|---|---|---|---|---|
| expected | 0.901 (0.823) | 0.944 (0.918) | 0.925 (0.881) | 0.959 (0.951) |
| argmax | 0.886 (0.795) | 0.917 (0.865) | 0.887 (0.800) | 0.911 (0.874) |
| median | 0.900 (0.820) | 0.943 (0.916) | 0.925 (0.881) | 0.959 (0.951) |
| p_closest | 0.897 (0.816) | 0.944 (0.921) | 0.924 (0.879) | 0.962 (0.959) |
| p_down | 0.894 (0.804) | 0.943 (0.909) | 0.921 (0.865) | 0.961 (0.948) |

Same pattern for the other two models. **Reading:** among sensible rules
(expected, median, quantiles, p_closest) the differences are within noise
(≤ 0.01); argmax is clearly worse (ties), p_down slightly worse (it relies
on independent sibling errors, cf. THEORY §9). Columns differ by up to
0.13. So the *rule* barely matters; the *information* in the distributions
(wrappers, backup, more labels) does. The distributions are close to
unimodal with similar shapes, and then every location summary orders
siblings the same way.

Still open in Tier 0: fitting (w, beta) by val best-child instead of NLL.

## Tier 1: new training runs, existing code

Only new run IDs (new seeds, new tags), so old runs are not touched.

| # | Idea | Note |
|---|---|---|
| 1.1 | **20k labels with a step-based budget** (planned experiment 1) | needs `max_steps: int \| None = None` in `TrainConfig` (None = old behaviour) |
| 1.2 | **3 seeds**: categorical vs ordinal+Bellman (planned experiment 2) | seeds 43, 44 are new run IDs |
| 1.3 | **Ensembles** from the seeds of 1.2 | `EnsemblePredictor` (average probabilities); also a Tier 0 wrapper once the seeds exist |

## Tier 2: new training code (opt-in flags, new tags)

| # | Idea | Note |
|---|---|---|
| 2.1 | **Sibling ranking loss** | softmax over a parent's 6 children; target = the "down" children (labels from the table). Trains P(c down) directly (THEORY §3); a shared sibling bias cancels. Flag `ranking_weight: float = 0.0` |
| 2.2 | **Inversion in training** | augment with s⁻¹, or a consistency loss between p(s) and p(s⁻¹) |
| 2.3 | **Better Bellman targets** | two-sided target (THEORY §2), depth curriculum, clip targets from below by the pattern-database bound |
| 2.4 | **Pattern-database values as input features** | new preset. First check that d_perm and d_twist are invariant under S3, otherwise equivariance breaks |

### 2.1 status (2026-10-07): implemented, one smoke run

Code: `training/ranking.py` (`listwise_loss`, `sibling_ranking_loss`,
`SiblingRanking`), `TrainConfig.ranking_weight / ranking_batch / ranking_tau`
(default 0 = old behaviour; checked: old and new code train identically on
CPU up to float noise), `experiments/ranking.py` (tags `categorical`,
`categorical+rank`, output `results/ranking/`).

Smoke run: 20k labels, seed 42, ranking_weight 1, tau 1, MPS. Post-hoc on
the test subset (200/depth; search: 1000 starts):

| | best (band) | bk best (band) | greedy | greedy+bk | w5+bk | w20 (raw) | NLL |
|---|---|---|---|---|---|---|---|
| CE | 0.671 (0.431) | 0.742 (0.496) | 0.036 | 0.440 | 0.676 | 0.831 | 2.044 |
| CE + rank | 0.720 (0.471) | 0.783 (0.576) | 0.368 | 0.515 | 0.770 | 0.936 | 1.865 |
| CE, bound+inv | 0.729 (0.471) | 0.778 (0.550) | 0.129 | 0.493 | 0.799 | | 1.291 |
| CE + rank, bound+inv | 0.770 (0.515) | 0.803 (0.599) | 0.445 | 0.595 | 0.894 | | 1.116 |

**Caveats before reading this as "ranking beats estimation":**
1. **Not equal information.** The ranking term reads the labels of 6
   children per parent, so CE+rank sees ~7× more labeled states. Needed
   control: CE on the same states (parents + their children, ~140k labels).
2. **One seed, and training is not bit-reproducible on MPS** (retraining
   the CE baseline gave val CE 2.115 vs the saved 2.129, same epochs).
3. Early stopping is on val CE, which the ranking term does not optimize.
4. Surprise: NLL also improved (2.04 → 1.87), consistent with caveat 1
   (more information) rather than with "ranking only".

Next for 2.1: the equal-information control (`ce_children`), then 3 seeds,
then 200k; also ranking_weight ∈ {0.3, 1, 3} and scramble-move labels.

## Tier 3: search

| # | Idea | Note |
|---|---|---|
| 3.1 | Backup inside beam search, with a cache (planned experiment 4) | ≈1× cost |
| 3.2 | Weighted A* with the ε-quantile heuristic | gives solution-quality guarantees |

## Tier 4: larger changes

- Piece encoding (7 corners × position/orientation) instead of stickers.
  Makes inversion natural; closer to a 3×3×3 encoding.
- Capacity study (40k → 200k parameters), to rule capacity out as the bottleneck.
- Full preset grid (planned experiment 5, ~10 h).

## What carries over to the 3×3×3

Pattern databases (Korf), inversion, ensembles, the ranking loss with Bellman
targets, and pathmax/parity rounding all carry over. Features that need the
exact table (2.1 with table labels) do not; there they need Bellman targets.

## Reading for these ideas

- Pattern databases: Korf 1997 (optimal Rubik's cube); Felner et al., additive PDBs.
- Learning to rank: Burges, *From RankNet to LambdaRank to LambdaMART*; ListNet.
- Ensembles: Lakshminarayanan et al. 2017, *Deep Ensembles*.
- Symmetrization: Bronstein et al., *Geometric Deep Learning*, invariance chapters.
