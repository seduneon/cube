# Status (paused October 2026)

Where the project stands, what we learned, and how to pick it up again.
Design: `docs/ARCHITECTURE.md`. Derivations: `docs/THEORY.md`.

## Built (branch `v2`)

symmetry (groups, actions, G-sets) → puzzles (pocket cube, coordinates, S3
task symmetry) → data (exact BFS table, partition, samplers, suites) → nn
(generic equivariant layers, dense / factored strategies) → models (heads,
parity mask, config, checkpoints) → training (losses, Bellman consistency)
→ inference (temperature, Bellman backup) → evaluation → search (beam) →
experiments (resumable grids). About 250 tests; `pytest -m "not slow"` takes ~50 s.

## Findings so far (single seed unless noted; confirm with seeds before claiming)

1. The true symmetry of the R/U/F pocket cube is **S3** (6 elements), not O
   (24) or S6. The 24 rotations were "used up" by fixing the DBL corner (one
   representative per coset). S3 is verified on all 3,674,160 states.
   v1's rotation invariance constrains nothing on reachable states.
2. At equal size (~40k params, 200k labels) the S3 model beats the MLP:
   solve 99.8% vs 94.6%, extra moves 0.13 vs 0.64.
3. Hard band = depths 7–10 (few "closer" moves, many states). Errors there
   are pulls toward the most common depth (11) plus parity confusion.
4. Accuracy and **ranking** (what search needs) decouple. Parity, α = 0 and
   band-weighted sampling raise accuracy but not ranking. Only more
   information raises ranking: 1M labels, or Bellman consistency.
5. Bellman consistency at 200k: best-child (band) 0.666 → 0.736, extra moves
   0.154 → 0.039. At 20k it failed, likely because training stopped too early
   (an untested hypothesis).
6. The test-time backup (min rule, from grandchildren) adds 5–14 points of
   best-child for every decent model; the exact two-sided rule does not beat
   it (THEORY.md §9: correlated sibling errors).
7. Models are overconfident (temperature 1.5–2.2).
8. (Oct 7, post-hoc, no retraining) Exact facts as predictor wrappers help
   every model: the pattern-database bound + parity mostly raises accuracy
   (band acc 0.33 → 0.53 for categorical 200k), and inversion averaging
   (d(s) = d(s⁻¹)) mostly raises ranking. With the backup, the 1M model
   reaches band best-child 0.951 (was 0.918). Greedy search solve rate
   rises on every model (with backup: 0.725 → 0.820, 0.805 → 0.903,
   0.910 → 0.966). Details: IMPROVEMENTS.md.
9. (Oct 7) The ranking rule barely matters: expected value, median,
   quantiles and P(closest) agree within 0.01 best-child; argmax and
   P(down) are worse. What matters is the information in the distributions.
10. (Oct 7, smoke test, 1 seed) Sibling ranking loss at 20k labels: greedy
    solve 0.036 → 0.368, width-20 solve 0.831 → 0.936. Confounded: the loss
    reads 6 extra child labels per parent. Equal-information control needed
    (IMPROVEMENTS.md, 2.1).

## Open questions

- Which score should rank beam candidates? Empirically (finding 9) any
  sensible location summary works; the open part is *why* (when do rules
  disagree? only for multimodal or differently shaped distributions).
- How to model siblings' shared errors (a latent shift; ensembles)?
- Does Bellman help when labels are scarce, given enough training steps?
- Non-backtracking (cavity) depth-2 backup; a learned aggregation (GNN).

## Next experiments (in order)

Full improvement plan, ordered easiest/safest first: `docs/IMPROVEMENTS.md`
(Tier 0: done, except fitting the backup weights by ranking instead of NLL).

1. 20k labels with a step-based training budget (is Bellman's 20k failure a
   budget problem?).
2. 3 seeds for: categorical vs ordinal+Bellman, each with and without backup.
3. ~~Ranking-rule comparison on existing checkpoints~~ (done: finding 9).
4. Backup heuristic inside beam search, with a cache (≈1× cost).
5. The full preset grid (paper table), about 10 h on the M1.

## Resume

```bash
source .venv/bin/activate
pytest -m "not slow"                                   # sanity check
python -m eqcube.data.report --results results/pilot   # per-depth tables
python -m eqcube.experiments.hard_band                 # diagnostic grid
python -m eqcube.experiments.bellman                   # heads / Bellman grid
python -m eqcube.experiments.summarize results/bellman
```

## Toward the 3×3×3

Reusable as is: symmetry, G-sets, equivariant layers, search, inference,
evaluation, and task-symmetry computation (feed it O_h; with fixed centers
expect the full 48, then inversion).
Needs new work: the 3×3 puzzle geometry (corners, edges, centres), training
without an exact table (4.3·10¹⁹ states: random-walk labels, Bellman /
value iteration as in DeepCubeA, or self-supervised policies as in
EfficientCube), and test sets labelled by an optimal solver (Korf / Kociemba).
