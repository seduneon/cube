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

## Open questions

- Which score should rank beam candidates? (Expected value, quantiles,
  dominance, probability of being closest: THEORY.md + the study plan.)
- How to model siblings' shared errors (a latent shift; ensembles)?
- Does Bellman help when labels are scarce, given enough training steps?
- Non-backtracking (cavity) depth-2 backup; a learned aggregation (GNN).

## Next experiments (in order)

1. 20k labels with a step-based training budget (is Bellman's 20k failure a
   budget problem?).
2. 3 seeds for: categorical vs ordinal+Bellman, each with and without backup.
3. Ranking-rule comparison on existing checkpoints (`rank_children`).
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
