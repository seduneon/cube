# eqcube v2 architecture

How the math maps to code, and why each boundary is where it is.

## Layers

```
symmetry/   Perm, PermGroup, Action, GSet          pure Python + NumPy
   ▲
   ├── puzzles/  PermutationPuzzle, pocket cube, coordinates       NumPy
   │      ▲
   │      ├── search/  Heuristic protocol, BeamSearch               NumPy
   │      └── data/    DistanceTable, Partition, samplers, suites   NumPy
   │
   └── nn/       FeatureType, EquivariantLinear, Norm, Pool          torch
          ▲
          └── models/  ValueNet, ModelConfig, registry, checkpoint,  torch
                 ▲     NetPredictor / NetHeuristic (model -> NumPy callables)
                 ├── training/     TrainConfig, Trainer              torch
                 └── experiments/  grid runner, CLI, summary table   glue

evaluation/  value, sibling-ranking, equivariance, solve metrics   NumPy
             (models enter only as callables)
```

Rule: `symmetry`, `puzzles`, `search`, `data`, and `evaluation` never import torch
(enforced by `tests/test_architecture.py`). `search` never imports `models`:
`NetHeuristic` satisfies the `Heuristic` protocol structurally.

## Math → code

| Math | Code | File |
|---|---|---|
| permutation, g∘h | `Perm`, `g * h` = apply h then g | `symmetry/perm.py` |
| group from generators | `PermGroup(gens).elements` (closure BFS) | `symmetry/perm.py` |
| G × H | `direct_product` (acts on disjoint union) | `symmetry/perm.py` |
| action G × X → X | `Action[X]` protocol | `symmetry/action.py` |
| on points / tuples / functions / perms / sets | `Point/Tuple/Function/Conjugation/SetAction` | `symmetry/action.py` |
| orbit, stabilizer | generic `orbit(action, x)`, `stabilizer(action, x)` | `symmetry/action.py` |
| G-set (hom G → S_n) | `GSet` (+ `block`, `trivial`, `product`) | `symmetry/gset.py` |
| equivariant linear map | weights tied on orbits of G on out × in | `nn/layers.py` |
| invariant readout | mean over each point orbit | `nn/layers.py` |
| task symmetry | {g : g·solved = solved, gMg⁻¹ = M} | `puzzles/base.py` |
| cube isometries O, O_h | signed permutation matrices on sticker coordinates | `puzzles/pocket_cube.py` |

## Conventions (decided once, tested)

- `perm.arr[i]` is the image of `i`; `(g*h)(i) == g(h(i))`.
- Acting on arrays/states/features gathers with the inverse:
  `(g·f)[p] = f[g⁻¹(p)]` (`test_naive_gather_is_a_right_action_not_left`).
- A cube state is a function positions → colors. Symmetry groups act on
  24 positions + 6 colors (degree 30) so every group shares one encoding.
- Feature tensors are `(batch, points, channels)`. G permutes points only;
  channels are multiplicity. Colors become *points* (the `stickers` G-set)
  whenever the group moves them.
- `GSet` equality is identity: build once, share. Orbit tables are cached.

## Geometry is the source of truth

`puzzles/pocket_cube.py` gives every sticker a 3D position (corner, face
normal) with R = +x, U = +y, F = +z. A 3×3 signed permutation matrix M maps
sticker (c, n) to (Mc, Mn). From this:

- face turns R, U, F: rotate only the stickers in that layer;
- O = ⟨x-turn, y-turn⟩ (24) and O_h = ⟨x-turn, y-turn, −I⟩ (48).

Moves and symmetries therefore share one convention by construction. The
hand-written v1 cycles are kept only as test oracles, and they match.

## Coordinates (state <-> integer)

`puzzles/pocket_coords.py` maps every state to an index in [0, 3,674,160):

```
index = perm_coord * 729 + twist_coord
        perm_coord  = Lehmer code of the 7 movable pieces (7! = 5040)
        twist_coord = base-3 number of 6 slot twists (the 7th is fixed by sum = 0 mod 3)
```

Twists are recorded per slot, with every slot's stickers in the same
rotational order (`corner_slots`). A move's effect on the permutation then
ignores twists, and its effect on twists ignores which piece is where, so
moves act on indices through two small tables (5040×6 and 729×6), with no
decoding. The full BFS (`eqcube.data.table.bfs`) runs on indices in about 2 s
and stores one `int8` per state (3.6 MB). Tests check the bijections
exhaustively, check the tables against sticker moves, and check the BFS
counts against the known distribution.

## Data: one table, one partition, many samplers and suites

```
DistanceTable (BFS once, cached .npy)
   └── Partition: every index -> train | val | test, by splitmix64 hash + seed
         ├── samplers draw ONLY from train: TemperedSampler(alpha), RandomWalkSampler
         └── suites  draw ONLY from test/val: per_depth, uniform, search starts
```

- The partition never depends on the sampler, so samplers and suites can
  be swapped freely without leakage.
- `policy="random"` hashes each state on its own; `policy="orbit"` hashes its
  S3-orbit representative, so symmetric copies stay on the same side.
- States at depth <= 4 (688 of them) are shared by every split: they can be
  enumerated exhaustively, so held-out generalization means nothing there.
- `TemperedSampler`: p(d) ∝ n_d^alpha (1 = uniform states, 0.5 = v1 sqrt,
  0 = uniform depths). `sample_unique` builds fixed training sets: small
  depths are taken whole, and the rest is redistributed.
- Random walks rarely reach depth 14 (0 of 10k with length ≤ 20), a known
  weakness of walk-based data.

## Training and evaluation

- `Trainer` trains on a fixed set of `train_size` distinct states (the
  tempered sampler) with AdamW and cosine LR, early-stops on cross-entropy
  over the **per-depth** validation suite, and stores the best checkpoint
  with model config, train config, and metrics.
- Evaluation reports metrics overall and per depth:
  value (CE, accuracy, MAE), **sibling ranking** (does the heuristic rank
  d−1 children above d+1 ones, and is the lowest-cost child a d−1 one: what
  greedy and beam search rely on), equivariance error, and beam-search solve
  rate and excess length, for both `argmax` and `expected` heuristics.
- Runs are resumable (`result.json` per run), so a grid can move between
  machines. `python -m eqcube.experiments.run --quick` checks a machine in
  about 15 s.

Fact found while testing: **local maxima** (states where every move goes
closer to solved) are 51% of depth 12, 98% of depth 13, and all of depth 14.
The hard ranking decisions are at middle depths.

## Compute (measured on an M1 MacBook Air, batch 512)

| preset | CPU | MPS | 1M-state epoch (MPS) |
|---|---|---|---|
| task, color | 130–140k/s | 190k/s | ~5 s |
| rotation | 47k/s | 74k/s | ~14 s |
| mlp_small, mlp | 230–430k/s | 330–350k/s | ~3 s |
| rotation_color | 4k/s | 6k/s | **~175 s** |

`rotation_color` is slow because `EquivariantLinear` materializes the full
144×144 kernel. The Kronecker (α/β) fast path for direct-product groups is the
fix (see "Known costs").

## The symmetry finding

`task_symmetry(spatial)` *computes* the symmetries of the solving task:
g must keep the solved state solved (this fixes g's recoloring) and must map
the move set to itself. Because the recoloring is determined by the spatial
part, the search scans |O_h| = 48 candidates instead of |O_h × S6| = 34,560;
the brute-force filter is kept as a test oracle.

Result: **S3 (6 elements)**. Look at the cube from the UFR corner: the faces
U, F, R form a triangle, and S3 is that triangle's symmetry group.

- 3 rotations about the DBL–UFR diagonal cycle R → U → F;
- 3 reflections in planes containing that diagonal send each move to an
  inverse move (R → F', U → U', F → R', …).

Pure rotations (O, used by v1 and the paper), S6, and every other element of
O_h × S6 do **not** preserve BFS distance in this environment, because R/U/F
never move the DBL corner. `tests/data/test_data.py` checks that S3 preserves
the distance of **all 3,674,160 states** (marked `slow`), and
`tests/puzzles/test_pocket_cube.py` checks that every other element fails.

Not yet used: **inversion**, d(g) = d(g⁻¹), which would double the group to
12. It is not a permutation of sticker-color features, so it needs group
averaging (with a colors → sticker-permutation decoder) or a piece encoding.

## Presets (~40k parameters each, except `mlp`)

| preset | group | input → hidden | notes |
|---|---|---|---|
| `rotation` | O on positions | positions → positions | = v1 `RotValueNet` |
| `rotation_color` | O × S6 | stickers → stickers | = v1 `BothValueNet` (48 orbits) |
| `color` | S6 | stickers → positions | |
| `isometry_color` (symmetry only) | O_h × S6 | — | wrong prior, available via config |
| `task` | computed S3 | stickers → positions | the correct prior |
| `mlp`, `mlp_small` | none | one-hot → MLP | baselines |

## How to extend

- **New symmetry**: add a `() -> PermGroup` of degree P + C to `SYMMETRIES`.
  The layers need no changes.
- **New architecture**: write a builder `(ModelConfig) -> nn.Module` and
  decorate it with `@register_arch("name")`.
- **New puzzle**: build a `PermutationPuzzle` and register a `PuzzleSpec`
  in `puzzles/__init__.py`.
- **New search**: any class with `solve(puzzle, starts) -> list[SolveResult]`.
- **New heuristic**: any callable `states -> costs`.

## Known costs / next optimizations

- `EquivariantLinear` materializes the full kernel `(c_out, c_in, n_out, n_in)`
  on every forward pass. That is fine for positions (24) but heavy for
  stickers × stickers (144²). Options: cache the kernel in eval mode, or use
  the Kronecker fast path for direct-product groups (v1's α/β trick).
- `PermGroup` stores every element. That is fine up to about 10⁵; beyond
  that, Schreier–Sims.
- `NetHeuristic` defaults to `argmax` (v1 behaviour); `expected` may rank
  children better.
- Not built yet: data generation, the training loop, and experiment scripts.
