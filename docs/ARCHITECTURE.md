# eqcube v2 architecture

How the math maps to code, and why each boundary is where it is.

## Layers

```
symmetry/   Perm, PermGroup, Action, GSet          pure Python + NumPy
   ▲
   ├── puzzles/  PermutationPuzzle, pocket cube, symmetry groups   NumPy
   │      ▲
   │      └── search/  Heuristic protocol, BeamSearch               NumPy
   │
   └── nn/       FeatureType, EquivariantLinear, Norm, Pool          torch
          ▲
          └── models/  ValueNet, ModelConfig, registry, checkpoint,  torch
                       NetHeuristic (adapter: model -> Heuristic)
```

Rule: `symmetry`, `puzzles`, and `search` never import torch
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

## The symmetry finding

`task_group()` *computes* the symmetries of the solving task inside O × S6.
The result has order 3: the rotations about the DBL–UFR diagonal, each paired
with the recoloring that undoes it. They cycle R → U → F. Pure rotations
(O, used by v1 and the paper) and S6 do **not** preserve BFS distance in this
environment, because R/U/F never move the DBL corner.
`tests/puzzles/test_pocket_cube.py` checks both claims exhaustively up to depth 5.

## Presets (~40k parameters each, except `mlp`)

| preset | group | input → hidden | notes |
|---|---|---|---|
| `rotation` | O on positions | positions → positions | = v1 `RotValueNet` |
| `rotation_color` | O × S6 | stickers → stickers | = v1 `BothValueNet` (48 orbits) |
| `color` | S6 | stickers → positions | |
| `task` | computed C3 | stickers → positions | the correct prior |
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
