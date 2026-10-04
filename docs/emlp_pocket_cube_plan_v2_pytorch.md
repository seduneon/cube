# Equivariant MLP for the 2×2×2 Rubik's Cube — Revised Plan (Pure PyTorch)

## Why This Revision

The `emlp` library is outdated and has brittle JAX/objax version dependencies. This plan implements the same equivariant architecture from scratch in PyTorch. The math is simple — we just need to project weight matrices onto the symmetry-respecting subspace. No external equivariance libraries required.

---

## Core Math (What EMLP Was Doing)

Given a finite group G with representation ρ(g) (a permutation matrix for each of the 24 cube rotations), an equivariant linear map W must satisfy:

```
ρ_out(g) @ W = W @ ρ_in(g)    for all g in G
```

This means W lives in a constrained subspace. We find this subspace once at initialization by computing the **equivariant projector**:

```
P = (1/|G|) * Σ_g  (ρ_out(g)^T ⊗ ρ_in(g))
```

Concretely, flatten W into a vector w. Then `w_equivariant = P @ w` gives the nearest equivariant weight matrix. At init, we compute P (a matrix), find its column space via SVD to get a basis Q of shape `(n_out * n_in, rank)`. Then we parameterize the layer with `rank` free parameters α, and reconstruct W = reshape(Q @ α, (n_out, n_in))`.

For **invariant** outputs (scalar value head), ρ_out(g) = 1 for all g, so the constraint simplifies to `W @ ρ_in(g) = W` — the rows of W must be invariant vectors.

For bias vectors b in an invariant output layer: b is unconstrained (a scalar is always invariant). For equivariant layers with non-trivial output rep: `b_equivariant = P_bias @ b` where `P_bias = (1/|G|) Σ ρ_out(g)`.

---

## Dependencies

```
pip install torch numpy matplotlib tqdm
```

That's it. No JAX, no objax, no emlp.

---

## File Structure

```
pocket_cube/
├── cube_env.py            # 2×2×2 simulator, moves, state encoding
├── cube_symmetry.py       # 24 rotation matrices, projector computation
├── equivariant_layers.py  # EquivariantLinear, InvariantLinear in PyTorch
├── models.py              # EquivariantMLP and baseline MLP
├── dataset.py             # BFS + backward scramble dataset
├── train.py               # Training loop
├── evaluate.py            # Metrics and plots
└── run_all.py             # Full pipeline orchestrator
```

---

## Phase 1 — Cube Environment

### File: `cube_env.py`

**State representation.** Length-24 int array, `state[i]` = color (0–5) of sticker position i. Solved: `state[i] = i // 4`.

**Sticker labeling.** 6 faces × 4 stickers, numbered 0–23:
```
U: 0  1  2  3        (top face, reading order looking down)
D: 4  5  6  7        (bottom face, reading order looking up from below)
F: 8  9  10 11       (front face)
B: 12 13 14 15       (back face)
L: 16 17 18 19       (left face)
R: 20 21 22 23       (right face)
```

**Moves.** Fix the BLD corner. 3 generators + 3 inverses = 6 moves: R, R', U, U', F, F'.

Each move is stored as a length-24 permutation array `perm` where `new_state[i] = old_state[perm[i]]`.

Derive these permutations carefully from the physical cube. Cross-check: SymPy defines the 2×2×2 generators (adapted to our labeling). Test that each move applied 4 times returns to identity.

**Encoding:**
```python
def encode(state):
    """(24,) int -> (144,) float32 one-hot"""
    out = np.zeros(144, dtype=np.float32)
    for i in range(24):
        out[i * 6 + state[i]] = 1.0
    return out
```

**Key functions:**
```python
def apply_move(state, move_idx) -> state
def is_solved(state) -> bool
def scramble(n_moves, rng) -> (state, move_list)
```

---

## Phase 2 — Rotation Group and Equivariant Projectors

### File: `cube_symmetry.py`

**Objective:** Enumerate the 24 spatial rotations as permutations of the 24 sticker positions, then compute the equivariant projectors for the neural network layers.

### Step 2a: Build the 24 rotation permutations

Two generators produce all 24 rotations:

- **x_rot:** 90° rotation about the R–L axis (R face stays R, L face stays L, U→F→D→B cycle). Determine exactly which sticker index maps to which.
- **y_rot:** 90° rotation about the U–D axis (U stays U, D stays D, F→R→B→L cycle).

Multiply these generators in all combinations, collecting unique permutations until we have exactly 24. Store as `rotations`: a list of 24 permutation arrays, each of length 24.

**Verification:**
```python
assert len(set(tuple(r) for r in rotations)) == 24
for r in rotations:
    assert apply_and_return_4_times(r) == identity  # order divides 24
```

### Step 2b: Lift permutations to the 144-dim encoding space

Each rotation permutes sticker positions. In the one-hot encoding (24 stickers × 6 colors = 144 dims), this becomes a 144×144 permutation matrix:

```python
def sticker_perm_to_encoding_perm(perm_24):
    """
    perm_24: length-24 array, perm_24[i] = where sticker i goes
    Returns: 144x144 permutation matrix acting on the one-hot encoding
    
    If sticker position i moves to position perm_24[i], then the 6-dim 
    one-hot block at position i in the input moves to position perm_24[i] 
    in the output. So for the 144-dim vector:
      new_vec[perm_24[i]*6 + c] = old_vec[i*6 + c]  for c in 0..5
    """
    P = np.zeros((144, 144))
    for i in range(24):
        for c in range(6):
            P[perm_24[i] * 6 + c, i * 6 + c] = 1.0
    return P
```

Store: `rho_in`: list of 24 matrices, each 144×144. These are the group representations acting on the network input.

### Step 2c: Compute equivariant basis for each layer type

**For a hidden equivariant layer** (input dim `d_in`, output dim `d_out`, both transforming under the same group action):

```python
def compute_equivariant_basis(rho_in_list, rho_out_list, d_in, d_out):
    """
    rho_in_list:  list of |G| matrices, each (d_in, d_in)
    rho_out_list: list of |G| matrices, each (d_out, d_out)
    
    Compute projector P = (1/|G|) Σ_g kron(rho_out(g), rho_in(g))
    acting on vectorized weight matrices of shape (d_out * d_in,).
    
    Then SVD of P to get the basis Q of the equivariant subspace.
    Q has shape (d_out * d_in, rank) where rank = dim of equivariant subspace.
    
    Returns Q (the basis matrix).
    """
    G = len(rho_in_list)
    dim = d_out * d_in
    P = np.zeros((dim, dim))
    for g in range(G):
        # For equivariance constraint rho_out(g) W = W rho_in(g):
        # Vectorized: (rho_out(g) ⊗ rho_in(g)^T) vec(W) = vec(W)
        # So projector = (1/|G|) Σ kron(rho_out(g), rho_in(g)^T)
        P += np.kron(rho_out_list[g], rho_in_list[g].T)
    P /= G
    
    # SVD to find column space (eigenvalues ≈ 1)
    U, S, Vt = np.linalg.svd(P)
    rank = np.sum(S > 0.5)  # eigenvalues of projector are 0 or 1
    Q = U[:, :rank]
    return Q  # shape (d_out * d_in, rank)
```

**For the invariant output layer** (scalar output, d_out=1):
```python
rho_out_list = [np.array([[1.0]])] * 24  # trivial representation
```
This simplifies the constraint: each row of W must be an invariant vector.

**For bias vectors:**
```python
def compute_bias_basis(rho_out_list, d_out):
    """Equivariant basis for the bias vector."""
    P = np.zeros((d_out, d_out))
    for g in range(len(rho_out_list)):
        P += rho_out_list[g]
    P /= len(rho_out_list)
    U, S, Vt = np.linalg.svd(P)
    rank = np.sum(S > 0.5)
    return U[:, :rank]
```

### Step 2d: Precompute and cache

Compute the equivariant bases for each layer shape needed by the network. These are computed once and stored as numpy arrays. For a 3-layer network with hidden dim `h`:

| Layer | Input dim | Output dim | Weight basis shape |
|-------|-----------|------------|--------------------|
| Layer 1 | 144 | h | (144 * h, rank_1) |
| Layer 2 | h | h | (h * h, rank_2) |
| Layer 3 | h | h | (h * h, rank_2) — same as layer 2 |
| Output | h | 1 | (h, rank_out) |

**Important practical note:** For hidden layers, we need the group to act on the hidden space too. The simplest approach: let the hidden representation be `k` copies of the 24-dim permutation representation, so hidden dim = 24k. With k=16, hidden dim = 384. Then `rho_hidden(g)` is block-diagonal with k copies of the 24×24 permutation matrix.

This means:
- `rho_hidden(g) = I_k ⊗ rho_24(g)` — a (24k × 24k) permutation matrix
- The 144-dim input is 6 copies of the 24-dim rep (k_in = 6)
- The 1-dim output is a scalar (trivial rep)

The **projector computation** for a (384 × 384) hidden-to-hidden layer involves a (384² × 384²) = (147456 × 147456) matrix, which is way too large.

### Efficient approach: exploit the block structure

Since representations are copies of the base 24-dim rep, the equivariant basis has a known structure. A linear map from `k_in` copies of ρ to `k_out` copies of ρ is parameterized by a `(k_out × k_in)` matrix of scalars (by Schur's lemma, since ρ is likely irreducible or decomposes into a known set of irreps).

**Practical simplification:**

For the permutation representation of the 24-element rotation group on 24 stickers, decompose it into irreducible representations (irreps). The 24-dim permutation rep decomposes as:
```
ρ_24 = trivial ⊕ (other irreps of the rotation group)
```

The equivariant maps between copies of ρ_24 are then block-diagonal in the irrep basis, with free scalar coefficients within each irrep block (Schur's lemma).

**Implementation strategy:**

1. Decompose ρ_24 into irreps: Find the change-of-basis matrix `C` such that `C @ ρ_24(g) @ C^T` is block-diagonal for all g.
2. In the irrep basis, equivariant linear maps have a specific sparse structure: within each irrep block of multiplicity m and dimension d, the map is `A ⊗ I_d` where A is a free (m × m) matrix.
3. For k_in copies of ρ_24 → k_out copies of ρ_24, the total free parameters per irrep of dimension d and multiplicity m is: (k_out * m) × (k_in * m) free parameters, shared across the d dimensions of the irrep.

This avoids ever forming the full projector matrix.

**Alternatively (simpler, slightly less efficient):** Work with the 24-dim base representation directly. Build equivariant layers as maps ℝ^(24 × c_in) → ℝ^(24 × c_out) where c_in, c_out are channel dimensions. The equivariant constraint on W means:

```python
class EquivariantLinear(nn.Module):
    """
    Maps (batch, 24, c_in) -> (batch, 24, c_out).
    Equivariant to the 24-element rotation group permuting the 24 sticker positions.
    
    By Schur's lemma, this decomposes into:
    1. A per-position linear map: (c_in -> c_out) applied identically to each of the 24 positions
       (this is the "pointwise" part — same weight matrix at every sticker position)
    2. A "mixing" part that mixes information across positions, constrained by symmetry
    
    The simplest equivariant architecture: use the group convolution approach.
    For each pair of positions (i, j), the weight depends only on the 
    "relative transformation" between i and j under the group.
    """
```

### Recommended architecture: Group Convolution

This is cleaner than computing raw projectors. For the 24-element rotation group G acting on 24 positions:

```python
class GroupConvLayer(nn.Module):
    def __init__(self, c_in, c_out, group_permutations):
        """
        group_permutations: (24, 24) array — for each group element g,
            group_permutations[g] is the permutation it induces on 24 positions.
        
        Learnable parameters: weight matrix of shape (c_out, c_in, n_orbits)
        where n_orbits is the number of orbits of G acting on pairs (i, j).
        
        For each pair (i, j), find which orbit it belongs to, and use the 
        corresponding weight.
        """
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        
        # Precompute: for each pair (i,j), which orbit does it belong to?
        # Two pairs (i,j) and (i',j') are in the same orbit if there exists g 
        # such that g(i)=i' and g(j)=j'.
        n = 24
        pair_to_orbit = {}
        orbit_id = 0
        self.pair_orbit = np.zeros((n, n), dtype=np.int64)
        
        for i in range(n):
            for j in range(n):
                if (i, j) not in pair_to_orbit:
                    # Find all pairs in the same orbit
                    for g in range(len(group_permutations)):
                        gi = group_permutations[g][i]
                        gj = group_permutations[g][j]
                        pair_to_orbit[(gi, gj)] = orbit_id
                    orbit_id += 1
                self.pair_orbit[i, j] = pair_to_orbit[(i, j)]
        
        self.n_orbits = orbit_id
        self.weight = nn.Parameter(torch.randn(c_out, c_in, self.n_orbits) * 0.01)
        self.bias = nn.Parameter(torch.zeros(c_out))
    
    def forward(self, x):
        """
        x: (batch, 24, c_in)
        returns: (batch, 24, c_out)
        
        For each output position i and output channel:
          out[b, i, co] = Σ_j Σ_ci  weight[co, ci, orbit(i,j)] * x[b, j, ci] + bias[co]
        """
        # Build the (24, 24, c_out, c_in) kernel from weight and pair_orbit
        # kernel[i, j] = weight[:, :, pair_orbit[i, j]]
        kernel = self.weight[:, :, self.pair_orbit]  # (c_out, c_in, 24, 24)
        # Einsum: batch, position_j, c_in  ×  c_out, c_in, position_i, position_j -> batch, position_i, c_out
        out = torch.einsum('bjd, odjk -> bko', x, kernel) + self.bias
        # Actually let me write this more carefully:
        # kernel shape: (c_out, c_in, 24_i, 24_j)
        # x shape: (batch, 24_j, c_in)
        # out[b, i, co] = sum_j sum_ci kernel[co, ci, i, j] * x[b, j, ci]
        out = torch.einsum('bjc, ocij -> bio', x, kernel) + self.bias
        return out
```

**Number of orbits:** For the 24-element rotation group acting on 24 positions, the number of orbits on pairs determines the number of free weight parameters per (c_in, c_out) pair. This is typically much smaller than 24×24=576. Compute it and report.

### Invariant output layer

```python
class InvariantLinear(nn.Module):
    """Maps (batch, 24, c_in) -> (batch, 1) invariantly.
    
    Average-pool over the 24 positions (this is always invariant to permutations),
    then apply a standard linear layer on the c_in channels.
    """
    def __init__(self, c_in):
        super().__init__()
        self.linear = nn.Linear(c_in, 1)
    
    def forward(self, x):
        # x: (batch, 24, c_in)
        pooled = x.mean(dim=1)  # (batch, c_in) — invariant to position permutation
        return self.linear(pooled)  # (batch, 1)
```

---

## Phase 3 — Models

### File: `models.py`

### Equivariant model

```python
class EquivariantValueNet(nn.Module):
    def __init__(self, c_hidden=64, n_layers=3, group_perms=None):
        """
        Input: (batch, 24, 6) — one-hot sticker colors
        Output: (batch, 1) — predicted distance to solved
        
        Architecture:
          GroupConvLayer(6, c_hidden)  + ReLU
          GroupConvLayer(c_hidden, c_hidden) + ReLU   (repeated n_layers-1 times)
          InvariantLinear(c_hidden) -> scalar
        """
        super().__init__()
        layers = [GroupConvLayer(6, c_hidden, group_perms), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.append(GroupConvLayer(c_hidden, c_hidden, group_perms))
            layers.append(nn.ReLU())
        self.backbone = nn.Sequential(*layers)
        self.head = InvariantLinear(c_hidden)
    
    def forward(self, x):
        # x: (batch, 144) -> reshape to (batch, 24, 6)
        x = x.view(-1, 24, 6)
        x = self.backbone(x)
        return self.head(x)
```

### MLP baseline

```python
class MLPValueNet(nn.Module):
    def __init__(self, hidden_dim=384, n_layers=3):
        """
        Same total hidden capacity, no symmetry constraints.
        Input: (batch, 144) flattened
        Output: (batch, 1)
        """
        super().__init__()
        layers = [nn.Linear(144, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)
```

### Parameter matching

The equivariant model with c_hidden=64 has hidden dim 24×64 = 1536 positions×channels, but far fewer free parameters because the group convolution kernel has only `n_orbits` weights per (c_in, c_out) pair instead of 24×24.

Report and compare:
- `sum(p.numel() for p in model.parameters())`
- Effective degrees of freedom (the equivariant model's true free parameters)

Choose c_hidden for the equivariant model and hidden_dim for the MLP so that they have **roughly comparable parameter counts**, to make the comparison fair. Alternatively, run both at matched architecture size (same hidden dim) AND at matched parameter count, and report both.

---

## Phase 4 — Dataset

### File: `dataset.py`

### BFS for ground truth

```python
def compute_bfs():
    """
    BFS from solved state over all 3,674,160 reachable states.
    Returns:
        dist: dict mapping state_tuple -> optimal_distance (int, 0-11)
        policy: dict mapping state_tuple -> optimal_first_move (int, 0-5)
    
    Uses the 6 moves {R, R', U, U', F, F'}.
    States are stored as tuples for hashing.
    
    Memory: ~300 MB for the dicts. Runtime: ~5-10 min.
    """
```

The policy label for each state: any move that leads to a neighbor with distance d-1. If multiple such moves exist, pick one arbitrarily (or store all and sample during training).

### Training data

```python
def generate_dataset(n_samples, dist_table, policy_table):
    """
    1. Start from solved state
    2. Apply k random moves (k ~ Uniform(1, 20))
    3. Look up true distance and optimal move from BFS tables
    4. Encode state as one-hot
    
    Returns:
        X: (n_samples, 144) float32
        y_dist: (n_samples,) float32 — normalized to [0, 1] by dividing by 11
        y_move: (n_samples,) int64 — move index 0-5
    """
```

**Distance normalization:** Divide by 11 (max distance) so the value target is in [0, 1]. This helps training stability. The model predicts a value in [0, 1], multiply by 11 to get the distance estimate.

### Datasets to generate

| Split | Size | Purpose |
|-------|------|---------|
| train_50k | 50,000 | Data efficiency experiment |
| train_200k | 200,000 | Data efficiency experiment |
| train_1m | 1,000,000 | Full training |
| val | 20,000 | Early stopping |
| test | 10,000 | Final evaluation (stratified by depth) |

Save as `.pt` files (PyTorch tensors).

---

## Phase 5 — Training

### File: `train.py`

```
Optimizer:       Adam
Learning rate:   1e-3, cosine annealing to 1e-5
Batch size:      512
Max epochs:      200
Early stopping:  patience=15 on val MSE
Loss:            MSE on normalized distance
Seeds:           3 runs per config (seeds 42, 43, 44)
Device:          CPU (small enough)
```

### Training configurations

| Config | Model | Train data | 
|--------|-------|------------|
| equiv_50k | EquivariantValueNet | 50K |
| equiv_200k | EquivariantValueNet | 200K |
| equiv_1m | EquivariantValueNet | 1M |
| mlp_50k | MLPValueNet | 50K |
| mlp_200k | MLPValueNet | 200K |
| mlp_1m | MLPValueNet | 1M |

3 seeds each = 18 training runs total.

### Logging

Per epoch: train_mse, val_mse, lr, epoch_time_seconds.
Save to CSV: `results/{config}_seed{s}_log.csv`

### Checkpointing

Save best model (lowest val MSE): `results/{config}_seed{s}_best.pt`

---

## Phase 6 — Evaluation

### File: `evaluate.py`

### Metric 1: Equivariance error

```python
def measure_equivariance_error(model, test_states, rotation_perms_144):
    """
    For 500 test states, 24 rotations each:
      err = |f(g·x) - f(x)|
    Report mean and max.
    
    EquivariantValueNet should give ~1e-7 (float precision).
    MLPValueNet will give something larger.
    """
```

### Metric 2: Distance prediction quality

On full test set:
- MAE (in original scale, multiply by 11)
- Per-depth MAE (depths 0 through 11)
- Pearson correlation
- Rounded accuracy: `round(pred * 11) == true_distance`

### Metric 3: Greedy solve rate

```python
def greedy_solve(model, state, max_steps=50):
    """
    At each step:
    1. Apply all 6 moves to get 6 successor states
    2. Predict distance for each successor
    3. Move to the successor with lowest predicted distance
    4. Stop if solved or max_steps reached
    """
```

Test on 1000 scrambles at depths 4, 7, 10, and random (uniform over all states).
Report: solve rate, mean solution length, mean excess over optimal.

### Metric 4: Data efficiency

Plot val MSE (y-axis) vs training set size (x-axis, log scale) for both models. Three seeds give error bars.

### Plots to generate (matplotlib, saved to `results/`)

1. `learning_curves.png` — Train+Val MSE vs epoch at 200K, both models
2. `data_efficiency.png` — Val MSE vs dataset size, both models
3. `per_depth_mae.png` — Grouped bar chart, depth on x-axis
4. `equivariance_error.png` — Bar chart (log y), both models
5. `solve_rates.png` — Grouped bars by scramble depth, both models

---

## Phase 7 — Orchestrator

### File: `run_all.py`

Runs the full pipeline in order:

```python
# 1. BFS (or load cached)
# 2. Generate datasets (or load cached)
# 3. Train all configurations
# 4. Evaluate all trained models
# 5. Generate plots
```

Cache BFS results and datasets to disk so re-runs skip expensive steps.

Expected total runtime: ~30-60 min on CPU.

---

## Key Design Decisions

1. **Group convolution instead of raw projectors.** Computing the full equivariant projector for a 384×384 layer requires a (147K × 147K) matrix — impractical. Group convolution on the (batch, 24, channels) tensor achieves the same equivariance with a small kernel parameterized by pair-orbit indices. This is both memory-efficient and conceptually cleaner.

2. **Invariant output via average pooling.** Averaging over the 24 positions is the simplest invariant operation. It's provably invariant to any permutation of positions, and it preserves all the channel information. A learned invariant pooling (with attention weights constrained to be invariant) is possible but unnecessary for this experiment.

3. **Value network only.** The policy head (predicting which of 6 moves to make) has a complicated transformation law under rotations — the move labels themselves permute. This requires careful handling of the output representation. For the experiment, we focus on the value head (distance prediction) and use it for greedy solving. This isolates the equivariance benefit cleanly.

4. **Rotation group, not cube group.** We impose equivariance to the 24-element spatial rotation group (whole-cube rotations), not the 3.67M-element cube move group. The rotation group captures the physical symmetry we want: rotating the whole cube doesn't change the puzzle difficulty.

5. **Fair comparison.** We compare at both matched architecture (same hidden dim → equivariant model has fewer params) and matched parameters (adjust hidden dims so param counts are similar). This distinguishes "equivariance helps because fewer params to overfit" from "equivariance helps because of the right inductive bias."

---

## Expected Results

- **Param count:** EquivariantValueNet should have 5–20× fewer parameters than MLPValueNet at the same hidden dimensionality, depending on the number of pair orbits.
- **Data efficiency:** Equivariant model at 50K samples should match or beat MLP at 200K–1M.
- **Equivariance error:** ~1e-7 for equivariant model, ~0.1–1.0 for MLP.
- **Solve rate:** Both >90% on shallow scrambles; equivariant model should maintain higher solve rates on deep scrambles (depth 10+) especially with small training sets.
- **Per-depth MAE:** Largest equivariant advantage at depths 8–11 where training data is sparsest and the MLP hasn't seen enough rotated variants.

---

## Troubleshooting Notes

- **BFS memory:** If 3.67M dict entries exceed RAM, encode states as integers (base-6 number from 24 stickers = fits in int64) instead of tuples. Or use a numpy array indexed by a perfect hash.
- **Pair orbit computation:** For 24 positions and 24 group elements, there are at most 24×24=576 pairs. The number of orbits should be much smaller (likely 10–30). Print this number as a sanity check.
- **Training speed:** GroupConvLayer's einsum may be slow. If so, precompute the (24, 24) → orbit index mapping as a sparse gather operation and use `torch.index_select` instead of building the full kernel tensor.
- **Numerical precision:** Use float32 throughout. The equivariance error test should show ~1e-7, not exactly 0 — this is expected from floating point.
