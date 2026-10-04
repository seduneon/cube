# Equivariant MLP for the 2×2×2 Rubik's Cube — Plan v3 (PyTorch, with Color Symmetries)

## Summary of Changes from v2

- Added **color permutation symmetry** (S₆, 720 elements) on top of the 24 spatial rotations
- Full symmetry group: 24 × 720 = **17,280 elements**, still tractable
- Group convolution kernel is now parameterized by **triple-orbit indices** (position_i, position_j, color_perm), giving even fewer free parameters and stronger generalization
- Updated layer implementation, model architecture, and evaluation metrics

---

## Dependencies

```
pip install torch numpy matplotlib tqdm
```

No external equivariance libraries.

---

## File Structure

```
pocket_cube/
├── cube_env.py            # 2×2×2 simulator, moves, state encoding
├── cube_symmetry.py       # Rotation perms + color perms, orbit computation
├── equivariant_layers.py  # GroupConvLayer, InvariantHead in PyTorch
├── models.py              # EquivariantValueNet, MLPValueNet
├── dataset.py             # BFS + backward scramble dataset
├── train.py               # Training loop
├── evaluate.py            # Metrics and plots
└── run_all.py             # Full pipeline orchestrator
```

---

## The Two Symmetries

### Symmetry 1: Spatial rotations (O, 24 elements)

Rotating the whole cube in space permutes the 24 sticker positions. The puzzle difficulty doesn't change. This group acts on **axis 1** of the `(batch, 24, 6)` tensor — it shuffles which physical sticker slot holds which color.

Generators: 90° rotation about R–L axis, 90° rotation about U–D axis. Multiply to get all 24 elements.

### Symmetry 2: Global color permutations (S₆, 720 elements)

Relabeling all colors simultaneously (e.g., every red sticker becomes blue and vice versa) produces a puzzle state that is equally hard to solve — it's the same scramble on a cube with different paint. This group acts on **axis 2** of the `(batch, 24, 6)` tensor — it shuffles which color channel is which.

Generators: the transposition (0,1) and the cyclic permutation (0,1,2,3,4,5) generate all of S₆.

### Combined group: G = O × S₆ (17,280 elements)

The two symmetries act on independent axes and commute, so the combined group is a direct product. A group element (r, σ) with rotation r and color permutation σ acts as:

```
(r, σ) · x[i, c] = x[r⁻¹(i), σ⁻¹(c)]
```

on the `(24, 6)` input tensor.

The value function must be **invariant** to this full group: `v((r,σ) · x) = v(x)`.

---

## Phase 1 — Cube Environment

### File: `cube_env.py`

Identical to v2. State is a length-24 int array, sticker labeling 0–23, 6 moves {R, R', U, U', F, F'} fixing the BLD corner.

**Encoding:**
```python
def encode(state):
    """(24,) int -> (24, 6) float32 one-hot"""
    x = np.zeros((24, 6), dtype=np.float32)
    for i in range(24):
        x[i, state[i]] = 1.0
    return x
```

Note: we now keep the `(24, 6)` shape throughout instead of flattening to 144. This makes the two symmetry axes explicit.

---

## Phase 2 — Symmetry Group and Orbit Computation

### File: `cube_symmetry.py`

### Step 2a: Enumerate spatial rotations

Same as v2. Build 24 permutations of length 24 (one per rotation). Store as `spatial_perms`: numpy array of shape `(24, 24)` where `spatial_perms[g]` is the permutation for rotation g.

Verify: exactly 24 unique permutations, identity is element 0.

### Step 2b: Enumerate color permutations

Generate all 720 elements of S₆. Practical approach:

```python
from itertools import permutations
color_perms = np.array(list(permutations(range(6))))  # shape (720, 6)
```

Each row is a permutation of (0,1,2,3,4,5). Verify: 720 elements, identity is (0,1,2,3,4,5).

### Step 2c: Compute pair-color orbits for the GroupConvLayer

The GroupConvLayer maps `(batch, 24, c_in) -> (batch, 24, c_out)`. Its kernel `K[i, j, co, ci]` must satisfy:

```
K[r(i), r(j), σ(co), σ(ci)] = K[i, j, co, ci]
```

for all rotations r and color permutations σ. (Here σ acts on channel indices if the channels correspond to color-equivariant features; see below for details.)

**How color symmetry constrains the kernel depends on the layer:**

**Input layer (c_in=6, channels = literal color indices):**

The 6 input channels are one-hot color encodings. Color permutation σ permutes these channels. So the kernel's (ci) axis transforms under σ. The output channels are abstract features — they also transform under σ if we want the next layer to be equivariant too.

The cleanest design: **all hidden channels are organized as multiples of the 6-dim color representation**. So c_hidden = 6k for some integer k (e.g., k=10 gives c_hidden=60). Each group of 6 channels transforms as a copy of the color permutation rep. Then:

- Input: 1 copy of ℝ⁶ per position (the one-hot colors)
- Hidden: k copies of ℝ⁶ per position
- σ acts by permuting within each group of 6

This way, the kernel constraint is:
```
K[r(i), r(j), group_o, σ(co_within), group_i, σ(ci_within)] = K[i, j, group_o, co_within, group_i, ci_within]
```

where `co_within` and `ci_within` are indices within a 6-dim color block, and `group_o`, `group_i` are block indices.

**Orbit computation for constrained kernel:**

The free parameters of the kernel live on orbits of the combined group G = O × S₆ acting on tuples `(i, j, ci_within, co_within)`:

Two tuples `(i, j, a, b)` and `(i', j', a', b')` are in the same orbit if there exists `(r, σ)` such that `r(i)=i', r(j)=j', σ(a)=a', σ(b)=b'`.

Since the spatial and color parts act independently:
- Spatial part: orbit of `(i, j)` under the 24 rotations
- Color part: orbit of `(a, b)` under S₆ acting on pairs

The combined orbit is the Cartesian product of these two orbit types. So:

```
n_total_orbits = n_spatial_pair_orbits × n_color_pair_orbits
```

**Color pair orbits under S₆:**
- Pairs `(a, b)` with `a, b ∈ {0,...,5}` — there are 36 total
- Under S₆: orbit of (a, a) (diagonal) has 6 elements → 1 orbit
- Orbit of (a, b) with a≠b has 30 elements → 1 orbit
- **Total: 2 color pair orbits** (same-color and different-color)

**Spatial pair orbits under O:**
- 24×24 = 576 pairs, under the 24-element group
- Expected: 10–30 orbits (compute exactly)

**So the kernel has `n_spatial_orbits × 2` free parameters per (block_out, block_in) pair.** With k_in=1, k_out=k blocks, the input layer has `k × n_spatial_orbits × 2` free weights. A hidden-to-hidden layer has `k² × n_spatial_orbits × 2` free weights.

This is dramatically fewer parameters than an unconstrained layer.

### Implementation of orbit computation:

```python
def compute_combined_orbits(spatial_perms, n_colors=6):
    """
    Compute orbit indices for tuples (pos_i, pos_j, color_a, color_b)
    under the combined group O × S₆.
    
    Since the groups act independently on (pos, color), we compute:
    1. spatial_pair_orbits: (24, 24) array mapping (i,j) -> spatial orbit id
    2. color_pair_orbits: (6, 6) array mapping (a,b) -> color orbit id
       (only 2 orbits: a==b and a!=b)
    3. combined_orbit[i, j, a, b] = spatial_pair_orbits[i,j] * n_color_orbits + color_pair_orbits[a,b]
    
    Returns:
        spatial_pair_orbits: (24, 24) int array
        n_spatial_orbits: int
        n_color_orbits: int (= 2)
        n_total_orbits: int (= n_spatial_orbits * 2)
    """
    n_pos = spatial_perms.shape[1]
    
    # Spatial pair orbits
    spatial_pair_orbit = -np.ones((n_pos, n_pos), dtype=np.int64)
    orbit_id = 0
    for i in range(n_pos):
        for j in range(n_pos):
            if spatial_pair_orbit[i, j] == -1:
                for g in range(len(spatial_perms)):
                    gi = spatial_perms[g][i]
                    gj = spatial_perms[g][j]
                    spatial_pair_orbit[gi, gj] = orbit_id
                orbit_id += 1
    n_spatial_orbits = orbit_id
    
    # Color pair orbits: just 2 (diagonal vs off-diagonal)
    color_pair_orbit = np.ones((n_colors, n_colors), dtype=np.int64)  # off-diagonal = 1
    np.fill_diagonal(color_pair_orbit, 0)  # diagonal = 0
    n_color_orbits = 2
    
    return spatial_pair_orbit, n_spatial_orbits, color_pair_orbit, n_color_orbits
```

---

## Phase 3 — Equivariant Layers

### File: `equivariant_layers.py`

### GroupConvLayer with color symmetry

```python
class GroupConvLayer(nn.Module):
    def __init__(self, k_in, k_out, spatial_pair_orbit, n_spatial_orbits, n_color_orbits=2):
        """
        Maps (batch, 24, 6*k_in) -> (batch, 24, 6*k_out).
        
        Channels are organized as k groups of 6 (one per color rep copy).
        
        Kernel is parameterized by:
          weight: (k_out, k_in, n_spatial_orbits, n_color_orbits)
        
        Total free parameters: k_out * k_in * n_spatial_orbits * n_color_orbits
        
        For comparison, unconstrained would be:
          k_out*6 * k_in*6 * 24*24 = k_out*k_in * 36 * 576 parameters
        
        spatial_pair_orbit: (24, 24) int array of orbit indices
        """
        super().__init__()
        self.k_in = k_in
        self.k_out = k_out
        self.n_sp = n_spatial_orbits
        self.n_co = n_color_orbits
        self.register_buffer('sp_orbit', torch.from_numpy(spatial_pair_orbit).long())
        
        # color_pair_orbit: (6, 6), just diagonal(0) vs off-diagonal(1)
        co_orbit = torch.ones(6, 6, dtype=torch.long)
        co_orbit.fill_diagonal_(0)
        self.register_buffer('co_orbit', co_orbit)
        
        self.weight = nn.Parameter(
            torch.randn(k_out, k_in, n_spatial_orbits, n_color_orbits) * 0.02
        )
        # Bias: one per output block (invariant under color perm within each block)
        self.bias = nn.Parameter(torch.zeros(k_out))
    
    def forward(self, x):
        """
        x: (batch, 24, 6*k_in)
        out: (batch, 24, 6*k_out)
        """
        B = x.shape[0]
        x = x.view(B, 24, self.k_in, 6)  # (B, pos, block_in, color)
        
        # Build kernel: (k_out, k_in, 24, 24, 6, 6) from weight via orbit lookup
        # kernel[ko, ki, i, j, co, ci] = weight[ko, ki, sp_orbit[i,j], co_orbit[co,ci]]
        kernel = self.weight[:, :, self.sp_orbit, :]  # (k_out, k_in, 24, 24, n_co)
        # Now index the color orbit dimension
        # kernel_full[ko, ki, i, j, co, ci] = kernel[ko, ki, i, j, co_orbit[co, ci]]
        kernel_full = kernel[:, :, :, :, self.co_orbit]  # (k_out, k_in, 24, 24, 6, 6)
        
        # Contract: out[b, i, ko, co] = Σ_j Σ_ki Σ_ci kernel_full[ko, ki, i, j, co, ci] * x[b, j, ki, ci]
        out = torch.einsum('bjkc, okijdc -> biod', x, kernel_full)
        # out shape: (B, 24, k_out, 6)
        
        # Add bias (broadcast over batch, position, color)
        out = out + self.bias[None, None, :, None]
        
        return out.reshape(B, 24, self.k_out * 6)
```

**Note on bias:** The bias is shared across all 6 colors within each output block and across all 24 positions. This ensures invariance to both spatial rotations and color permutations. One scalar per output block → k_out bias parameters.

### InvariantHead

```python
class InvariantHead(nn.Module):
    def __init__(self, k_in):
        """
        Maps (batch, 24, 6*k_in) -> (batch, 1) invariantly.
        
        1. Average over 24 positions (spatial invariance)
        2. Average over 6 color channels within each block (color invariance)  
        3. Linear on the k_in block-level features -> scalar
        
        Alternatively:
        1. Average over all 24*6 = 144 entries per block (both invariances at once)
        2. Linear k_in -> 1
        """
        super().__init__()
        self.k_in = k_in
        self.linear = nn.Linear(k_in, 1)
    
    def forward(self, x):
        B = x.shape[0]
        x = x.view(B, 24, self.k_in, 6)
        pooled = x.mean(dim=(1, 3))  # (B, k_in) — average over positions and colors
        return self.linear(pooled)    # (B, 1)
```

---

## Phase 4 — Models

### File: `models.py`

### EquivariantValueNet

```python
class EquivariantValueNet(nn.Module):
    def __init__(self, k_hidden=10, n_layers=3, spatial_pair_orbit=None, n_spatial_orbits=None):
        """
        Input:  (batch, 24, 6)   — one-hot sticker colors, i.e. k_in=1
        Output: (batch, 1)       — predicted normalized distance
        
        Hidden dim = 6 * k_hidden per position (e.g., k_hidden=10 -> 60 channels).
        Total hidden features = 24 * 60 = 1440.
        
        Architecture:
          GroupConvLayer(1, k_hidden) + ReLU
          GroupConvLayer(k_hidden, k_hidden) + ReLU   × (n_layers - 1)
          InvariantHead(k_hidden) -> scalar
        """
        super().__init__()
        layers = []
        layers.append(GroupConvLayer(1, k_hidden, spatial_pair_orbit, n_spatial_orbits))
        layers.append(nn.ReLU())
        for _ in range(n_layers - 1):
            layers.append(GroupConvLayer(k_hidden, k_hidden, spatial_pair_orbit, n_spatial_orbits))
            layers.append(nn.ReLU())
        self.backbone = nn.Sequential(*layers)
        self.head = InvariantHead(k_hidden)
    
    def forward(self, x):
        # x: (batch, 24, 6)
        x = self.backbone(x)
        return self.head(x)
```

### MLPValueNet (baseline)

```python
class MLPValueNet(nn.Module):
    def __init__(self, hidden_dim=512, n_layers=3):
        """
        Flat MLP. No symmetry awareness.
        Input:  (batch, 144) — flattened one-hot
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
        return self.net(x.view(x.shape[0], -1))
```

### Parameter matching

Report parameter counts for both. Adjust `k_hidden` (equivariant) and `hidden_dim` (MLP) to create two comparison pairs:

1. **Matched architecture size**: k_hidden=10 (60 channels, 1440 hidden features) vs hidden_dim=1440. The equivariant model will have far fewer free parameters due to weight sharing.

2. **Matched parameter count**: Find the hidden_dim for the MLP that gives the same number of free params as the equivariant model. This isolates the inductive bias benefit from the regularization benefit of fewer parameters.

---

## Phase 5 — Dataset

### File: `dataset.py`

Identical to v2 except the encoding returns `(24, 6)` instead of `(144,)`.

### BFS

```python
def compute_bfs(moves):
    """
    BFS from solved state. Returns:
      dist:   dict[tuple -> int]     (state -> optimal distance 0–11)
      policy: dict[tuple -> int]     (state -> optimal first move 0–5)
    
    Runtime: ~5–10 min. Memory: ~300 MB.
    """
```

### Dataset generation

```python
def generate_dataset(n_samples, dist_table):
    """
    Backward scrambles from solved state.
    Returns:
      X:      (n_samples, 24, 6) float32    — one-hot encoded states
      y_dist: (n_samples,) float32           — distance / 11 (normalized to [0,1])
    """
```

### Splits

| Split | Size |
|-------|------|
| train_50k | 50,000 |
| train_200k | 200,000 |
| train_1m | 1,000,000 |
| val | 20,000 |
| test | 10,000 (stratified by depth) |

Save as `.pt` files.

---

## Phase 6 — Training

### File: `train.py`

```
Optimizer:       Adam
Learning rate:   1e-3, cosine annealing to 1e-5
Batch size:      512
Max epochs:      200
Early stopping:  patience=15 on val MSE
Loss:            MSE on normalized distance (target in [0, 1])
Seeds:           3 per config (42, 43, 44)
Device:          CPU (should be fast enough; use CUDA if available)
```

### Configurations (18 runs total)

| Config | Model | Train data |
|--------|-------|------------|
| equiv_50k | EquivariantValueNet | 50K |
| equiv_200k | EquivariantValueNet | 200K |
| equiv_1m | EquivariantValueNet | 1M |
| mlp_50k | MLPValueNet | 50K |
| mlp_200k | MLPValueNet | 200K |
| mlp_1m | MLPValueNet | 1M |

### Logging

Per epoch to CSV: train_mse, val_mse, lr, epoch_seconds.

Save best model checkpoint per config+seed.

---

## Phase 7 — Evaluation

### File: `evaluate.py`

### Metric 1: Equivariance error (now tests BOTH symmetries)

```python
def measure_equivariance_error(model, test_states, spatial_perms, color_perms):
    """
    For 500 test states:
      Sample 10 random spatial rotations and 10 random color permutations.
      For each (state, rotation r, color perm σ):
        transformed = apply_rotation_and_color_perm(state, r, σ)
        err = |model(transformed) - model(state)|
      Report mean and max error.
    
    Equivariant model: should be ~1e-7 for spatial rotations AND color perms.
    MLP: will have nonzero error for both.
    """
```

### Metric 2: Distance prediction accuracy

On the 10K test set:
- MAE (in original scale: pred × 11 vs true distance)
- Per-depth MAE (depths 0–11)
- Pearson correlation
- Rounded accuracy: `round(pred × 11) == true_distance`

### Metric 3: Greedy solve rate

```python
def greedy_solve(model, state, max_steps=50):
    """
    Each step: try all 6 moves, pick the one whose successor
    has the lowest predicted distance. Stop when solved or max_steps.
    """
```

Test on 1000 scrambles at depths 4, 7, 10, random.
Report: solve rate %, mean length, mean excess over optimal.

### Metric 4: Data efficiency curve

Val MSE vs dataset size (50K, 200K, 1M) for both models.

### Metric 5: Augmentation comparison (new)

To confirm the equivariant architecture is better than data augmentation, add a third model:

```
mlp_aug_200k: MLPValueNet trained on 200K samples with on-the-fly augmentation
```

The augmentation: for each training sample, apply a random rotation and random color permutation before feeding to the network. This gives the MLP an equivalent amount of symmetry information through data rather than architecture. Compare against `equiv_200k` to isolate the architectural benefit.

### Plots (saved to `results/`)

1. `learning_curves.png` — Train+Val MSE vs epoch at 200K for all 3 models (equiv, mlp, mlp_aug)
2. `data_efficiency.png` — Val MSE vs dataset size for equiv and mlp
3. `per_depth_mae.png` — Grouped bars by depth
4. `equivariance_error.png` — Log-scale bars for spatial error and color error, both models
5. `solve_rates.png` — Grouped bars by scramble depth

---

## Phase 8 — Orchestrator

### File: `run_all.py`

```python
"""
Full pipeline:
1. BFS (cached to bfs_cache.pkl)
2. Generate datasets (cached to data/)
3. Compute symmetry orbits
4. Print orbit statistics (n_spatial_orbits, n_total_orbits, param counts)
5. Train all configs (3 seeds each)
6. Evaluate
7. Generate plots
"""
```

---

## Key Design Decisions

1. **Hidden channels as multiples of 6.** This lets color permutations act cleanly on the hidden representation. Each block of 6 channels transforms as a copy of the color permutation representation. The kernel weight sharing works within and across these blocks.

2. **Only 2 color pair orbits.** Under S₆, all same-color pairs (a, a) are equivalent, and all different-color pairs (a, b) with a≠b are equivalent. So the color part of the kernel has just 2 free parameters — "same color" weight and "different color" weight. This is a massive constraint. Combined with spatial orbits, the total kernel has `n_spatial_orbits × 2` free parameters per block pair.

3. **Bias structure.** The bias is one scalar per output block (k_out values total), shared across positions and colors. This is the maximally constrained choice — it enforces invariance to both symmetries. An alternative would be to have a bias per position but averaged over colors; however, the position-invariant choice is simpler and consistent with the pooling in the output head.

4. **Augmentation baseline.** Adding the MLP-with-augmentation comparison (metric 5) is crucial. Without it, one could argue the equivariant model wins simply because it sees more effective training data. The augmentation baseline controls for this: the MLP sees augmented data (equivalent to 17,280× more examples) but still has unconstrained weights. If the equivariant model still wins, the benefit is architectural, not just data augmentation.

5. **Value network focus.** Same rationale as v2 — the scalar output is invariant under both spatial rotations AND color permutations, making it the cleanest target. A policy head would need careful treatment of how moves transform under whole-cube rotations (the move labels permute), and color permutations don't affect the policy at all.

---

## Expected Results

- **Parameter count:** With k_hidden=10 and ~20 spatial orbits, the equivariant model has roughly `4 × 10 × 10 × 20 × 2 = 16,000` kernel parameters plus biases, versus ~1M+ for the MLP at hidden_dim=512.
- **Equivariance error:** Equivariant model ~1e-7 for both spatial and color transformations. MLP will be ~0.01–1.0.
- **Data efficiency:** Equivariant model at 50K should outperform MLP at 1M due to the 17,280× symmetry factor.
- **Augmentation comparison:** Equivariant model should outperform augmented MLP because exact equivariance is a stronger constraint than approximate symmetry from augmented data, especially on rare states.

---

## Troubleshooting

- **Pair orbit count:** Print `n_spatial_orbits` immediately after computing. For the 24-element cube rotation group on 24 positions, expect 10–40 orbits. If it's 576 (= 24²), the rotation permutations are wrong.
- **Einsum performance:** The 6-index einsum `bjkc, okijdc -> biod` may be slow. Alternatives: (a) precompute the gathered kernel as a `(k_out*6, k_in*6, 24, 24)` dense matrix and use `torch.einsum('bjc, ocij -> bio', ...)`, or (b) reshape to 2D and use `torch.mm`.
- **Memory for kernel:** The full kernel tensor `(k_out, k_in, 24, 24, 6, 6)` with k=10 is `10 × 10 × 24 × 24 × 6 × 6 × 4 bytes ≈ 8 MB`. Fine for CPU.
- **BFS memory:** 3.67M states × ~50 bytes each ≈ 180 MB. If tight, encode states as int64 (base-6 packing of 24 stickers) instead of tuples.
- **Color permutation enumeration:** Don't enumerate all 720 elements explicitly if only the orbit structure matters. Since S₆ acts transitively on pairs, the only thing that matters is whether `a == b` or `a != b`. You can hardcode this instead of looping over 720 permutations.
