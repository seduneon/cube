# Training Specification — Equivariant MLP for the 2×2×2 Rubik's Cube

This document covers everything needed to implement `train.py` and `evaluate.py`: hyperparameters, training loop, configurations, evaluation metrics, and output format.

---

## Hyperparameters

```
Optimizer:           Adam (torch.optim.Adam)
Learning rate:       1e-3 initial
LR schedule:         Cosine annealing to 1e-5 (torch.optim.lr_scheduler.CosineAnnealingLR)
Batch size:          512
Max epochs:          200
Early stopping:      patience = 15 epochs on val MSE (restore best weights)
Loss function:       MSE on normalized distance
Distance norm:       target = true_bfs_distance / 11  (so targets lie in [0, 1])
Weight init:         PyTorch defaults (Kaiming uniform for Linear, 0.02 * randn for GroupConvLayer)
Gradient clipping:   None (add if training is unstable, clip at 1.0)
Seeds:               3 independent runs per config: seeds 42, 43, 44
Device:              CPU by default; use CUDA if available
```

---

## Models to Train

### EquivariantValueNet

```
Architecture:    GroupConvLayer(1, k) + ReLU → [GroupConvLayer(k, k) + ReLU] × (n_layers-1) → InvariantHead(k)
k_hidden:        10  (so hidden channels = 60 per position, hidden features = 24 × 60 = 1440)
n_layers:        3
Input shape:     (batch, 24, 6)
Output shape:    (batch, 1)
```

Requires precomputed `spatial_pair_orbit` array and `n_spatial_orbits` from `cube_symmetry.py`.

### MLPValueNet

```
Architecture:    Linear(144, h) + ReLU → [Linear(h, h) + ReLU] × (n_layers-1) → Linear(h, 1)
hidden_dim:      512
n_layers:        3
Input shape:     (batch, 144)  — flattened from (24, 6)
Output shape:    (batch, 1)
```

### MLPValueNet with augmentation (third model)

Same architecture as MLPValueNet. During training only, each sample is augmented on-the-fly:

```python
def augment(x, spatial_perms, rng):
    """
    x: (24, 6) tensor — single state
    
    1. Apply a random spatial rotation: permute the 24 sticker positions
    2. Apply a random color permutation: permute the 6 color channels
    
    Returns: augmented (24, 6) tensor
    """
    # Random spatial rotation
    g = rng.randint(0, 24)
    x = x[spatial_perms[g]]           # permute rows (sticker positions)
    
    # Random color permutation
    sigma = rng.permutation(6)
    x = x[:, sigma]                    # permute columns (colors)
    
    return x
```

Apply `augment` to every sample in the batch before the forward pass. This is equivalent to training on a dataset 17,280× larger. At evaluation time, no augmentation is applied — the model sees the original state.

---

## Training Configurations

18 runs total (6 configs × 3 seeds):

| Config ID | Model | Train set | Augmentation | Seeds |
|-----------|-------|-----------|--------------|-------|
| `equiv_50k` | EquivariantValueNet | train_50k (50K) | None (built into architecture) | 42, 43, 44 |
| `equiv_200k` | EquivariantValueNet | train_200k (200K) | None | 42, 43, 44 |
| `equiv_1m` | EquivariantValueNet | train_1m (1M) | None | 42, 43, 44 |
| `mlp_50k` | MLPValueNet | train_50k (50K) | None | 42, 43, 44 |
| `mlp_200k` | MLPValueNet | train_200k (200K) | None | 42, 43, 44 |
| `mlp_1m` | MLPValueNet | train_1m (1M) | None | 42, 43, 44 |

Additionally, 3 augmentation runs:

| Config ID | Model | Train set | Augmentation | Seeds |
|-----------|-------|-----------|--------------|-------|
| `mlp_aug_200k` | MLPValueNet | train_200k (200K) | Spatial + color on-the-fly | 42, 43, 44 |

**Total: 21 training runs.**

---

## Training Loop

```python
def train(model, train_loader, val_loader, config):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-5)
    
    best_val_mse = float('inf')
    patience_counter = 0
    best_state = None
    log = []
    
    for epoch in range(200):
        t0 = time.time()
        
        # --- Train ---
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        for X_batch, y_batch in train_loader:
            if config.augment:
                X_batch = apply_augmentation_batch(X_batch, spatial_perms, rng)
            
            pred = model(X_batch).squeeze(-1)
            loss = F.mse_loss(pred, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item() * len(y_batch)
            train_n += len(y_batch)
        
        scheduler.step()
        train_mse = train_loss_sum / train_n
        
        # --- Validate ---
        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                pred = model(X_batch).squeeze(-1)
                loss = F.mse_loss(pred, y_batch)
                val_loss_sum += loss.item() * len(y_batch)
                val_n += len(y_batch)
        val_mse = val_loss_sum / val_n
        
        epoch_time = time.time() - t0
        lr = scheduler.get_last_lr()[0]
        
        log.append({
            'epoch': epoch,
            'train_mse': train_mse,
            'val_mse': val_mse,
            'lr': lr,
            'epoch_seconds': epoch_time,
        })
        
        # --- Early stopping ---
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= 15:
                print(f"Early stopping at epoch {epoch}")
                break
    
    model.load_state_dict(best_state)
    return model, log
```

---

## Logging Format

Per-run CSV file: `results/{config_id}_seed{seed}_log.csv`

```csv
epoch,train_mse,val_mse,lr,epoch_seconds
0,0.08234,0.07891,0.001000,12.3
1,0.06512,0.06234,0.000998,11.8
...
```

---

## Checkpointing

Best model (lowest val MSE) saved as: `results/{config_id}_seed{seed}_best.pt`

```python
torch.save({
    'model_state_dict': model.state_dict(),
    'config': config_dict,
    'best_val_mse': best_val_mse,
    'best_epoch': best_epoch,
    'n_params': sum(p.numel() for p in model.parameters()),
}, path)
```

---

## Evaluation Metrics

All evaluation runs on the **depth-stratified test set** (~6,000 samples) using the best checkpoint.

### Metric 1: Equivariance error

```python
def measure_equivariance_error(model, test_X, spatial_perms_24, n_rotations=24, n_color_perms=20):
    """
    For each test state x (use first 500):
      For each of the 24 spatial rotations g:
        err_spatial += |model(g·x) - model(x)|
      For n_color_perms random color permutations σ:
        err_color += |model(σ·x) - model(x)|
      For n_combined random (g, σ) pairs:
        err_combined += |model((g,σ)·x) - model(x)|
    
    Report:
      mean_spatial_err, max_spatial_err
      mean_color_err, max_color_err  
      mean_combined_err, max_combined_err
    
    Expected:
      EquivariantValueNet: all ~1e-7 (float32 precision)
      MLPValueNet:         spatial ~0.01–1.0, color ~0.01–1.0
      MLPValueNet (aug):   somewhere in between
    """
```

### Metric 2: Distance prediction accuracy

```python
def evaluate_distance_prediction(model, test_X, test_y_dist, test_depths):
    """
    Compute on the full test set:
    
    1. overall_mae:   mean |pred * 11 - true_dist|
    2. per_depth_mae: {d: mean |pred*11 - d| for states at depth d}, for d = 0..11
    3. pearson_r:     correlation(pred, true_dist)
    4. rounded_acc:   fraction where round(pred * 11) == true_dist
    
    Returns dict with all metrics.
    """
```

### Metric 3: Greedy solve rate

```python
def greedy_solve(model, initial_state, moves, max_steps=50):
    """
    Greedy search using the value network:
    
    1. Encode current state
    2. For each of 6 possible moves, encode the resulting successor state
    3. Run model on all 6 successors (batched)
    4. Pick the move with the lowest predicted distance
    5. Apply that move
    6. If solved, return success + move count
    7. If max_steps reached, return failure
    
    Returns: (solved: bool, n_steps: int, path: list[int])
    """

def evaluate_solve_rate(model, all_states, all_dists, by_depth, moves, n_tests=1000):
    """
    Test greedy solver on n_tests random states from each depth bucket:
      - depth 4:  easy scrambles
      - depth 7:  medium
      - depth 10: hard
      - random:   uniform over all states
    
    For each bucket report:
      - solve_rate: fraction solved within 50 steps
      - mean_length: average solution length (solved cases only)
      - mean_excess: average (solution_length - optimal_distance)
      - optimal_rate: fraction where solution_length == optimal_distance
    """
```

### Metric 4: Data efficiency

```python
def evaluate_data_efficiency(results_dir):
    """
    Load val MSE from training logs for all configs.
    
    For each model type (equiv, mlp), collect:
      {50k: [val_mse_seed0, seed1, seed2], 200k: [...], 1m: [...]}
    
    Compute mean ± std across seeds.
    
    Key question to answer:
      At what dataset size does MLP match EquivariantValueNet's 50K performance?
    """
```

### Metric 5: Parameter efficiency

```python
def compute_parameter_efficiency(results):
    """
    For each model:
      param_count = sum(p.numel() for p in model.parameters())
      test_mae = overall MAE from metric 2
      efficiency = test_mae / param_count  (lower is better)
    
    Also report:
      effective_params_equiv = param_count (equivariant model)
      equivalent_mlp_params = param count of MLP that achieves the same test MAE
      compression_ratio = equivalent_mlp_params / effective_params_equiv
    """
```

---

## Plots

Generate with matplotlib. Save to `results/` as PNG (300 dpi).

### 1. `learning_curves.png`

- X-axis: epoch
- Y-axis: MSE (log scale)
- Lines: train and val for each of the 3 models at 200K dataset size
- Colors: equiv = blue, mlp = gray, mlp_aug = orange
- Show mean ± shaded std over 3 seeds
- Title: "Learning curves (200K training samples)"

### 2. `data_efficiency.png`

- X-axis: training set size (log scale: 50K, 200K, 1M)
- Y-axis: best val MSE
- Lines: equiv (blue), mlp (gray)
- Error bars: ±1 std over 3 seeds
- Title: "Data efficiency: val MSE vs training set size"

### 3. `per_depth_mae.png`

- X-axis: depth (0–11)
- Y-axis: MAE (original scale, i.e., in units of moves)
- Grouped bars: equiv (blue), mlp (gray), mlp_aug (orange) — all at 200K
- Title: "Per-depth prediction error (200K training samples)"

### 4. `equivariance_error.png`

- X-axis: model (equiv, mlp, mlp_aug)
- Y-axis: mean equivariance error (log scale)
- Grouped bars: spatial error, color error, combined error
- Title: "Equivariance error by symmetry type"

### 5. `solve_rates.png`

- X-axis: scramble depth (4, 7, 10, random)
- Y-axis: solve rate (%)
- Grouped bars: equiv (blue), mlp (gray), mlp_aug (orange) — all at 200K
- Title: "Greedy solve rate by scramble depth"

### 6. `param_comparison.png`

- Simple bar chart comparing total parameter counts
- Two bars: EquivariantValueNet, MLPValueNet
- Annotate with exact counts
- Title: "Parameter count comparison"

---

## Summary Table (to print to stdout)

After all evaluation completes, print a table like:

```
=== RESULTS SUMMARY (200K training set, mean ± std over 3 seeds) ===

                      Equiv          MLP            MLP+Aug
Parameters:           16,432         790,529        790,529
Val MSE:              0.0012±0.0001  0.0089±0.0003  0.0034±0.0002
Test MAE (moves):     0.38±0.02      1.21±0.05      0.72±0.04
Rounded accuracy:     0.87±0.01      0.52±0.02      0.71±0.02
Pearson r:            0.992±0.001    0.941±0.003    0.978±0.002
Spatial equiv err:    1.2e-7         0.43           0.12
Color equiv err:      1.4e-7         0.38           0.09
Solve rate (d=4):     100%           98%            100%
Solve rate (d=7):     99%            82%            95%
Solve rate (d=10):    95%            61%            84%
Solve rate (random):  97%            68%            88%
```

(Numbers are illustrative — actual values will differ.)

---

## Execution Order

```
1. Load BFS cache (or compute if missing)
2. Generate datasets (or load if cached)
3. Compute symmetry orbits
4. Print: n_spatial_orbits, n_total_orbits, param counts for both models
5. For each config in [equiv_50k, equiv_200k, equiv_1m, mlp_50k, mlp_200k, mlp_1m, mlp_aug_200k]:
     For each seed in [42, 43, 44]:
       Train model, save log + checkpoint
6. For each saved checkpoint:
     Run all 5 evaluation metrics
7. Generate all 6 plots
8. Print summary table
```

---

## Notes for Implementation

- **DataLoader setup:** Use `torch.utils.data.TensorDataset` + `DataLoader` with `shuffle=True` for training, `shuffle=False` for val/test. Pin memory if using CUDA.

- **Augmentation in DataLoader:** For the `mlp_aug` config, apply augmentation in the training loop (after fetching the batch, before forward pass), not in the dataset. This ensures each epoch sees different augmented views.

- **Input reshaping:** EquivariantValueNet expects `(batch, 24, 6)`. MLPValueNet expects `(batch, 144)`. Handle this inside each model's `forward()` method, or use a wrapper. Store data as `(N, 24, 6)` on disk and reshape as needed.

- **Reproducibility:** At the start of each training run:
  ```python
  torch.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)
  torch.backends.cudnn.deterministic = True  # if using CUDA
  ```

- **Timing:** Print estimated total runtime at the start based on a single-epoch timing probe. Expected: ~30–60 min total for all 21 runs on CPU.
