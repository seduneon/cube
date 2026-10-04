"""
Quick end-to-end tests for the pure PyTorch group-conv rewrite.

Run with: python test_models.py
"""

import os
import sys
import tempfile
import numpy as np
import torch

# ── ensure repo root is on path ───────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from equivariant_layers import GroupConvLayer, InvariantLinear, compute_pair_orbits
from models import (
    RotValueNet, MLPValueNet,
    build_emlp_model, build_mlp_model,
    get_param_count, save_model, load_model, cosine_decay_lr,
    DEVICE,
)
from cube_env import SOLVED_STATE, MOVES, apply_move, encode_state
from cube_group import ALL_ROTATIONS, apply_rotation
from dataset import load_dataset

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
results = []


def check(name, cond, detail=""):
    status = PASS if cond else FAIL
    msg = f"  [{status}] {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    results.append(cond)


# ─── 1. Pair orbits ───────────────────────────────────────────────────────────
print("\n1. Pair orbits")
pair_orbit, n_orbits = compute_pair_orbits()
check("all pairs assigned", (pair_orbit >= 0).all())
check("shape is (24,24)", pair_orbit.shape == (24, 24))
check("n_orbits > 1", n_orbits > 1, f"n_orbits={n_orbits}")
# diagonal pairs must all be in the same orbit (G acts transitively on positions)
diag_ids = {pair_orbit[i, i] for i in range(24)}
check("diagonal is single orbit", len(diag_ids) == 1, f"diagonal orbit ids={diag_ids}")


# ─── 2. GroupConvLayer shapes ────────────────────────────────────────────────
print("\n2. GroupConvLayer shapes")
layer = GroupConvLayer(6, 16)
x = torch.randn(8, 24, 6)
out = layer(x)
check("output shape (8,24,16)", out.shape == (8, 24, 16), str(out.shape))
check("pair_orbit is buffer", isinstance(layer.pair_orbit, torch.Tensor))


# ─── 3. InvariantLinear shapes ───────────────────────────────────────────────
print("\n3. InvariantLinear shapes")
inv = InvariantLinear(16)
x2 = torch.randn(8, 24, 16)
out2 = inv(x2)
check("output shape (8,1)", out2.shape == (8, 1), str(out2.shape))


# ─── 4. EquivariantValueNet equivariance ─────────────────────────────────────
print("\n4. Equivariance check")
emlp, _, _, _ = build_emlp_model(ch=16, num_layers=3)
emlp.eval()

# Scramble a state 7 moves deep
state = SOLVED_STATE.copy()
for i in range(7):
    state = apply_move(state, MOVES[i % len(MOVES)])

x_orig = torch.tensor(encode_state(state), dtype=torch.float32).unsqueeze(0)

with torch.no_grad():
    f_orig = emlp(x_orig).item()
    max_err = 0.0
    for rot in ALL_ROTATIONS:
        rotated = apply_rotation(rot, state)
        x_rot = torch.tensor(encode_state(rotated), dtype=torch.float32).unsqueeze(0)
        f_rot = emlp(x_rot).item()
        max_err = max(max_err, abs(f_orig - f_rot))

check("max equivariance error < 1e-4", max_err < 1e-4, f"max_err={max_err:.2e}")


# ─── 5. MLPValueNet is NOT equivariant (sanity check) ────────────────────────
print("\n5. MLP is NOT equivariant (sanity)")
mlp, _, _, _ = build_mlp_model(ch=384, num_layers=3)
mlp.eval()
with torch.no_grad():
    f_orig_mlp = mlp(x_orig).item()
    mlp_errs = []
    for rot in ALL_ROTATIONS[1:]:  # skip identity
        rotated = apply_rotation(rot, state)
        x_rot = torch.tensor(encode_state(rotated), dtype=torch.float32).unsqueeze(0)
        mlp_errs.append(abs(f_orig_mlp - mlp(x_rot).item()))
max_mlp_err = max(mlp_errs)
check("MLP equivariance error > 0 (expected)", max_mlp_err > 1e-4, f"max_err={max_mlp_err:.4f}")


# ─── 6. Parameter counts ─────────────────────────────────────────────────────
print("\n6. Parameter counts")
n_emlp = get_param_count(emlp)
n_mlp = get_param_count(mlp)
check("EMLP params < MLP params", n_emlp < n_mlp, f"emlp={n_emlp:,}  mlp={n_mlp:,}")
print(f"     MLP/EMLP ratio: {n_mlp/n_emlp:.1f}x")


# ─── 7. Cosine LR schedule ────────────────────────────────────────────────────
print("\n7. LR schedule")
lr0 = cosine_decay_lr(0, 1000, 3e-4, 1e-5)
lr_end = cosine_decay_lr(1000, 1000, 3e-4, 1e-5)
check("LR starts at lr_init", abs(lr0 - 3e-4) < 1e-8, f"{lr0:.2e}")
check("LR ends at lr_min", abs(lr_end - 1e-5) < 1e-8, f"{lr_end:.2e}")


# ─── 8. Save / load checkpoint ───────────────────────────────────────────────
print("\n8. Save / load checkpoint")
with tempfile.TemporaryDirectory() as tmpdir:
    path = os.path.join(tmpdir, "emlp_test.pt")
    save_model(emlp, path)
    check("checkpoint file created", os.path.exists(path))
    emlp2 = load_model('emlp_rot', path)
    # Compare outputs
    with torch.no_grad():
        x_check = torch.randn(4, 144)
        diff = (emlp(x_check) - emlp2(x_check)).abs().max().item()
    check("loaded model matches saved", diff < 1e-6, f"max diff={diff:.2e}")

    path_mlp = os.path.join(tmpdir, "mlp_test.pt")
    save_model(mlp, path_mlp)
    mlp2 = load_model('mlp', path_mlp)
    with torch.no_grad():
        diff_mlp = (mlp(x_check) - mlp2(x_check)).abs().max().item()
    check("loaded MLP matches saved", diff_mlp < 1e-6, f"max diff={diff_mlp:.2e}")


# ─── 9. Mini training loop (1k data, 3 epochs) ───────────────────────────────
print("\n9. Mini training loop (1k samples, 3 epochs)")
try:
    X_train, y_train = load_dataset("train", n_train=1000)
    X_val, y_val = load_dataset("val")
    y_train = y_train.astype(np.float32)
    y_val = y_val.astype(np.float32)

    mini_model, _, _, _ = build_emlp_model(ch=16, num_layers=2)
    mini_model = mini_model.to(DEVICE)
    optimizer = torch.optim.Adam(mini_model.parameters(), lr=1e-3)

    for epoch in range(3):
        mini_model.train()
        x_t = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
        y_t = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)
        optimizer.zero_grad()
        pred = mini_model(x_t).squeeze()
        loss = ((pred - y_t) ** 2).mean()
        loss.backward()
        optimizer.step()

    mini_model.eval()
    with torch.no_grad():
        x_v = torch.tensor(X_val[:500], dtype=torch.float32, device=DEVICE)
        y_v = torch.tensor(y_val[:500], dtype=torch.float32, device=DEVICE)
        val_loss = ((mini_model(x_v).squeeze() - y_v) ** 2).mean().item()

    check("training loop completes", True, f"val_mse={val_loss:.4f}")
    check("val MSE is finite", np.isfinite(val_loss))
except Exception as e:
    check("training loop completes", False, str(e))


# ─── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
n_pass = sum(results)
n_fail = len(results) - n_pass
print(f"Results: {n_pass}/{len(results)} passed", end="")
if n_fail:
    print(f"  ({n_fail} FAILED)")
    sys.exit(1)
else:
    print("  — all good!")
