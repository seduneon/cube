"""
Value networks and model registry for the 2x2x2 Pocket Cube experiment.

Models
------
RotValueNet   — group-conv equivariant to 24 spatial rotations          (emlp_rot)
BothValueNet  — group-conv equivariant to spatial rotations × S6        (emlp_both)
ColorValueNet — MLP on S6-invariant color-matching matrix               (emlp_col)
MLPValueNet   — unconstrained MLP baseline

Registry
--------
ModelSpec       — dataclass describing a model variant (class, kwargs, symmetries, …)
MODEL_REGISTRY  — dict[str, ModelSpec] with keys: "emlp_rot", "emlp_both", "emlp_col", "mlp", "mlp_aug"

Adding a new model
------------------
1. Define (or import) the nn.Module subclass.
2. Add a ModelSpec entry to MODEL_REGISTRY with an appropriate model_kwargs_fn.
3. Extend load_model() with a weight-shape inference branch.
That's it — train.py and evaluate.py consume MODEL_REGISTRY automatically.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Callable

from dataset import MAX_DISTANCE
NUM_CLASSES = MAX_DISTANCE + 1   # 15 distance classes: 0, 1, …, 14

from equivariant_layers import (
    GroupConvLayer,
    InvariantLinear,
    BothConvLayer,
    ColorConvLayer,
    InvariantHead,
    ColorOnlyInvariantHead,
    RotResBlock,
    BothResBlock,
    ColorResBlock,
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ─── Model definitions ────────────────────────────────────────────────────────

class RotValueNet(nn.Module):
    """Group-conv value network equivariant to the 24 spatial cube rotations.

    Architecture:
        (batch, 144) → reshape (batch, 24, 6)
        GroupConvLayer(6, c_hidden) + ReLU          [input projection]
        RotResBlock(c_hidden) × (num_layers - 1)    [pre-norm residual blocks]
        InvariantLinear(c_hidden) → (batch, NUM_CLASSES)   [class logits]
    """

    def __init__(self, c_hidden: int = 84, num_layers: int = 3):
        super().__init__()
        self.input_layer = GroupConvLayer(6, c_hidden)
        self.blocks = nn.ModuleList(
            [RotResBlock(c_hidden) for _ in range(num_layers - 1)]
        )
        self.output = InvariantLinear(c_hidden, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[0], 24, 6)
        x = F.relu(self.input_layer(x))
        for block in self.blocks:
            x = block(x)
        return self.output(x)


class BothValueNet(nn.Module):
    """Group-conv value network equivariant to spatial rotations × S6 color perms.

    Hidden channels are organized in blocks of 6 (k_hidden blocks = 6*k_hidden
    total channels).  This lets S6 act cleanly on the hidden representation.

    Architecture:
        (batch, 144) → reshape (batch, 24, 6)          [= (batch, 24, 6*1)]
        BothConvLayer(1, k_hidden) + ReLU               [input projection]
        BothResBlock(k_hidden) × (num_layers - 1)       [pre-norm residual blocks]
        InvariantHead(k_hidden) → (batch, 1)
    """

    def __init__(self, k_hidden: int = 14, num_layers: int = 3):
        super().__init__()
        self.input_layer = BothConvLayer(1, k_hidden)
        self.blocks = nn.ModuleList(
            [BothResBlock(k_hidden) for _ in range(num_layers - 1)]
        )
        self.output = InvariantHead(k_hidden, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[0], 24, 6)   # (batch, 24, 6*1)
        x = F.relu(self.input_layer(x))  # (batch, 24, 6*k_hidden)
        for block in self.blocks:
            x = block(x)
        return self.output(x)             # (batch, 1)


class ColorValueNet(nn.Module):
    """S6-invariant value network using a color-matching matrix representation.

    The color-matching matrix M[i,j] = x[i,:]·x[j,:] is 1 iff positions i and j
    share the same face color, and 0 otherwise.  This 24×24 binary matrix is the
    *complete* S6-invariant descriptor of the cube state: two states are related by
    a color permutation iff they have identical M.  An unconstrained MLP on the
    flattened (batch, 576) matrix can therefore learn any S6-invariant function.

    Contrast with the (broken) group-conv approach: ColorConvLayer processes each
    position independently, so color-averaging gives the same scalar regardless of
    which colors are where — making the output provably constant.

    Architecture:
        (batch, 144) → reshape (batch, 24, 6)
        M = einsum('bid,bjd->bij', x, x)  → (batch, 576)   [S6-invariant features]
        Linear(576, h_hidden) + ReLU      [input projection]
        (Linear(h_hidden, h_hidden) + ReLU) × (num_layers - 1)
        Linear(h_hidden, 1)

    h_hidden=60 gives ~38K params, matching BothValueNet(k_hidden=20).
    """

    def __init__(self, h_hidden: int = 60, num_layers: int = 3):
        super().__init__()
        layers = [nn.Linear(576, h_hidden), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(h_hidden, h_hidden), nn.ReLU()]
        layers.append(nn.Linear(h_hidden, NUM_CLASSES))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = x.view(B, 24, 6)
        M = torch.einsum('bid,bjd->bij', x, x)  # (B, 24, 24) — S6-invariant
        return self.net(M.reshape(B, 576))


class MLPValueNet(nn.Module):
    """Unconstrained MLP baseline.

    hidden_dim=384 matches RotValueNet(c_hidden=84)'s 24×84 effective
    feature count for a fair architectural-size comparison.
    """

    def __init__(self, hidden_dim: int = 384, num_layers: int = 3):
        super().__init__()
        layers = [nn.Linear(144, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, NUM_CLASSES))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(x.shape[0], -1))


# ─── Model registry ───────────────────────────────────────────────────────────

@dataclass
class ModelSpec:
    """Describes one model variant for training, evaluation, and plotting.

    Fields
    ------
    key           : short identifier used in filenames and CLI ("emlp", "mlp", …)
    label         : human-readable name for plots ("EMLP (spatial)", …)
    model_class   : the nn.Module subclass to instantiate
    model_kwargs_fn : (width: int, num_layers: int) -> dict passed to model_class()
    color_augment : if True, apply a random S6 color permutation to each training
                    batch (used for the augmentation baseline)
    symmetries    : tuple of symmetry names to test in equivariance evaluation;
                    supported values: "spatial", "color"
    """
    key: str
    label: str
    model_class: type
    model_kwargs_fn: Callable[[int, int], dict]
    color_augment: bool = False
    symmetries: tuple = field(default_factory=tuple)


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "emlp_rot": ModelSpec(
        key="emlp_rot",
        label="EMLP (rotation)",
        model_class=RotValueNet,
        model_kwargs_fn=lambda w, nl: {"c_hidden": w, "num_layers": nl},
        symmetries=("spatial",),
    ),
    "emlp_both": ModelSpec(
        key="emlp_both",
        label="EMLP (rotation+color)",
        model_class=BothValueNet,
        model_kwargs_fn=lambda w, nl: {"k_hidden": w, "num_layers": nl},
        symmetries=("spatial", "color"),
    ),
    "emlp_col": ModelSpec(
        key="emlp_col",
        label="MLP (color-matching)",
        model_class=ColorValueNet,
        model_kwargs_fn=lambda w, nl: {"h_hidden": w, "num_layers": nl},
        symmetries=("color",),
    ),
    "mlp": ModelSpec(
        key="mlp",
        label="MLP",
        model_class=MLPValueNet,
        model_kwargs_fn=lambda w, nl: {"hidden_dim": w, "num_layers": nl},
        symmetries=(),
    ),
    "mlp_aug": ModelSpec(
        key="mlp_aug",
        label="MLP + color aug",
        model_class=MLPValueNet,
        model_kwargs_fn=lambda w, nl: {"hidden_dim": w, "num_layers": nl},
        color_augment=True,
        symmetries=(),
    ),
    "mlp_matched": ModelSpec(
        key="mlp_matched",
        label="MLP (matched)",
        model_class=MLPValueNet,
        model_kwargs_fn=lambda w, nl: {"hidden_dim": w, "num_layers": nl},
        symmetries=(),
    ),
}


# ─── Factory / IO helpers ─────────────────────────────────────────────────────

def build_model(model_type: str, width: int, num_layers: int = 3) -> nn.Module:
    """Instantiate a model from the registry."""
    spec = MODEL_REGISTRY[model_type]
    return spec.model_class(**spec.model_kwargs_fn(width, num_layers))


def get_param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(model: nn.Module, path: str) -> None:
    """Save only model weights (legacy format)."""
    path = str(path).replace('.npz', '.pt')
    torch.save(model.state_dict(), path)


def save_checkpoint(model: nn.Module, path: str, **metadata) -> None:
    """Save model weights with metadata.

    Stored keys: model_state_dict, n_params, plus any extra keyword arguments
    (e.g. config dict, best_val_mse, best_epoch).
    """
    path = str(path).replace('.npz', '.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'n_params': get_param_count(model),
        **metadata,
    }, path)


def load_model(model_type: str, path: str) -> nn.Module:
    """Load a model checkpoint, inferring architecture from saved weight shapes.

    Handles both checkpoint formats:
    - New format: dict with 'model_state_dict' key (saved by save_checkpoint)
    - Old format: bare state_dict (saved by save_model)
    """
    path = str(path).replace('.npz', '.pt')
    data = torch.load(path, map_location='cpu', weights_only=True)

    # Unwrap new-format checkpoint
    if isinstance(data, dict) and 'model_state_dict' in data:
        state = data['model_state_dict']
    else:
        state = data

    def _count_layers(state):
        n_blocks = sum(1 for k in state if k.startswith('blocks.') and k.endswith('.conv.weight'))
        return n_blocks + 1   # blocks + input_layer

    if model_type == 'emlp_rot':
        # input_layer.weight: (c_hidden, 6, n_orbits)
        c_hidden = state['input_layer.weight'].shape[0]
        model = RotValueNet(c_hidden=c_hidden, num_layers=_count_layers(state))

    elif model_type == 'emlp_both':
        # input_layer.weight: (k_hidden, k_in=1, n_sp, n_co)
        k_hidden = state['input_layer.weight'].shape[0]
        model = BothValueNet(k_hidden=k_hidden, num_layers=_count_layers(state))

    elif model_type == 'emlp_col':
        # net.0.weight: (h_hidden, 576)
        h_hidden = state['net.0.weight'].shape[0]
        num_layers = sum(1 for k in state if k.startswith('net.') and k.endswith('.weight')) - 1
        model = ColorValueNet(h_hidden=h_hidden, num_layers=num_layers)

    elif model_type in ('mlp', 'mlp_aug', 'mlp_matched'):
        # net.0.weight: (hidden_dim, 144)
        hidden_dim = state['net.0.weight'].shape[0]
        model = MLPValueNet(hidden_dim=hidden_dim)

    else:
        raise ValueError(f"Unknown model_type: {model_type!r}. "
                         f"Known types: {list(MODEL_REGISTRY)}")

    model.load_state_dict(state)
    return model


# ─── Legacy factory functions (kept for backward compatibility) ───────────────

def build_emlp_model(ch: int = 84, num_layers: int = 3):
    model = RotValueNet(c_hidden=ch, num_layers=num_layers)
    return model, None, None, None


def build_emlp_color_model(k: int = 14, num_layers: int = 3):
    model = BothValueNet(k_hidden=k, num_layers=num_layers)
    return model, None, None, None


def build_mlp_model(ch: int = 384, num_layers: int = 3):
    model = MLPValueNet(hidden_dim=ch, num_layers=num_layers)
    return model, None, None, None


def densify_emlp_model(model):
    """No-op: kept for API compatibility."""
    pass


# ─── Training utilities ───────────────────────────────────────────────────────

def cosine_decay_lr(step, total_steps, lr_init=3e-4, lr_min=1e-5):
    progress = min(step / max(total_steps, 1), 1.0)
    return lr_min + 0.5 * (lr_init - lr_min) * (1 + np.cos(np.pi * progress))


# ─── Main: print registry info and verify equivariance ───────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from cube_env import SOLVED_STATE, apply_move, encode_state, MOVES
    from cube_symmetry import ALL_ROTATIONS, ALL_COLOR_PERMS, apply_color_perm_batch, apply_rotation

    print(f"Device: {DEVICE}\n")
    print(f"{'Model':<20} {'Params':>10}  {'Symmetries'}")
    print("-" * 50)

    # Default widths matching training defaults
    widths = {"emlp_rot": 84, "emlp_both": 14, "emlp_col": 60, "mlp": 384, "mlp_aug": 384, "mlp_matched": 384}
    models_built = {}
    for key, spec in MODEL_REGISTRY.items():
        m = build_model(key, widths[key])
        n = get_param_count(m)
        syms = ", ".join(spec.symmetries) if spec.symmetries else "none"
        aug = " [color aug]" if spec.color_augment else ""
        print(f"  {spec.label:<20} {n:>10,}  {syms}{aug}")
        models_built[key] = m

    # Verify spatial equivariance
    print("\nSpatial equivariance check:")
    state = SOLVED_STATE.copy()
    for _ in range(7):
        state = apply_move(state, MOVES[0])
    x_orig = torch.tensor(encode_state(state), dtype=torch.float32).unsqueeze(0)

    for key in ["emlp_rot", "emlp_both"]:
        m = models_built[key]
        m.eval()
        with torch.no_grad():
            f0 = m(x_orig).item()
            max_err = 0.0
            for rot in ALL_ROTATIONS:
                rotated = apply_rotation(rot, state)
                xr = torch.tensor(encode_state(rotated), dtype=torch.float32).unsqueeze(0)
                max_err = max(max_err, abs(m(xr).item() - f0))
        print(f"  {key}: spatial max_err = {max_err:.2e}  (should be < 1e-4)")

    # Verify color invariance/equivariance for emlp_both and emlp_col
    print("\nColor invariance check (emlp_both, emlp_col):")
    x_flat = encode_state(state).reshape(1, 144).astype(np.float32)
    rng = np.random.RandomState(0)
    for key in ["emlp_both", "emlp_col"]:
        m = models_built[key]
        m.eval()
        with torch.no_grad():
            f0 = m(torch.tensor(x_flat)).item()
            max_err = 0.0
            for _ in range(50):
                perm = ALL_COLOR_PERMS[rng.randint(720)]
                x_perm = apply_color_perm_batch(x_flat, perm)
                fp = m(torch.tensor(x_perm)).item()
                max_err = max(max_err, abs(fp - f0))
        print(f"  {key}: color max_err = {max_err:.2e}  (should be < 1e-4)")
