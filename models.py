"""
Value networks and model registry for the 2x2x2 Pocket Cube experiment.

Models
------
EquivariantValueNet       — group-conv equivariant to 24 spatial rotations
EquivariantColorValueNet  — group-conv equivariant to spatial rotations × S6 color perms
MLPValueNet               — unconstrained MLP baseline

Registry
--------
ModelSpec       — dataclass describing a model variant (class, kwargs, symmetries, …)
MODEL_REGISTRY  — dict[str, ModelSpec] with keys: "emlp", "emlp_col", "mlp", "mlp_aug"

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
from dataclasses import dataclass, field
from typing import Callable

from equivariant_layers import (
    GroupConvLayer,
    InvariantLinear,
    ColorEquivariantConvLayer,
    InvariantHead,
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ─── Model definitions ────────────────────────────────────────────────────────

class EquivariantValueNet(nn.Module):
    """Group-conv value network equivariant to the 24 spatial cube rotations.

    Architecture:
        (batch, 144) → reshape (batch, 24, 6)
        GroupConvLayer(6, c_hidden) + ReLU
        [GroupConvLayer(c_hidden, c_hidden) + ReLU] × (num_layers - 1)
        InvariantLinear(c_hidden) → (batch, 1)
    """

    def __init__(self, c_hidden: int = 84, num_layers: int = 3):
        super().__init__()
        layers = [GroupConvLayer(6, c_hidden), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [GroupConvLayer(c_hidden, c_hidden), nn.ReLU()]
        self.conv_layers = nn.Sequential(*layers)
        self.output = InvariantLinear(c_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[0], 24, 6)
        x = self.conv_layers(x)
        return self.output(x)


class EquivariantColorValueNet(nn.Module):
    """Group-conv value network equivariant to spatial rotations × S6 color perms.

    Hidden channels are organized in blocks of 6 (k_hidden blocks = 6*k_hidden
    total channels).  This lets S6 act cleanly on the hidden representation.

    Architecture:
        (batch, 144) → reshape (batch, 24, 6)          [= (batch, 24, 6*1)]
        ColorEquivariantConvLayer(1, k_hidden) + ReLU
        [ColorEquivariantConvLayer(k_hidden, k_hidden) + ReLU] × (num_layers - 1)
        InvariantHead(k_hidden) → (batch, 1)

    k_hidden=14 gives 84 effective channels, matching EquivariantValueNet(c_hidden=84).
    """

    def __init__(self, k_hidden: int = 14, num_layers: int = 3):
        super().__init__()
        layers = [ColorEquivariantConvLayer(1, k_hidden), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [ColorEquivariantConvLayer(k_hidden, k_hidden), nn.ReLU()]
        self.conv_layers = nn.Sequential(*layers)
        self.output = InvariantHead(k_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[0], 24, 6)   # (batch, 24, 6) = (batch, 24, 6*1)
        x = self.conv_layers(x)           # (batch, 24, 6*k_hidden)
        return self.output(x)             # (batch, 1)


class MLPValueNet(nn.Module):
    """Unconstrained MLP baseline.

    hidden_dim=384 matches EquivariantValueNet(c_hidden=84)'s 24×84 effective
    feature count for a fair architectural-size comparison.
    """

    def __init__(self, hidden_dim: int = 384, num_layers: int = 3):
        super().__init__()
        layers = [nn.Linear(144, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, 1))
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
    "emlp": ModelSpec(
        key="emlp",
        label="EMLP (spatial)",
        model_class=EquivariantValueNet,
        model_kwargs_fn=lambda w, nl: {"c_hidden": w, "num_layers": nl},
        symmetries=("spatial",),
    ),
    "emlp_col": ModelSpec(
        key="emlp_col",
        label="EMLP (spatial+color)",
        model_class=EquivariantColorValueNet,
        model_kwargs_fn=lambda w, nl: {"k_hidden": w, "num_layers": nl},
        symmetries=("spatial", "color"),
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

    if model_type == 'emlp':
        # conv_layers.0.weight: (c_hidden, 6, n_orbits)
        c_hidden = state['conv_layers.0.weight'].shape[0]
        model = EquivariantValueNet(c_hidden=c_hidden)

    elif model_type == 'emlp_col':
        # conv_layers.0.weight: (k_out, k_in=1, n_sp, n_co)
        k_hidden = state['conv_layers.0.weight'].shape[0]
        model = EquivariantColorValueNet(k_hidden=k_hidden)

    elif model_type in ('mlp', 'mlp_aug'):
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
    model = EquivariantValueNet(c_hidden=ch, num_layers=num_layers)
    return model, None, None, None


def build_emlp_color_model(k: int = 14, num_layers: int = 3):
    model = EquivariantColorValueNet(k_hidden=k, num_layers=num_layers)
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

    # Default widths matching CH_EMLP=84, K_EMLP_COL=14, CH_MLP=384
    widths = {"emlp": 84, "emlp_col": 14, "mlp": 384, "mlp_aug": 384}
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

    for key in ["emlp", "emlp_col"]:
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

    # Verify color equivariance for emlp_col only
    print("\nColor equivariance check (emlp_col only):")
    m = models_built["emlp_col"]
    m.eval()
    x_flat = encode_state(state).reshape(1, 144).astype(np.float32)
    with torch.no_grad():
        f0 = m(torch.tensor(x_flat)).item()
        max_err = 0.0
        rng = np.random.RandomState(0)
        for _ in range(50):
            perm = ALL_COLOR_PERMS[rng.randint(720)]
            x_perm = apply_color_perm_batch(x_flat, perm)
            fp = m(torch.tensor(x_perm)).item()
            max_err = max(max_err, abs(fp - f0))
    print(f"  emlp_col: color max_err = {max_err:.2e}  (should be < 1e-4)")
