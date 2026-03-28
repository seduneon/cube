"""
Equivariant and baseline MLP value networks for the 2x2x2 Pocket Cube.
Pure PyTorch — no external equivariance library required.

Both models map a 144-dim one-hot state encoding to a scalar distance prediction.
"""

import numpy as np
import torch
import torch.nn as nn

from equivariant_layers import GroupConvLayer, InvariantLinear

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMLP_AVAILABLE = True  # always True — no external library dependency

# Default hidden size for the equivariant model.
# c_hidden=16 → 24 positions × 16 channels = 384 effective features,
# matching MLPValueNet's hidden_dim=384 for a fair parameter comparison.
DEFAULT_CH = 16
DEFAULT_LAYERS = 3


# ─── Model definitions ────────────────────────────────────────────────────────

class EquivariantValueNet(nn.Module):
    """Group-convolutional value network equivariant to all 24 cube rotations.

    Architecture:
        (batch, 144) → reshape (batch, 24, 6)
        GroupConvLayer(6, c_hidden) + ReLU
        [GroupConvLayer(c_hidden, c_hidden) + ReLU] × (num_layers - 1)
        InvariantLinear(c_hidden) → (batch, 1)
    """

    def __init__(self, c_hidden: int = DEFAULT_CH, num_layers: int = DEFAULT_LAYERS):
        super().__init__()
        layers = [GroupConvLayer(6, c_hidden), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [GroupConvLayer(c_hidden, c_hidden), nn.ReLU()]
        self.conv_layers = nn.Sequential(*layers)
        self.output = InvariantLinear(c_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[0], 24, 6)   # (batch, 24, 6)
        x = self.conv_layers(x)           # (batch, 24, c_hidden)
        return self.output(x)             # (batch, 1)


class MLPValueNet(nn.Module):
    """Unconstrained MLP baseline.

    hidden_dim=384 matches EquivariantValueNet's 24×16=384 effective features.
    """

    def __init__(self, hidden_dim: int = 384, num_layers: int = DEFAULT_LAYERS):
        super().__init__()
        layers = [nn.Linear(144, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─── Factory functions ────────────────────────────────────────────────────────

def build_emlp_model(ch=DEFAULT_CH, num_layers=DEFAULT_LAYERS):
    """Build an equivariant (group-conv) value network."""
    model = EquivariantValueNet(c_hidden=ch, num_layers=num_layers)
    return model, None, None, None


def build_mlp_model(ch=384, num_layers=DEFAULT_LAYERS):
    """Build an unconstrained MLP baseline."""
    model = MLPValueNet(hidden_dim=ch, num_layers=num_layers)
    return model, None, None, None


def get_param_count(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def densify_emlp_model(model):
    """No-op: kept for API compatibility. Pure PyTorch models need no densification."""
    pass


# ─── Training utilities ───────────────────────────────────────────────────────

def cosine_decay_lr(step, total_steps, lr_init=3e-4, lr_min=1e-5):
    """Cosine annealing learning rate schedule."""
    progress = min(step / max(total_steps, 1), 1.0)
    return lr_min + 0.5 * (lr_init - lr_min) * (1 + np.cos(np.pi * progress))


def save_model(model, path):
    """Save model weights to .pt file."""
    path = str(path).replace('.npz', '.pt')
    torch.save(model.state_dict(), path)


def load_model(model_type, path):
    """Load model from checkpoint."""
    path = str(path).replace('.npz', '.pt')
    if model_type == 'emlp':
        model, _, _, _ = build_emlp_model()
    else:
        model, _, _, _ = build_mlp_model()
    state = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    return model


# ─── Main: print model info and verify equivariance ──────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from cube_env import SOLVED_STATE, apply_move, encode_state, MOVES
    from cube_group import ALL_ROTATIONS, apply_rotation

    print(f"Device: {DEVICE}")

    emlp_model, _, _, _ = build_emlp_model()
    mlp_model, _, _, _ = build_mlp_model()

    n_emlp = get_param_count(emlp_model)
    n_mlp = get_param_count(mlp_model)
    print(f"EquivariantValueNet: {n_emlp:,} parameters")
    print(f"MLPValueNet:         {n_mlp:,} parameters")
    print(f"MLP / EMLP ratio:    {n_mlp / n_emlp:.1f}x")

    from equivariant_layers import compute_pair_orbits
    _, n_orbits = compute_pair_orbits()
    print(f"Pair orbits:         {n_orbits}")

    # Verify output shapes
    x_test = torch.ones(4, 144)
    out_emlp = emlp_model(x_test)
    out_mlp = mlp_model(x_test)
    print(f"EquivariantValueNet output: {out_emlp.shape}")
    print(f"MLPValueNet output:         {out_mlp.shape}")

    # Verify equivariance
    emlp_model.eval()
    state = SOLVED_STATE.copy()
    for _ in range(5):
        state = apply_move(state, MOVES[0])

    x_orig = torch.tensor(encode_state(state), dtype=torch.float32).unsqueeze(0)
    max_err = 0.0
    with torch.no_grad():
        f_orig = emlp_model(x_orig).item()
        for rot in ALL_ROTATIONS:
            rotated = apply_rotation(rot, state)
            x_rot = torch.tensor(encode_state(rotated), dtype=torch.float32).unsqueeze(0)
            f_rot = emlp_model(x_rot).item()
            max_err = max(max_err, abs(f_orig - f_rot))
    print(f"Equivariance max error: {max_err:.2e}  (should be < 1e-4)")
