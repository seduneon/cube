"""
EMLP and MLP model definitions for the 2x2x2 Pocket Cube value network.
PyTorch backend — works on Kaggle without JAX version conflicts.

Both models map a 144-dim one-hot state encoding to a scalar distance prediction.
"""

import numpy as np
import torch

try:
    from emlp.nn.pytorch import EMLP as EMLP_PT, MLP as MLP_PT
    from emlp.reps import V, Scalar
    from cube_group import CubeRotationGroup, EMLP_AVAILABLE
    EMLP_AVAILABLE = EMLP_AVAILABLE and (CubeRotationGroup is not None)
except ImportError:
    EMLP_AVAILABLE = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DEFAULT_CH = 384
DEFAULT_LAYERS = 3


# ─── Model factory ────────────────────────────────────────────────────────────

def build_emlp_model(ch=DEFAULT_CH, num_layers=DEFAULT_LAYERS):
    """Build an EMLP model for the value network."""
    if not EMLP_AVAILABLE:
        raise RuntimeError("emlp not installed. Run: pip install emlp")
    G = CubeRotationGroup()
    repin = 6 * V(G)
    repout = Scalar(G)
    model = EMLP_PT(repin, repout, group=G, ch=ch, num_layers=num_layers)
    return model, G, repin, repout


def build_mlp_model(ch=DEFAULT_CH, num_layers=DEFAULT_LAYERS):
    """Build an unconstrained MLP baseline with the same architecture."""
    if not EMLP_AVAILABLE:
        raise RuntimeError("emlp not installed.")
    G = CubeRotationGroup()
    repin = 6 * V(G)
    repout = Scalar(G)
    model = MLP_PT(repin, repout, group=G, ch=ch, num_layers=num_layers)
    return model, G, repin, repout


def get_param_count(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
    state = torch.load(path, map_location='cpu')
    model.load_state_dict(state)
    return model


# ─── Main: print model info ───────────────────────────────────────────────────

if __name__ == "__main__":
    if not EMLP_AVAILABLE:
        print("emlp not available. Install with: pip install emlp")
        exit(1)

    print(f"Device: {DEVICE}")

    print("Building EMLP model...")
    emlp_model, G, repin, repout = build_emlp_model()
    n_emlp = get_param_count(emlp_model)
    print(f"EMLP model: {n_emlp:,} parameters")

    print("\nBuilding MLP model...")
    mlp_model, _, _, _ = build_mlp_model()
    n_mlp = get_param_count(mlp_model)
    print(f"MLP  model: {n_mlp:,} parameters")

    if n_mlp > 0 and n_emlp > 0:
        print(f"\nMLP / EMLP ratio: {n_mlp / n_emlp:.1f}x")

    x_test = torch.ones(4, 144)
    out_emlp = emlp_model(x_test)
    out_mlp = mlp_model(x_test)
    print(f"\nEMLP forward pass output shape: {out_emlp.shape}")
    print(f"MLP  forward pass output shape: {out_mlp.shape}")
