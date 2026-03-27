"""
EMLP and MLP model definitions for the 2x2x2 Pocket Cube value network.

Both models map a 144-dim one-hot state encoding to a scalar distance prediction.

Input representation:
  The 144-dim input is 24 copies of a 6-dim one-hot vector (one per sticker).
  In EMLP notation: repin = 6 * V(G), where V is the 24-dim permutation rep.
  When a rotation g permutes sticker positions, the 6-dim blocks permute accordingly.

Output representation:
  Scalar (1-dim invariant). The distance to solved is invariant under whole-cube rotation.

Architecture:
  Both models use ch=384 hidden channels, num_layers=3.
  EMLP: equivariant layers (weight basis constrained by group).
  MLP:  unconstrained linear layers (same dims, no symmetry constraint).
"""

import numpy as np

try:
    import emlp.nn as emlp_nn
    from emlp.reps import V, Scalar
    from cube_group import CubeRotationGroup, EMLP_AVAILABLE

    EMLP_AVAILABLE = EMLP_AVAILABLE and (CubeRotationGroup is not None)
except ImportError:
    EMLP_AVAILABLE = False

# Fallback: pure JAX/numpy MLP for environments without emlp
try:
    import jax
    import jax.numpy as jnp
    import objax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


# ─── Model factory ────────────────────────────────────────────────────────────

DEFAULT_CH = 384
DEFAULT_LAYERS = 3


def build_emlp_model(ch=DEFAULT_CH, num_layers=DEFAULT_LAYERS):
    """Build an EMLP model for the value network.

    Input:  6 * V(G)  — 144-dim, equivariant under sticker permutations
    Output: Scalar(G) — 1-dim invariant scalar (distance)
    """
    if not EMLP_AVAILABLE:
        raise RuntimeError(
            "emlp not installed. Run: pip install emlp\n"
            "Or try: pip install jax==0.4.13 jaxlib==0.4.13 && pip install objax emlp"
        )

    G = CubeRotationGroup()
    repin = 6 * V(G)      # 144-dim input (24 stickers × 6 colors)
    repout = Scalar(G)     # 1-dim invariant output

    model = emlp_nn.EMLP(repin, repout, group=G, ch=ch, num_layers=num_layers)
    return model, G, repin, repout


def build_mlp_model(ch=DEFAULT_CH, num_layers=DEFAULT_LAYERS):
    """Build an unconstrained MLP baseline with the same architecture.

    Uses EMLP's built-in MLP class, which has identical dims/nonlinearities
    but no equivariance constraints. The group argument only sets I/O shapes.
    """
    if not EMLP_AVAILABLE:
        raise RuntimeError("emlp not installed.")

    G = CubeRotationGroup()
    repin = 6 * V(G)
    repout = Scalar(G)

    model = emlp_nn.MLP(repin, repout, group=G, ch=ch, num_layers=num_layers)
    return model, G, repin, repout


def count_params(model):
    """Count the total number of trainable parameters in an objax model."""
    return sum(p.value.size for p in model.vars().tensors())


def get_param_count(model):
    """Return parameter count (handles different objax API versions)."""
    try:
        return sum(p.value.size for p in model.vars().tensors())
    except AttributeError:
        try:
            return sum(np.prod(v.shape) for v in model.vars().values())
        except Exception:
            return -1


# ─── Training utilities ───────────────────────────────────────────────────────

def make_loss_fn(model):
    """Create an MSE loss function for objax."""
    import objax
    import jax.numpy as jnp

    @objax.Function.with_vars(model.vars())
    def loss_fn(x, y):
        pred = model(x, training=True)
        return jnp.mean((pred.squeeze() - y) ** 2)

    return loss_fn


def make_val_fn(model):
    """Create a validation (no-dropout) forward pass."""
    import objax
    import jax.numpy as jnp

    @objax.Function.with_vars(model.vars())
    def val_fn(x):
        return model(x, training=False).squeeze()

    return val_fn


def cosine_decay_lr(step, total_steps, lr_init=3e-4, lr_min=1e-5):
    """Cosine annealing learning rate schedule."""
    progress = min(step / max(total_steps, 1), 1.0)
    return lr_min + 0.5 * (lr_init - lr_min) * (1 + np.cos(np.pi * progress))


def save_model(model, path):
    """Save model parameters to a .npz file."""
    import objax
    np.savez(path, **{k: v.value for k, v in model.vars().items()})


def load_model(model, path):
    """Load model parameters from a .npz file."""
    data = np.load(path)
    for k, v in model.vars().items():
        if k in data:
            v.assign(data[k])
        else:
            print(f"Warning: key {k} not found in checkpoint")


# ─── Main: print model info ───────────────────────────────────────────────────

if __name__ == "__main__":
    if not EMLP_AVAILABLE:
        print("emlp not available. Install with: pip install emlp")
        exit(1)

    print("Building EMLP model...")
    emlp_model, G, repin, repout = build_emlp_model()
    n_emlp = get_param_count(emlp_model)
    print(f"EMLP model: {n_emlp:,} parameters")

    print("\nBuilding MLP model...")
    mlp_model, _, _, _ = build_mlp_model()
    n_mlp = get_param_count(mlp_model)
    print(f"MLP  model: {n_mlp:,} parameters")

    if n_mlp > 0 and n_emlp > 0:
        ratio = n_mlp / n_emlp
        print(f"\nMLP / EMLP ratio: {ratio:.1f}x (expected ~10-24x)")

    print(f"\nInput:  {repin}  (dim={repin.size()})")
    print(f"Output: {repout} (dim={repout.size()})")
    n_elem = getattr(G, "num_elements", lambda: "?")()
    print(f"Group:  {G} (~24 elements)")

    # Quick forward pass test
    import jax.numpy as jnp
    x_test = jnp.ones((4, 144), dtype=jnp.float32)
    out_emlp = emlp_model(x_test, training=False)
    out_mlp = mlp_model(x_test, training=False)
    print(f"\nEMLP forward pass output shape: {out_emlp.shape}")
    print(f"MLP  forward pass output shape: {out_mlp.shape}")
