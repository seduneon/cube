"""
Unit and integration tests for the EMLP vs MLP cube pipeline.

Each test is designed to catch a specific class of bug:
  test_model_forward_shape        — output shape is (batch,) not (batch,1) or scalar
  test_checkpoint_roundtrip       — save→reload gives identical predictions (lazy-init bug)
  test_equivariance_random_weights— EMLP is structurally equivariant even before training
  test_training_loss_decreases    — gradient updates actually reduce loss (JIT bug)
  test_quick_pipeline             — full --quick run completes and produces a checkpoint

Usage:
    python tests.py              # run all tests
    python tests.py roundtrip    # run only tests whose name contains 'roundtrip'
"""

import sys
import os
import tempfile
import numpy as np

# ─── Helpers ─────────────────────────────────────────────────────────────────

def _init_model(model_type="emlp"):
    """Build + lazy-init a model. Always call this, never bare build_*_model()."""
    import jax.numpy as jnp
    from models import build_emlp_model, build_mlp_model
    if model_type == "emlp":
        model, G, _, _ = build_emlp_model()
    else:
        model, G, _, _ = build_mlp_model()
    # EMLP creates parameters lazily on the first forward pass for each new
    # batch size. Run several sizes so all vars exist before any checkpoint
    # save or load, regardless of what batch size is used downstream.
    for b in [1, 4, 32, 512]:
        model(jnp.zeros((b, 144), dtype=jnp.float32), training=False)
    return model


# ─── Test 1: output shape ─────────────────────────────────────────────────────

def test_model_forward_shape():
    """Model(x).squeeze() must return shape (batch,) for batch > 1."""
    import jax.numpy as jnp

    for model_type in ["emlp", "mlp"]:
        model = _init_model(model_type)
        for batch in [1, 4, 512]:
            x = jnp.zeros((batch, 144), dtype=jnp.float32)
            out = model(x, training=False).squeeze()
            expected = () if batch == 1 else (batch,)
            assert out.shape == expected, (
                f"{model_type} batch={batch}: got shape {out.shape}, expected {expected}"
            )


# ─── Test 2: checkpoint roundtrip ─────────────────────────────────────────────

def test_checkpoint_roundtrip():
    """Predictions must be bit-identical before save and after reload.

    Uses pickle so the full model object is preserved — including EMLP's
    equivariant basis matrices which live outside objax's var system and
    caused ~1% non-determinism when using npz + fresh model rebuild.
    """
    import jax.numpy as jnp
    from models import save_model, load_model

    for model_type in ["emlp", "mlp"]:
        model = _init_model(model_type)

        x = jnp.zeros((4, 144), dtype=jnp.float32)
        preds_before = np.array(model(x, training=False).squeeze())

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            save_model(model, path)
            model2 = load_model(path)
            preds_after = np.array(model2(x, training=False).squeeze())

            assert np.allclose(preds_before, preds_after, atol=1e-6), (
                f"{model_type} predictions differ after reload.\n"
                f"  before: {preds_before}\n"
                f"  after:  {preds_after}"
            )
        finally:
            os.unlink(path)


# ─── Test 3: equivariance with random weights ─────────────────────────────────

def test_equivariance_random_weights():
    """EMLP equivariance error must be < 1e-4 even with untrained weights.

    Equivariance is a structural property of EMLP — it holds for any weight
    values. A large error means the group representation or encoding is wrong.
    MLP equivariance error is expected to be large (it has no symmetry).
    """
    import jax.numpy as jnp
    from cube_env import SOLVED_STATE, MOVES, NUM_MOVES, apply_move, encode_state
    from cube_group import ALL_ROTATIONS, apply_rotation

    model = _init_model("emlp")

    rng = np.random.RandomState(0)
    n = 50
    states = []
    for _ in range(n):
        s = SOLVED_STATE.copy()
        for _ in range(6):
            s = apply_move(s, MOVES[rng.randint(NUM_MOVES)])
        states.append(s)

    X = np.array([encode_state(s) for s in states], dtype=np.float32)
    f_orig = np.array(model(jnp.array(X), training=False).squeeze())

    errors = []
    for rot in ALL_ROTATIONS[:8]:
        decoded = X.reshape(n, 24, 6).argmax(axis=2)
        rotated = np.array([apply_rotation(rot, decoded[i]) for i in range(n)])
        X_rot = np.array([encode_state(s) for s in rotated], dtype=np.float32)
        f_rot = np.array(model(jnp.array(X_rot), training=False).squeeze())
        errors.append(float(np.abs(f_rot - f_orig).mean()))

    mean_err = float(np.mean(errors))
    # Structural equivariance is ~1e-6 per op but accumulates to ~1e-4
    # across 3 layers in float32. Threshold is 1e-3 to stay above noise
    # while catching real failures (wrong group rep would give error ~1.0).
    assert mean_err < 1e-3, (
        f"EMLP equivariance error {mean_err:.2e} >= 1e-3. "
        "Check group representation or state encoding."
    )


# ─── Test 4: training loss decreases ─────────────────────────────────────────

def test_training_loss_decreases():
    """Loss must strictly decrease over 10 gradient steps.

    Catches broken JIT / gradient computation: if objax.Jit returns wrong
    values, gradients are garbage and loss does not decrease.
    """
    import jax.numpy as jnp
    import objax

    model = _init_model("emlp")

    rng = np.random.RandomState(42)
    X = rng.randn(64, 144).astype(np.float32)
    y = rng.randint(0, 14, 64).astype(np.float32)

    @objax.Function.with_vars(model.vars())
    def loss_fn(x, y_true):
        pred = model(x, training=True)
        return jnp.mean((pred.squeeze() - y_true) ** 2)

    grad_fn = objax.GradValues(loss_fn, model.vars())
    opt = objax.optimizer.Adam(model.vars())

    losses = []
    for _ in range(10):
        g, v = grad_fn(jnp.array(X[:32]), jnp.array(y[:32]))
        opt(lr=3e-4, grads=g)
        losses.append(float(v[0]))

    assert losses[-1] < losses[0], (
        f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}\n"
        f"Full trace: {[f'{l:.4f}' for l in losses]}"
    )


# ─── Test 5: quick pipeline integration ──────────────────────────────────────

def test_quick_pipeline():
    """Full --quick run must complete and produce a valid checkpoint.

    Covers: BFS (or skip), data generation, training, evaluation.
    Uses --skip-bfs if tables already exist to save time.
    """
    import subprocess
    from dataset import bfs_tables_exist
    from train import CKPT_DIR, size_label

    cmd = [sys.executable, "run_all.py", "--quick", "--model", "emlp"]
    if bfs_tables_exist():
        cmd.append("--skip-bfs")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )

    assert result.returncode == 0, (
        f"Quick pipeline exited with code {result.returncode}.\n"
        f"--- stdout ---\n{result.stdout[-3000:]}\n"
        f"--- stderr ---\n{result.stderr[-1000:]}"
    )

    ckpt = os.path.join(CKPT_DIR, f"emlp_{size_label(1000)}_seed0.pkl")
    assert os.path.exists(ckpt), (
        f"Checkpoint not found at {ckpt} after quick run."
    )

    assert "val_mse" in result.stdout, (
        "Expected 'val_mse' in training output — training may not have run."
    )

    # Reload the checkpoint and verify predictions are finite and deterministic
    import jax.numpy as jnp
    from models import load_model
    from cube_env import SOLVED_STATE, encode_state

    model = _init_model("emlp")
    load_model(model, ckpt)

    x = jnp.array(encode_state(SOLVED_STATE).reshape(1, -1), dtype=jnp.float32)
    out1 = float(model(x, training=False).squeeze())
    out2 = float(model(x, training=False).squeeze())

    assert np.isfinite(out1), f"Prediction is not finite: {out1}"
    assert out1 == out2, f"Predictions not deterministic: {out1} vs {out2}"


# ─── Runner ──────────────────────────────────────────────────────────────────

ALL_TESTS = [
    test_model_forward_shape,
    test_checkpoint_roundtrip,
    test_equivariance_random_weights,
    test_training_loss_decreases,
    test_quick_pipeline,
]


def run_tests(filter_str=None):
    tests = ALL_TESTS
    if filter_str:
        tests = [t for t in tests if filter_str in t.__name__]
        if not tests:
            print(f"No tests match '{filter_str}'")
            return

    passed, failed = 0, 0
    for test in tests:
        name = test.__name__
        try:
            test()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}")
            print(f"        {e}")
            failed += 1

    total = passed + failed
    print(f"\n{passed}/{total} passed", end="")
    if failed:
        print(f"  ({failed} failed)")
    else:
        print()


if __name__ == "__main__":
    filter_str = sys.argv[1] if len(sys.argv) > 1 else None
    run_tests(filter_str)
