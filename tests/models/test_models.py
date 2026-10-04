"""Whole models: invariance under their symmetry, config, checkpoints, heuristic."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from eqcube.models import (
    PRESETS,
    ModelConfig,
    NetHeuristic,
    build_model,
    load_checkpoint,
    save_checkpoint,
)
from eqcube.puzzles.pocket_cube import SYMMETRIES, pocket_cube
from eqcube.symmetry.action import FunctionAction

PC = pocket_cube()
STATES = PC.scramble(16, 9, np.random.default_rng(0))


def logits(model: torch.nn.Module, states: np.ndarray) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(states)).double()


@pytest.mark.parametrize("preset", ["rotation", "rotation_color", "color", "task"])
def test_preset_is_invariant_under_its_symmetry(preset: str) -> None:
    torch.manual_seed(0)
    cfg = PRESETS[preset]
    model = build_model(cfg)
    group = SYMMETRIES[cfg.symmetry]()
    action = FunctionAction(group, n_domain=24)
    base = logits(model, STATES)
    elements = sorted(group.elements, key=lambda g: g.arr)
    rng = np.random.default_rng(0)
    for i in rng.choice(len(elements), size=min(8, len(elements)), replace=False):
        g = elements[i]
        moved = np.array([action(g, tuple(int(c) for c in s)) for s in STATES], dtype=np.int8)
        torch.testing.assert_close(logits(model, moved), base, atol=1e-4, rtol=1e-4)


def test_mlp_is_not_task_invariant() -> None:
    torch.manual_seed(0)
    model = build_model(PRESETS["mlp_small"])
    task = SYMMETRIES["task"]()
    action = FunctionAction(task, n_domain=24)
    g = next(g for g in task.elements if not g.is_identity())
    moved = np.array([action(g, tuple(int(c) for c in s)) for s in STATES], dtype=np.int8)
    assert not torch.allclose(logits(model, moved), logits(model, STATES), atol=1e-4)


@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_every_preset_builds_and_outputs_logits(preset: str) -> None:
    out = logits(build_model(PRESETS[preset]), STATES)
    assert out.shape == (16, 15)


def test_positions_layout_rejected_when_colors_move() -> None:
    with pytest.raises(ValueError, match="permutes colors"):
        build_model(ModelConfig(symmetry="rotation_color", input_layout="positions"))


def test_unknown_arch_rejected() -> None:
    with pytest.raises(ValueError, match="unknown arch"):
        build_model(replace(ModelConfig(), arch="nope"))  # type: ignore[arg-type]


def test_config_roundtrip() -> None:
    cfg = PRESETS["task"]
    assert ModelConfig.from_dict(cfg.to_dict()) == cfg


@pytest.mark.parametrize("preset", ["task", "mlp_small"])
def test_checkpoint_roundtrip(tmp_path: Path, preset: str) -> None:
    torch.manual_seed(1)
    model = build_model(PRESETS[preset])
    path = tmp_path / "m.pt"
    save_checkpoint(path, model, PRESETS[preset], epoch=3, val_loss=0.5)
    loaded, cfg, meta = load_checkpoint(path)
    assert cfg == PRESETS[preset]
    assert meta == {"epoch": 3, "val_loss": 0.5}
    torch.testing.assert_close(logits(loaded, STATES), logits(model, STATES))


@pytest.mark.parametrize("reduce", ["argmax", "expected"])
def test_net_heuristic(reduce: str) -> None:
    torch.manual_seed(0)
    model = build_model(PRESETS["mlp_small"])
    h = NetHeuristic(model, reduce=reduce)  # type: ignore[arg-type]
    costs = h(STATES)
    assert costs.shape == (16,)
    assert np.all((costs >= 0) & (costs <= 14))
    small_batches = NetHeuristic(model, reduce=reduce, batch_size=3)  # type: ignore[arg-type]
    np.testing.assert_allclose(small_batches(STATES), costs, rtol=1e-6)
    if reduce == "argmax":
        assert np.all(costs == np.round(costs))


@pytest.mark.parametrize("preset", ["rotation", "rotation_color", "color", "task", "mlp_small"])
def test_matched_presets_have_similar_capacity(preset: str) -> None:
    n = sum(p.numel() for p in build_model(PRESETS[preset]).parameters())
    assert 30_000 <= n <= 45_000
