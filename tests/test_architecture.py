"""The dependency rule: the math, puzzle and search layers never import torch."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("module", ["eqcube.symmetry", "eqcube.puzzles", "eqcube.search"])
def test_layer_does_not_import_torch(module: str) -> None:
    code = f"import sys, {module}; sys.exit('torch' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
