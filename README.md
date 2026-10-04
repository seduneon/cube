# eqcube

Equivariant value networks + search for the 2x2x2 Pocket Cube.

## Layout

| Path | Contents |
|---|---|
| `src/eqcube/` | Library (v2 rewrite) |
| `tests/` | Tests, mirroring `src/` |
| `legacy/` | Original v1 implementation, kept as a reference / test oracle |
| `docs/` | Design notes (`ARCHITECTURE.md`) and old plans |
| `paper/` | Research Methods paper |

## Dependency rule

```
symmetry  <-  puzzles  <-  search          (NumPy only)
symmetry  <-  nn       <-  models          (torch)
```

`symmetry`, `puzzles`, `search` never import torch (tested). See
`docs/ARCHITECTURE.md` for the full design.

## Quick tour

```python
from eqcube.models import PRESETS, NetHeuristic, build_model
from eqcube.puzzles.pocket_cube import pocket_cube, task_group
from eqcube.search import BeamSearch
import numpy as np

task_group().order                      # 6: the computed task symmetry (S3)
model = build_model(PRESETS["task"])    # equivariant value net
cube = pocket_cube()
starts = cube.scramble(10, 6, np.random.default_rng(0))
results = BeamSearch(NetHeuristic(model), width=20).solve(cube, starts)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,torch]"
pytest
```

Legacy code runs from its own folder: `cd legacy && python train.py ...`
