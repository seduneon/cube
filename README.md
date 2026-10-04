# eqcube

Equivariant value networks + search for the 2x2x2 Pocket Cube.

## Layout

| Path | Contents |
|---|---|
| `src/eqcube/` | Library (v2 rewrite) |
| `tests/` | Tests, mirroring `src/` |
| `legacy/` | Original v1 implementation, kept as a reference / test oracle |
| `docs/` | Design notes and old plans |
| `paper/` | Research Methods paper |

## Dependency rule

```
symmetry  <-  nn  <-  models
puzzles   <-  search
symmetry never imports torch
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,torch]"
pytest
```

Legacy code runs from its own folder: `cd legacy && python train.py ...`
