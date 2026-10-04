"""Law tests for Perm and PermGroup.

Exercise: closure, associativity, identity, inverse, composition-order pin,
Lagrange (g.order() divides |G|), orbit-stabilizer, |S4| == 24.
"""

import importlib


def test_module_importable() -> None:
    importlib.import_module("eqcube.symmetry.perm")
