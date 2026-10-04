"""Permutations and finite permutation groups.

Conventions
-----------
- ``g.arr[i]`` is the image of point ``i`` under ``g``.
- Composition: ``(g * h)(i) == g(h(i))`` -- apply ``h`` first, then ``g``.
- ``Perm`` is immutable and hashable.

Exercise: implement ``Perm`` and ``PermGroup`` according to the spec.
"""
