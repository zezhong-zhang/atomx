"""Exact Liouville coordinate closure from signed affine generators.

A coordinate is excluded only if no sequence of literal nonzero generator
entries reaches it from the initial density. There is no amplitude cutoff.
Include reference generators when certifying differences between evolutions.
"""

from collections import deque

import numpy as np
from scipy.sparse import csc_matrix, eye, kron


def dipole_commutator(dipole):
    d = csc_matrix(dipole)
    unit = eye(d.shape[0], format="csc", dtype=complex)
    return -1j * (kron(unit, d, format="csc") - kron(d.T, unit, format="csc"))


def reachable_coordinates(initial_vector, generators):
    initial = np.asarray(initial_vector, complex)
    if initial.ndim != 1 or not np.isfinite(initial).all():
        raise ValueError("finite vectorized initial density required")
    size = len(initial)
    graph = csc_matrix((size, size), dtype=bool)
    for generator in generators:
        matrix = csc_matrix(generator)
        if matrix.shape != graph.shape or not np.isfinite(matrix.data).all():
            raise ValueError("matching finite affine generators required")
        pattern = matrix.copy()
        pattern.eliminate_zeros()
        pattern.data = np.ones(len(pattern.data), dtype=bool)
        # Boolean union before arithmetic: independent terms can cancel only
        # at particular pulse amplitudes, which cannot define fixed closure.
        graph = graph + pattern.astype(bool)
    active = initial != 0
    queue = deque(np.flatnonzero(active))
    while queue:
        column = queue.popleft()
        neighbors = graph.indices[graph.indptr[column] : graph.indptr[column + 1]]
        unseen = neighbors[~active[neighbors]]
        active[unseen] = True
        queue.extend(unseen)
    return np.flatnonzero(active)
