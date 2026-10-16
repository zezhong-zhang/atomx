"""Exact bound existing N3 test model and independent occupation-list oracle."""
import itertools

import numpy as np

from atomx.physics.csf.spinor_integrals import SpinorIntegrals


def integrals(phases=None, frozen=-2.3, spinors=6):
    rng = np.random.default_rng(194025)
    n, occupied = spinors, 3
    h = np.diag([-2.1, -1.6, -1.2, -.7, -.4, -.15, .1][:n])
    x = rng.normal(size=(n, n)) * .025
    h += x + x.T
    pairs = tuple(itertools.combinations(range(n), 2))
    lookup = {pair: i for i, pair in enumerate(pairs)}
    x = rng.normal(size=(len(pairs), len(pairs))) * .018
    pair_matrix = x + x.T
    if phases is not None:
        h = phases[:, None] * h * phases[None, :]
        pair_phases = np.array([phases[p] * phases[q] for p, q in pairs])
        pair_matrix = pair_phases[:, None] * pair_matrix * pair_phases[None, :]

    def eri(p, q, r, s):
        if p == q or r == s:
            return 0
        sign = (-1 if p > q else 1) * (-1 if r > s else 1)
        return sign * pair_matrix[lookup[tuple(sorted((p, q)))], lookup[tuple(sorted((r, s)))]]

    packed = np.array([eri(p, q, r, s)
                       for p in range(n - 1) for q in range(p + 1, n)
                       for r in range(p, n - 1) for s in range(r + 1, n)])
    fock = h + np.array([[sum(eri(p, i, q, i) for i in range(occupied))
                          for q in range(n)] for p in range(n)])
    energy = frozen + np.trace(h[:occupied, :occupied]) + .5 * sum(
        eri(i, j, i, j) for i in range(occupied) for j in range(occupied))
    obj = SpinorIntegrals(tuple((p + 1, -1, 1) for p in range(n)), occupied,
                         h, fock, packed, frozen, energy, {"scope": "finite two-body anchor"})
    return obj, h, eri

def independent_hamiltonian(obj, h, eri, electrons):
    """Occupation-list action with the full 1/4 tensor, not pair-reduced code."""
    n = len(obj.labels)
    masks = tuple(sum(1 << p for p in row)
                  for row in itertools.combinations(range(n), electrons))
    index = {mask: i for i, mask in enumerate(masks)}
    matrix = np.eye(len(masks)) * obj.frozen_energy_Ha
    terms = [(h[p, q], ((p, True), (q, False)))
             for p in range(n) for q in range(n)]
    terms += [(.25 * eri(p, q, r, s),
               ((p, True), (q, True), (s, False), (r, False)))
              for p, q, r, s in itertools.product(range(n), repeat=4)]
    for col, mask in enumerate(masks):
        for value, operations in terms:
            if value == 0:
                continue
            occupied = [p for p in range(n) if mask & (1 << p)]
            sign = 1
            for p, create in reversed(operations):
                if (p in occupied) == create:
                    break
                sign *= (-1)**sum(v < p for v in occupied)
                if create:
                    occupied.append(p)
                else:
                    occupied.remove(p)
            else:
                target = sum(1 << p for p in occupied)
                matrix[index[target], col] += value * sign
    assert np.max(abs(matrix - matrix.T)) < 1e-14
    return masks, .5 * (matrix + matrix.T)
