"""Independent full signed diagnostic tensor using shared native physics."""

import itertools
from functools import lru_cache

import numpy as np

from atomx.angular.csf_hamiltonian import two_body_angular
from atomx.numerics.bspline.convol import ConvolCache, compute_slater_rk_cell


def direct_coulomb(problem, orbitals, workspace, *, radial_representation="cell"):
    labels = problem.integrals.labels
    n = len(labels)
    if 8 * n**4 * 3 > 64 * 1024**2:
        raise MemoryError("complete direct diagnostic tensor exceeds cap")
    cache = ConvolCache(workspace)
    if radial_representation not in ("cell", "bound_features"):
        raise ValueError("declared complete radial representation required")

    @lru_cache(None)
    def radial(a, b, c, d, rank):
        if radial_representation == "bound_features":
            return problem.features.integral(
                [orbitals[key] for key in (a, b, c, d)], rank
            )
        return compute_slater_rk_cell(
            *orbitals[a],
            *orbitals[b],
            *orbitals[c],
            *orbitals[d],
            workspace,
            rank,
            cache,
        )

    direct = np.zeros((n,) * 4)
    for p, q, r, s in itertools.product(range(n), repeat=4):
        a, b, c, d = (labels[i] for i in (p, q, r, s))
        direct[p, q, r, s] = sum(
            value * radial(a[:2], c[:2], b[:2], d[:2], rank)
            for rank, value in two_body_angular(
                a[1], a[2], b[1], b[2], c[1], c[2], d[1], d[2]
            )
        )
    antisym = direct - direct.swapaxes(2, 3)
    error = max(
        abs(antisym[p, q, r, s] - problem.integrals.eri(p, q, r, s))
        for p, q, r, s in itertools.product(range(n), repeat=4)
    )
    if not np.all(np.isfinite(direct)) or error > 1e-11:
        raise AssertionError("complete signed Coulomb identity failed")
    return direct, float(error)
