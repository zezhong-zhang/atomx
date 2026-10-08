"""One-electron B-spline integrals for the Dirac equation.

DBSR_HF-compatible: supports mixed-order bases (ks1 != ks2) for
kinetic balance where P uses order ksp and Q uses order ksq.

The key function _assemble_integral follows DBSR_HF's ZINTYm convention:
it accepts (ks1, ks2) and produces an (mm x mm) matrix where mm = max(ns1, ns2).
"""

from dataclasses import dataclass, field
from types import MappingProxyType

import numba as nb
import numpy as np


@nb.njit(cache=True)
def _assemble_integral(nv, ks1, ks2, mm, bsp1, bsp2, weights):
    """Assemble integral matrix <B1_i | w(r) | B2_j>.

    DBSR_HF ZINTYm equivalent. Supports mixed-order bases.

    Parameters
    ----------
    nv : int
        Number of knot intervals.
    ks1, ks2 : int
        Orders of the two B-spline bases (can differ for KB).
    mm : int
        Output matrix dimension (= max(nv+ks1-1, nv+ks2-1) typically).
    bsp1 : ndarray, shape (nv, n_gauss, ks1)
    bsp2 : ndarray, shape (nv, n_gauss, ks2)
    weights : ndarray, shape (nv, n_gauss)

    Returns
    -------
    mat : ndarray, shape (mm, mm)
    """
    n_gauss = bsp1.shape[1]
    mat = np.zeros((mm, mm))

    for iv in range(nv):
        for i in range(ks1):
            gi = iv + i
            if gi >= mm:
                continue
            for j in range(ks2):
                gj = iv + j
                if gj >= mm:
                    continue
                s = 0.0
                for m in range(n_gauss):
                    s += weights[iv, m] * bsp1[iv, m, i] * bsp2[iv, m, j]
                mat[gi, gj] += s

    return mat


def overlap_matrix(nv, ks, ns, bsp, grw):
    """Overlap S_ij = <B_i | B_j>. Symmetric."""
    S = _assemble_integral(nv, ks, ks, ns, bsp, bsp, grw)
    return 0.5 * (S + S.T)


@dataclass(frozen=True)
class SampledNuclearPotential:
    """Immutable operator values on one exact B-spline quadrature grid.

    This is a numerical integration input, not an interpolated potential
    for continuum solving. Off-grid evaluation and mismatched grids fail.
    """

    radius: np.ndarray
    values: np.ndarray
    _points: object = field(init=False, repr=False)

    def __post_init__(self):
        radius = np.array(self.radius, dtype=float, copy=True)
        values = np.array(self.values, dtype=float, copy=True)
        if radius.shape != values.shape:
            raise ValueError("sampled nuclear potential and grid shapes differ")
        radius.setflags(write=False)
        values.setflags(write=False)
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "values", values)
        object.__setattr__(
            self,
            "_points",
            MappingProxyType(
                dict(zip(map(float, radius.ravel()), values.ravel(), strict=True))
            ),
        )

    def on_grid(self, radius):
        if not np.array_equal(self.radius, radius):
            raise ValueError("sampled nuclear potential quadrature grid mismatch")
        return self.values

    def __call__(self, radius):
        return self._points[float(radius)]


def nuclear_matrix(nv, ks, ns, bsp, gr, grw, Z, nuclear_potential_func=None):
    """Nuclear potential V_ij = <B_i | V_nuc(r) | B_j>."""
    if nuclear_potential_func is None:
        v_weights = grw * (-Z / gr)
    elif isinstance(nuclear_potential_func, SampledNuclearPotential):
        # Identical nodewise multiplication, with no repeated Python calls.
        v_weights = grw * nuclear_potential_func.on_grid(gr)
    else:
        v_weights = np.zeros_like(grw)
        for iv in range(nv):
            for m in range(grw.shape[1]):
                v_weights[iv, m] = grw[iv, m] * nuclear_potential_func(gr[iv, m])

    V = _assemble_integral(nv, ks, ks, ns, bsp, bsp, v_weights)
    return 0.5 * (V + V.T)


def coupling_matrix(nv, ksp, ksq, ns, bsp_p, bsp_q, grw):
    """P-Q coupling <P_i | Q_j> (or with 1/r weight)."""
    return _assemble_integral(nv, ksp, ksq, ns, bsp_p, bsp_q, grw)


def coupling_deriv_matrix(nv, ksp, ksq, ns, bsp_p, bsp_q_deriv, grw):
    """P-Q derivative coupling <P_i | d/dr Q_j>."""
    return _assemble_integral(nv, ksp, ksq, ns, bsp_p, bsp_q_deriv, grw)
