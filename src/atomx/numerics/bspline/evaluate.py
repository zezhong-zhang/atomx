"""Evaluate B-spline orbital expansions and Y^k potentials on the radial grid.

This module provides the shared utilities for converting B-spline coefficient
vectors into function values on the quadrature grid.  Used by scf, mcdhf,
transitions, collisions, and corrections modules.

Key concepts
------------
**B-spline expansion**: An atomic orbital's radial component is expanded as

    P(r) = sum_i c_i B_i(r)

where c_i are coefficients (stored in ``orbitals_raw``) and B_i(r) are
B-spline basis functions (stored in ``bsp_data``).

**Y^k potential**: The multipole potential arising from an electron charge
density rho(r), defined as

    Y^k(r) = r^{-(k+1)} int_0^r rho(r') r'^k dr'
           + r^k int_r^inf rho(r') r'^{-(k+1)} dr'

This is the building block for Coulomb (k=0) and exchange (k>0) integrals.

**bsp_data dictionary**: The central data structure containing all pre-computed
B-spline grid and basis information.  Created by ``prepare_bspline_data()``
in ``bspline.basis``.  Fields:

    nv : int
        Number of knot intervals.
    ks : int
        B-spline order (degree + 1).
    ksp, ksq : int
        B-spline order for P (large) and Q (small) components.
        With Restricted Kinetic Balance (RKB): ksp = ks-1, ksq = ks.
        Without RKB: ksp = ksq = ks.
    nsp, nsq : int
        Number of B-splines for P and Q components.
    gr : ndarray, shape (nv, n_gauss)
        Gauss-Legendre quadrature points r in each knot interval.
    grw : ndarray, shape (nv, n_gauss)
        Gauss-Legendre quadrature weights in each interval.
    bsp_p : ndarray, shape (nv, n_gauss, ksp)
        P-component B-spline values at quadrature points.
    bsp_q : ndarray, shape (nv, n_gauss, ksq)
        Q-component B-spline values.
    bsp_p_d, bsp_q_d : ndarray
        First derivatives of B-splines (for kinetic/coupling integrals).
    t : ndarray, shape (ns + ks,)
        Knot sequence for the Q-component (full order).
    tp : ndarray
        Knot sequence for the P-component (may differ with RKB).
    ns : int
        max(nsp, nsq) for matrix allocation compatibility.

Performance: the Y^k kernel ``yk_flat`` in ``atomx.numerics.yk`` is
@nb.njit compiled; this module re-exports it under its historical name
``_compute_yk_flat``.  The exchange matrix kernel remains in
``scf/slater.py``.
"""

import numpy as np

from atomx.numerics.yk import compute_yk
from atomx.numerics.yk import yk_flat as _compute_yk_flat
from atomx.numerics.yk import yk_partial_matrices as _yk_partial_matrices

__all__ = [
    "_compute_yk_flat",
    "_yk_partial_matrices",
    "compute_yk",
    "eval_orbital_on_grid",
    "eval_P",
    "eval_Q",
]

# ---------------------------------------------------------------------------
# Y^k potential
# ---------------------------------------------------------------------------
# The Y^k mathematics (interpolatory partial-interval rule, numba kernel)
# lives in atomx.numerics.yk, the single home for the Y^k construction;
# it is re-exported here under its historical names so existing importers
# of this module keep working unchanged.


# ---------------------------------------------------------------------------
# Orbital evaluation on grid
# ---------------------------------------------------------------------------


def eval_orbital_on_grid(coeffs, bsp, nv, ks, ns):
    """Evaluate a B-spline expansion on the quadrature grid.

    Computes f(r) = sum_i c_i B_i(r) at all grid points using einsum.

    Parameters
    ----------
    coeffs : ndarray, shape (ns,)
        B-spline expansion coefficients.
    bsp : ndarray, shape (nv, n_gauss, ks)
        B-spline basis values at grid points (from bsp_data).
    nv : int
        Number of knot intervals.
    ks : int
        B-spline order (number of nonzero B-splines at any point).
    ns : int
        Total number of B-splines.

    Returns
    -------
    ndarray, shape (nv, n_gauss)
        Function values on the quadrature grid.
    """
    gi = np.arange(nv)[:, None] + np.arange(ks)[None, :]  # (nv, ks)
    gi_safe = np.clip(gi, 0, ns - 1)
    c = np.where(gi < ns, coeffs[gi_safe], 0.0)  # (nv, ks)
    return np.einsum("is,ims->im", c, bsp)


def eval_P(coeffs, bsp_data):
    """Evaluate the large (P) radial component on the grid.

    Kinetic-balance aware: uses ksp-order B-splines for the P component.
    """
    ksp = bsp_data.get("ksp", bsp_data["ks"])
    nsp = bsp_data.get("nsp", bsp_data["ns"])
    return eval_orbital_on_grid(coeffs, bsp_data["bsp_p"], bsp_data["nv"], ksp, nsp)


def eval_Q(coeffs, bsp_data):
    """Evaluate the small (Q) radial component on the grid.

    Kinetic-balance aware: uses ksq-order B-splines for the Q component.
    """
    ksq = bsp_data.get("ksq", bsp_data["ks"])
    nsq = bsp_data.get("nsq", bsp_data["ns"])
    bsp_q = bsp_data.get("bsp_q", bsp_data["bsp_p"])
    return eval_orbital_on_grid(coeffs, bsp_q, bsp_data["nv"], ksq, nsq)
