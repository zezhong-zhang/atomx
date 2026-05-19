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

Performance: ``_compute_yk_flat`` is @nb.njit compiled.
The exchange matrix kernel remains in ``scf/slater.py``.
"""

import numpy as np
from atomx._numba import nb


# ---------------------------------------------------------------------------
# Y^k potential: outward/inward cumulative integration (Numba)
# ---------------------------------------------------------------------------

@nb.njit(cache=True)
def _compute_yk_flat(rho_flat, r_flat, w_flat, n, k):
    """Compute Y^k potential on a flattened (1D) radial grid.

    Parameters
    ----------
    rho_flat : ndarray, shape (n,)
        Charge density rho(r) = P_a(r)*P_b(r) + Q_a(r)*Q_b(r) on grid.
    r_flat : ndarray, shape (n,)
        Radial grid points.
    w_flat : ndarray, shape (n,)
        Quadrature weights.
    n : int
        Total number of grid points (nv * n_gauss).
    k : int
        Multipole order (k=0 for Coulomb, k>0 for exchange).

    Returns
    -------
    ndarray, shape (n,)
        Y^k(r) on the grid.
    """
    rk = np.empty(n)
    rkm = np.empty(n)
    for i in range(n):
        rk[i] = r_flat[i] ** k
        rkm[i] = r_flat[i] ** (-(k + 1))

    # Outward integral: I_out(r) = int_0^r rho(r') r'^k dr'
    i_out = np.empty(n)
    s = 0.0
    for i in range(n):
        s += w_flat[i] * rho_flat[i] * rk[i]
        i_out[i] = s

    # Inward integral: I_in(r) = int_r^inf rho(r') r'^{-(k+1)} dr'
    i_in = np.empty(n)
    s = 0.0
    for i in range(n - 1, -1, -1):
        s += w_flat[i] * rho_flat[i] * rkm[i]
        i_in[i] = s

    # Y^k(r) = r^{-(k+1)} I_out + r^k I_in
    yk = np.empty(n)
    for i in range(n):
        yk[i] = rkm[i] * i_out[i] + rk[i] * i_in[i]
    return yk


def compute_yk(rho, gr, grw, nv, n_gauss, k):
    """Compute Y^k potential on the 2D quadrature grid.

    Parameters
    ----------
    rho : ndarray, shape (nv, n_gauss)
        Charge density on the grid.
    gr, grw : ndarray, shape (nv, n_gauss)
        Radial grid points and quadrature weights.
    nv : int
        Number of knot intervals.
    n_gauss : int
        Number of Gauss points per interval.
    k : int
        Multipole order.

    Returns
    -------
    ndarray, shape (nv, n_gauss)
        Y^k(r) on the grid.
    """
    yk_flat = _compute_yk_flat(rho.ravel(), gr.ravel(), grw.ravel(),
                                nv * n_gauss, k)
    return yk_flat.reshape(nv, n_gauss)


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
    return np.einsum('is,ims->im', c, bsp)


def eval_P(coeffs, bsp_data):
    """Evaluate the large (P) radial component on the grid.

    Kinetic-balance aware: uses ksp-order B-splines for the P component.
    """
    ksp = bsp_data.get("ksp", bsp_data["ks"])
    nsp = bsp_data.get("nsp", bsp_data["ns"])
    return eval_orbital_on_grid(coeffs, bsp_data["bsp_p"],
                                 bsp_data["nv"], ksp, nsp)


def eval_Q(coeffs, bsp_data):
    """Evaluate the small (Q) radial component on the grid.

    Kinetic-balance aware: uses ksq-order B-splines for the Q component.
    """
    ksq = bsp_data.get("ksq", bsp_data["ks"])
    nsq = bsp_data.get("nsq", bsp_data["ns"])
    bsp_q = bsp_data.get("bsp_q", bsp_data["bsp_p"])
    return eval_orbital_on_grid(coeffs, bsp_q, bsp_data["nv"], ksq, nsq)


