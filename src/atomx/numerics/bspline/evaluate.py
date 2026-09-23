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

import numba as nb
import numpy as np

from atomx.numerics.bspline.basis import gauss_legendre_points

# ---------------------------------------------------------------------------
# Y^k potential: outward/inward cumulative integration (Numba)
# ---------------------------------------------------------------------------

# Interpolatory partial-interval matrices, keyed by n_gauss.  mo[i, j] and
# mi[i, j] reproduce the truncated integrals of the degree-n_gauss-1
# interpolant of the node values on the straddled knot interval, normalised
# so they act directly on the weighted node values g_j = w_j f_j:
#     int_a^{r_i} f dr  =  sum_j g_j mo[i, j]      (same cell as r_i)
#     int_{r_i}^b f dr  =  sum_j g_j mi[i, j]
# They depend only on the Gauss nodes, so they are built once per n_gauss.
_YK_PARTIAL_MATRICES = {}


def _yk_partial_matrices(n_gauss):
    """Return ``(mo, mi)`` for the interpolatory partial-interval rule."""
    mats = _YK_PARTIAL_MATRICES.get(n_gauss)
    if mats is None:
        from numpy.polynomial import Polynomial

        xi, wi = gauss_legendre_points(n_gauss)
        theta = np.empty((n_gauss, n_gauss))
        psi = np.empty((n_gauss, n_gauss))
        for j in range(n_gauss):
            # Lagrange basis polynomial L_j on the Gauss nodes, followed by
            # its indefinite integral with constant zero.
            poly = Polynomial([1.0])
            for ell in range(n_gauss):
                if ell != j:
                    poly = poly * Polynomial([-xi[ell], 1.0]) / (xi[j] - xi[ell])
            anti = poly.integ()
            theta[:, j] = anti(xi) - anti(-1.0)
            psi[:, j] = anti(1.0) - anti(xi)
        # g_j = h wi_j f_j and the partial of the interpolant is
        # h sum_j f_j Theta[i, j], hence mo = Theta / wi column-wise.
        mats = (theta / wi[None, :], psi / wi[None, :])
        _YK_PARTIAL_MATRICES[n_gauss] = mats
    return mats


@nb.njit(cache=True)
def _compute_yk_flat(rho_flat, r_flat, w_flat, n, k, nv, ng, mo, mi):
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
        Total number of grid points (nv * ng).
    k : int
        Multipole order (k=0 for Coulomb, k>0 for exchange).
    nv, ng : int
        Number of knot intervals and Gauss points per interval.
    mo, mi : ndarray, shape (ng, ng)
        Partial-interval matrices from :func:`_yk_partial_matrices`.

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

    # Outward integral I_out(r) = int_0^r rho(r') r'^k dr', built cell by
    # cell: completed cells contribute their exact full-cell Gauss sums and
    # each node gets the interpolatory partial integral of its own straddled
    # cell (degree ng-1 interpolant of the node values of rho*r^k).  The
    # previous "half weight of the straddling node" estimator was a
    # first-order stand-in for that partial integral; the B-spline density
    # is a degree 2*ks-2 polynomial per interval, so the partial integral
    # is not determined by the node data the half-weight uses.  That left a
    # systematic O((dr/r)^2) bias in every Y^k node value (2.8e-5 relative
    # on R^0(1s,1s), growing as (k+1)(3k+2)/2 in k).  Measured against
    # closed forms and 40-digit mpmath references, the interpolatory rule
    # recovers machine precision on the same grid, nodes and weights
    # (2026-09-23); see tests/test_slater_integrator.py.
    i_out = np.empty(n)
    base = 0.0
    for m in range(nv):
        off = m * ng
        for i in range(ng):
            s = base
            for j in range(ng):
                s += w_flat[off + j] * rho_flat[off + j] * rk[off + j] * mo[i, j]
            i_out[off + i] = s
        for j in range(ng):
            base += w_flat[off + j] * rho_flat[off + j] * rk[off + j]

    # Inward integral I_in(r) = int_r^inf rho(r') r'^{-(k+1)} dr', same
    # construction with the inward matrices and reverse cell order.
    i_in = np.empty(n)
    base = 0.0
    for m in range(nv - 1, -1, -1):
        off = m * ng
        for i in range(ng):
            s = base
            for j in range(ng):
                s += w_flat[off + j] * rho_flat[off + j] * rkm[off + j] * mi[i, j]
            i_in[off + i] = s
        for j in range(ng):
            base += w_flat[off + j] * rho_flat[off + j] * rkm[off + j]

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
    mo, mi = _yk_partial_matrices(n_gauss)
    yk_flat = _compute_yk_flat(
        rho.ravel(), gr.ravel(), grw.ravel(), nv * n_gauss, k, nv, n_gauss, mo, mi
    )
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
