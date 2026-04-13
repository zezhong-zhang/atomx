"""One-electron B-spline integrals for the Dirac equation.

Computes overlap, nuclear potential, and coupling integrals
using Gauss-Legendre quadrature over the B-spline basis.
Following DBSR_HF conventions (ZINTYM routine).
"""

import numpy as np
import numba as nb


@nb.njit(cache=True)
def _assemble_integral(nv, ks1, ks2, ns1, bsp1, bsp2, weights):
    """Assemble integral matrix <B1_i | w(r) | B2_j> using Gauss quadrature.

    Parameters
    ----------
    nv : int
        Number of knot intervals.
    ks1, ks2 : int
        Orders of the two B-spline bases.
    ns1 : int
        Number of B-splines in basis 1 (for output dimension).
    bsp1 : ndarray, shape (nv, n_gauss, ks1)
        B-spline values (or derivatives) for basis 1 at Gauss points.
    bsp2 : ndarray, shape (nv, n_gauss, ks2)
        B-spline values (or derivatives) for basis 2 at Gauss points.
    weights : ndarray, shape (nv, n_gauss)
        Quadrature weights times any radial function: w(r) * grw.

    Returns
    -------
    mat : ndarray, shape (ns1, ns2)
        Integral matrix. Only a band of width max(ks1, ks2) is nonzero.
    """
    n_gauss = bsp1.shape[1]
    # Both dimensions use ns1 (= ns_full) so output is always square.
    # The loop only fills the (nv+ks1-1) × (nv+ks2-1) submatrix.
    mat = np.zeros((ns1, ns1))

    for iv in range(nv):
        for i in range(ks1):
            gi = iv + i  # global index in basis 1
            for j in range(ks2):
                gj = iv + j  # global index in basis 2
                s = 0.0
                for m in range(n_gauss):
                    s += weights[iv, m] * bsp1[iv, m, i] * bsp2[iv, m, j]
                mat[gi, gj] += s

    return mat


def overlap_matrix(nv, ks, ns, bsp, grw):
    """Compute overlap matrix S_ij = <B_i | B_j>.

    Parameters
    ----------
    nv : int
        Number of intervals.
    ks : int
        B-spline order.
    ns : int
        Number of B-splines.
    bsp : ndarray, shape (nv, n_gauss, ks)
        B-spline values at Gauss points.
    grw : ndarray, shape (nv, n_gauss)
        Gauss weights.

    Returns
    -------
    S : ndarray, shape (ns, ns)
        Symmetric positive-definite overlap matrix.
    """
    S = _assemble_integral(nv, ks, ks, ns, bsp, bsp, grw)
    # Symmetrize (should be symmetric by construction, but clean up roundoff)
    S = 0.5 * (S + S.T)
    return S


def nuclear_matrix(nv, ks, ns, bsp, gr, grw, Z, nuclear_potential_func=None):
    """Compute nuclear potential matrix V_ij = <B_i | V_nuc(r) | B_j>.

    For a point nucleus: V_nuc(r) = -Z/r.

    Parameters
    ----------
    nv, ks, ns : int
        Grid parameters.
    bsp : ndarray, shape (nv, n_gauss, ks)
        B-spline values at Gauss points.
    gr : ndarray, shape (nv, n_gauss)
        Radial coordinates of Gauss points.
    grw : ndarray, shape (nv, n_gauss)
        Gauss weights.
    Z : int or float
        Nuclear charge.
    nuclear_potential_func : callable, optional
        V_nuc(r) function. Default: point-nucleus Coulomb -Z/r.

    Returns
    -------
    V : ndarray, shape (ns, ns)
    """
    if nuclear_potential_func is None:
        # Point nucleus: V(r) = -Z/r
        v_weights = grw * (-Z / gr)
    else:
        v_weights = np.zeros_like(grw)
        for iv in range(nv):
            for m in range(grw.shape[1]):
                v_weights[iv, m] = grw[iv, m] * nuclear_potential_func(gr[iv, m])

    V = _assemble_integral(nv, ks, ks, ns, bsp, bsp, v_weights)
    V = 0.5 * (V + V.T)
    return V


def coupling_matrix(nv, ksp, ksq, ns, bsp_p, bsp_q, grw):
    """Compute P-Q coupling overlap <P_i | Q_j>.

    Used for the kappa/r coupling term in the Dirac equation.

    Parameters
    ----------
    nv : int
        Number of intervals.
    ksp, ksq : int
        B-spline orders for P (large) and Q (small) components.
    ns : int
        Number of B-splines (same for both bases, but orders may differ).
    bsp_p : ndarray, shape (nv, n_gauss, ksp)
        P-component B-spline values.
    bsp_q : ndarray, shape (nv, n_gauss, ksq)
        Q-component B-spline values.
    grw : ndarray, shape (nv, n_gauss)
        Gauss weights.

    Returns
    -------
    C : ndarray, shape (ns, ns)
        Coupling matrix (NOT symmetric in general when ksp != ksq).
    """
    return _assemble_integral(nv, ksp, ksq, ns, bsp_p, bsp_q, grw)


def coupling_deriv_matrix(nv, ksp, ksq, ns, bsp_p, bsp_q_deriv, grw):
    """Compute P-Q derivative coupling <P_i | d/dr Q_j>.

    Used for the kinetic coupling term in the Dirac equation.

    Parameters
    ----------
    nv : int
        Number of intervals.
    ksp, ksq : int
        B-spline orders for P and Q.
    ns : int
        Number of B-splines.
    bsp_p : ndarray, shape (nv, n_gauss, ksp)
        P-component B-spline values.
    bsp_q_deriv : ndarray, shape (nv, n_gauss, ksq)
        Q-component B-spline DERIVATIVE values.
    grw : ndarray, shape (nv, n_gauss)
        Gauss weights.

    Returns
    -------
    D : ndarray, shape (ns, ns)
        Derivative coupling matrix.
    """
    return _assemble_integral(nv, ksp, ksq, ns, bsp_p, bsp_q_deriv, grw)
