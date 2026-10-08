"""Single home for the Y^k potential mathematics.

The Y^k potential is the building block for every Coulomb/exchange
quantity::

    Y^k(r) = r^{-(k+1)} ∫_0^r ρ(r') r'^k dr' + r^k ∫_r^∞ ρ(r') r'^{-(k+1)} dr'

Two consumer classes exist in the codebase:

* **Dense Gauss-node grids** (the B-spline Galerkin path): the radial
  axis is knot intervals × ``n_gauss`` Gauss-Legendre nodes per interval,
  and every node is also a quadrature node.  Partial (truncated) integrals
  within the straddled cell use the *interpolatory rule*: exact integrals
  of the degree ``n_gauss - 1`` Lagrange interpolant of the cell's node
  values, via the cached matrices from :func:`yk_partial_matrices`.
  Cross-cell contributions are exact full-cell Gauss sums.  This module
  owns that kernel (:func:`yk_flat` + :func:`compute_yk`).

* **Arbitrary grids** (``atomx.numerics.quadrature``: scattering/DW/EII
  kernels): trapezoid cumulatives with their own fused numba batch
  kernels.  These are NOT migrated here yet — their gates are pinned
  against FAC benchmarks, and swapping their quadrature rule belongs
  with the open DW/EII defect pass.  When that lands, this module is
  the canonical place for the shared partial-interval construction.

The interpolatory rule replaced a "half Gauss weight of the straddling
node" estimator on 2026-09-23.  That estimator was a first-order
stand-in for the partial integral, which for a degree-``2ks-2``
B-spline density sampled at ``n_gauss = ks`` nodes is not determined by
the data it uses; it left a systematic ``O((dr/r)^2)`` bias in every
Y^k node value (2.8e-5 relative on R^0(1s,1s), growing as
``(k+1)(3k+2)/2`` in ``k``).  Measured against closed forms and 40-digit
mpmath references the interpolatory rule recovers machine precision on
the same grid, nodes and weights.  See
``tests/test_slater_integrator.py`` and
``docs/test_audit/PHYSICS_yk_node_values.md``.

High-order partial matrices use the equivalent orthogonal Legendre
expansion instead of cancellation-prone monomial products. Real and
complex c-product densities share this kernel; no conjugation is applied.

Matrix invariants (tested at n_gauss = 5, 9, 17, 25, 33 in
``tests/test_slater_integrator.py``)::

    mo + mi == 1        elementwise to floating-point roundoff
    mo @ wi == 1 + xi   (constant-density exactness)
    mi @ wi == 1 - xi
"""

import numba as nb
import numpy as np

from atomx.numerics.bspline.basis import gauss_legendre_points

# Interpolatory partial-interval matrices, keyed by n_gauss.  mo[i, j] and
# mi[i, j] reproduce the truncated integrals of the degree-n_gauss-1
# interpolant of the node values on the straddled knot interval, normalised
# so they act directly on the weighted node values g_j = w_j f_j:
#     int_a^{r_i} f dr  =  sum_j g_j mo[i, j]      (same cell as r_i)
#     int_{r_i}^b f dr  =  sum_j g_j mi[i, j]
# They depend only on the Gauss nodes, so they are built once per n_gauss.
_YK_PARTIAL_MATRICES = {}


def yk_partial_matrices(n_gauss):
    """Return ``(mo, mi)`` for the interpolatory partial-interval rule."""
    mats = _YK_PARTIAL_MATRICES.get(n_gauss)
    if mats is None:
        xi, wi = gauss_legendre_points(n_gauss)
        from numpy.polynomial.legendre import legvander

        # Same interpolatory rule in an orthogonal basis. Gauss discrete
        # orthogonality gives L_j(x)/wi_j=sum_l (2l+1)/2 P_l(x_j) P_l(x).
        # Avoid monomial products, whose cancellation loses precision at
        # the high quadrature orders required by graded DKB grids.
        values = legvander(xi, n_gauss)
        partial = np.empty((n_gauss, n_gauss))
        partial[:, 0] = xi+1
        for ell in range(1, n_gauss):
            partial[:, ell] = (values[:, ell+1]-values[:, ell-1])/(2*ell+1)
        mo = (partial*((2*np.arange(n_gauss)+1)/2)) @ values[:, :n_gauss].T
        mats = (mo, 1-mo)
        _YK_PARTIAL_MATRICES[n_gauss] = mats
    return mats


@nb.njit(cache=True)
def yk_flat(rho_flat, r_flat, w_flat, n, k, nv, ng, mo, mi):
    """Compute Y^k potential on a flattened (1D) dense Gauss grid.

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
        Partial-interval matrices from :func:`yk_partial_matrices`.

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
    # cell (degree ng-1 interpolant of the node values of rho*r^k).  See the
    # module docstring for the history of the replaced half-weight rule.
    i_out = np.empty_like(rho_flat)
    base = rho_flat[0] * 0.0
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
    i_in = np.empty_like(rho_flat)
    base = rho_flat[0] * 0.0
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
    yk = np.empty_like(rho_flat)
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
    # Preserve complex c-product densities; promote integer/float32 inputs
    # to the integration precision rather than truncating partial sums.
    rho = np.asarray(rho, dtype=np.result_type(rho, np.float64))
    mo, mi = yk_partial_matrices(n_gauss)
    yk = yk_flat(
        rho.ravel(), gr.ravel(), grw.ravel(), nv * n_gauss, k, nv, n_gauss, mo, mi
    )
    return yk.reshape(nv, n_gauss)
