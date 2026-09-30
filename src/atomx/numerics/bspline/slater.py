"""Slater radial integrals ``R^k`` on the B-spline quadrature grid.

Two interchangeable integrators, selected by the caller; the fast Poisson
method is the default.

    R^k(13, 24) = ∫∫ ρ₁₃(r₁) · [r_<^k / r_>^{k+1}] · ρ₂₄(r₂) dr₁ dr₂

with densities ρ₁₃ = P₁P₃ + Q₁Q₃ and ρ₂₄ = P₂P₄ + Q₂Q₄ supplied by the
caller (so the caller owns the large/small-component convention).

Methods
-------
``"poisson"`` (default) — **O(N)**.
    Builds the multipole potential ``Y^k[ρ₂₄](r)`` by the two-pass Poisson
    recursion (:func:`atomx.numerics.yk.compute_yk`) and
    contracts it once with ρ₁₃.  Resolves the ``r₁ = r₂`` kernel cusp
    analytically inside each knot interval, so it is both faster *and*
    more accurate than the direct product quadrature.

``"direct"`` — **O(N²)**.
    Independent double quadrature with the bare ``r_<^k / r_>^{k+1}``
    kernel. Coincident cells are split into triangles along r₁=r₂,
    resolving the cusp before quadrature. This is an accuracy reference;
    the fast Poisson recurrence remains the production default.

Design
------
Integrators live here in ``numerics`` and are chosen by name so callers
(Auger, DR, electron-impact, …) can trade speed for the textbook form
without duplicating kernels.  The default is always the fast method.
"""

from __future__ import annotations

from functools import lru_cache

import numba as nb
import numpy as np

from atomx.numerics.yk import compute_yk

#: Integrator used when the caller does not specify one.
DEFAULT_METHOD = "poisson"

#: Names accepted by :func:`slater_rk`.
AVAILABLE_METHODS = ("poisson", "direct")


@nb.njit(cache=True)
def _disjoint_cell_integral(rho13, rho24, r, w, k, n_gauss):
    """Product quadrature only where the multipole kernel is smooth."""
    total = 0.
    for i in range(len(r)):
        # Both triangular cell orderings, without a coincident-cell cusp.
        for j in range((i//n_gauss+1)*n_gauss, len(r)):
            kernel = (r[i]/r[j])**k/r[j]
            total += w[i]*w[j]*kernel*(rho13[i]*rho24[j]+rho13[j]*rho24[i])
    return total


@lru_cache(maxsize=16)
def _cell_triangle_rule(n_gauss, order):
    """Gauss nodes and density interpolation for a split square cell.

    On its lower triangle, r=a+h*x and s=a+h*x*y, x,y in (0,1).
    The Jacobian h²*x cancels the integrable Coulomb corner at a=0.
    This rule does not use the Poisson partial-interval matrices.
    """
    from numpy.polynomial.legendre import leggauss, legvander
    nodes, _ = leggauss(n_gauss)
    x, w = leggauss(order)
    x, w = (x+1)/2, w/2
    inverse = np.linalg.solve(legvander(nodes, n_gauss-1), np.eye(n_gauss))
    outer = legvander(2*x-1, n_gauss-1) @ inverse
    inner = legvander(2*x[:,None]*x[None,:]-1, n_gauss-1) @ inverse
    return nodes, x, w, outer, inner


def slater_rk_direct(rho13, rho24, gr, grw, k):
    """Independent double quadrature, split along each cell's r=s cusp.

    Grid and weights must be shaped (n_cells, n_gauss). Densities can be
    flat or have the grid shape. Disjoint cells use the ordinary product
    rule; coincident cells integrate both triangles with at least 2*n_gauss
    nodes per coordinate, increasing the order for high multipole ranks.
    Densities are evaluated from their cell interpolants.
    """
    gr, grw = np.asarray(gr), np.asarray(grw)
    if gr.ndim != 2 or gr.shape != grw.shape:
        raise ValueError('direct Slater integration requires a 2-D cell Gauss grid and weights')
    if k < 0 or int(k) != k:
        raise ValueError('Slater multipole rank must be a nonnegative integer')
    k = int(k)
    rho13 = np.asarray(rho13).reshape(gr.shape)
    rho24 = np.asarray(rho24).reshape(gr.shape)
    n_gauss = gr.shape[1]
    off_diagonal = _disjoint_cell_integral(rho13.ravel(), rho24.ravel(),
                                          gr.ravel(), grw.ravel(), k, n_gauss)
    # At the origin the inner-coordinate polynomial has degree at most
    # k+n_gauss-1. Retain its exact Gauss order even for unusually high k.
    order = max(2*n_gauss, (k+n_gauss+1)//2)
    nodes, x, w, outer, inner = _cell_triangle_rule(n_gauss, order)
    h = grw.sum(axis=1)
    a = gr[:,0]-h*(1+nodes[0])/2
    r = a[:,None,None]+h[:,None,None]*x[None,:,None]
    s = a[:,None,None]+h[:,None,None]*x[None,:,None]*x[None,None,:]
    kernel_jacobian = (s/r)**k/r*h[:,None,None]**2*x[None,:,None]
    f_outer, g_outer = rho13 @ outer.T, rho24 @ outer.T
    f_inner = np.einsum('cj,xyj->cxy', rho13, inner)
    g_inner = np.einsum('cj,xyj->cxy', rho24, inner)
    density = f_outer[:,:,None]*g_inner+g_outer[:,:,None]*f_inner
    diagonal = np.sum(density*kernel_jacobian*w[None,:,None]*w[None,None,:])
    return float(off_diagonal+diagonal)


def slater_rk_poisson(
    rho13: np.ndarray, rho24: np.ndarray,
    gr: np.ndarray, grw: np.ndarray, nv: int, n_gauss: int, k: int,
) -> float:
    """Fast O(N) Slater integral via the Poisson ``Y^k`` potential.

    ``rho13`` and ``rho24`` may be passed flat (``nv*n_gauss``) or already
    shaped ``(nv, n_gauss)``; ``gr``/``grw`` are the 2-D grid/weights.
    """
    rho24_2d = rho24.reshape(nv, n_gauss)
    rho13_2d = rho13.reshape(nv, n_gauss)
    yk = compute_yk(rho24_2d, gr, grw, nv, n_gauss, k)
    return float(np.sum(grw * rho13_2d * yk))


def slater_rk(
    rho13: np.ndarray, rho24: np.ndarray, bsp_data: dict, k: int,
    method: str = DEFAULT_METHOD,
) -> float:
    """Slater radial integral ``R^k(13, 24)`` with a selectable integrator.

    Parameters
    ----------
    rho13, rho24 : ndarray
        Pre-built densities ``P₁P₃ + Q₁Q₃`` and ``P₂P₄ + Q₂Q₄`` on the grid
        (flat ``nv*n_gauss`` or 2-D ``(nv, n_gauss)``).
    bsp_data : dict
        B-spline workspace (provides ``gr``, ``grw``, ``nv``).
    k : int
        Multipole rank.
    method : {"poisson", "direct"}, optional
        Integrator to use.  Defaults to the fast Poisson method
        (:data:`DEFAULT_METHOD`).

    Returns
    -------
    float
        ``R^k`` in atomic units.
    """
    gr = bsp_data["gr"]
    grw = bsp_data["grw"]
    nv = bsp_data["nv"]
    n_gauss = gr.shape[1]

    if method == "poisson":
        return slater_rk_poisson(rho13, rho24, gr, grw, nv, n_gauss, k)
    if method == "direct":
        return slater_rk_direct(
            rho13, rho24, gr, grw, k,
        )
    raise ValueError(
        f"unknown Slater integrator {method!r}; choose from {AVAILABLE_METHODS}"
    )
