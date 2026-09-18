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
    recursion (:func:`atomx.numerics.bspline.evaluate.compute_yk`) and
    contracts it once with ρ₁₃.  Resolves the ``r₁ = r₂`` kernel cusp
    analytically inside each knot interval, so it is both faster *and*
    more accurate than the direct product quadrature.

``"direct"`` — **O(N²)**.
    Explicit double sum over the Gauss grid with the bare
    ``r_<^k / r_>^{k+1}`` kernel.  Textbook form, kept as an independent
    cross-check; its diagonal (``r₁ ≈ r₂``) is only as good as the product
    Gauss rule there.  On physical closed-shell densities the two agree to
    ``~1e-3`` (rank-3), the residual being this diagonal quadrature error.

Design
------
Integrators live here in ``numerics`` and are chosen by name so callers
(Auger, DR, electron-impact, …) can trade speed for the textbook form
without duplicating kernels.  The default is always the fast method.
"""

from __future__ import annotations

import numba as nb
import numpy as np

from atomx.numerics.bspline.evaluate import compute_yk

#: Integrator used when the caller does not specify one.
DEFAULT_METHOD = "poisson"

#: Names accepted by :func:`slater_rk`.
AVAILABLE_METHODS = ("poisson", "direct")


@nb.njit(cache=True, fastmath=True)
def slater_rk_direct(
    rho13: np.ndarray, rho24: np.ndarray,
    r: np.ndarray, w: np.ndarray, k: int,
) -> float:
    """Direct O(N²) Slater integral.

    Computes ``Σ_{i,j} w_i w_j ρ₁₃(r_i) ρ₂₄(r_j) · r_<^k / r_>^{k+1}`` over
    the flattened Gauss grid.  ``rho13``, ``rho24``, ``r`` and ``w`` are all
    1-D arrays of equal length.
    """
    n = r.shape[0]
    total = 0.0
    for i in range(n):
        ri = r[i]
        rho13_i = rho13[i]
        wi = w[i]
        for j in range(n):
            rj = r[j]
            rk_val = ri ** k / rj ** (k + 1) if ri <= rj else rj ** k / ri ** (k + 1)
            total += wi * w[j] * rho13_i * rho24[j] * rk_val
    return total


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
            np.ascontiguousarray(rho13).ravel(),
            np.ascontiguousarray(rho24).ravel(),
            gr.ravel(), grw.ravel(), k,
        )
    raise ValueError(
        f"unknown Slater integrator {method!r}; choose from {AVAILABLE_METHODS}"
    )
