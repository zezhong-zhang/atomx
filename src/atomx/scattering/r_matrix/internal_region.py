"""Internal-region machinery for the single-channel R-matrix solver.

Single-channel radial Schrödinger problem on [0, a] in atomic units::

    -1/2 u''(r) + [V(r) + l(l+1)/(2 r^2)] u(r) = E u(r),  u(0) = 0

with R-matrix boundary value ``b = 0`` (Neumann at ``r = a``).
Adding the Bloch operator ``L_b = (1/2) δ(r-a) (d/dr - b/r)``
makes ``H + L_b`` Hermitian over square-integrable trial functions
that vanish at the origin and are otherwise unconstrained at ``r = a``
(Burke 2011, §3.2). For ``b = 0`` the surface contribution from
integrating ``-(1/2) u''`` by parts cancels exactly, leaving only the
symmetric stiffness integral ``(1/2) ∫ B_i'(r) B_j'(r) dr``.

A clamped B-spline basis is used for the radial expansion; the first
spline is dropped to enforce ``u(0) = 0``, the last spline carries the
surface amplitude at ``r = a``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import eigh


@dataclass
class InternalRegionSolution:
    """Eigenpairs of the Bloch-augmented internal-region Hamiltonian."""

    eigenvalues_au: np.ndarray
    surface_amplitudes: np.ndarray
    a_au: float
    l: int
    n_basis: int
    k_spline_order: int


def _make_clamped_knots(a: float, n_basis: int, k_order: int) -> np.ndarray:
    """Open/clamped knot vector on [0, a] producing ``n_basis`` B-splines."""
    n_interior = n_basis - k_order
    if n_interior < 0:
        raise ValueError(
            f"n_basis={n_basis} too small for spline order={k_order} "
            f"(need n_basis >= {k_order})"
        )
    if n_interior == 0:
        interior = np.empty(0)
    else:
        interior = np.linspace(0.0, a, n_interior + 2)[1:-1]
    return np.concatenate(
        [np.zeros(k_order), interior, np.full(k_order, a)]
    )


def _bspline_design(
    r: np.ndarray, knots: np.ndarray, degree: int
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the full ``n_basis`` clamped-spline basis and first derivative.

    Returns ``(B, dB)`` each of shape ``(len(r), n_basis)``.
    """
    n_basis = len(knots) - degree - 1
    coef = np.eye(n_basis)
    spline = BSpline(knots, coef, degree, extrapolate=False)
    B = spline(r)
    dB = spline.derivative(1)(r)
    # BSpline returns NaN for points exactly at the right end with
    # extrapolate=False; nudge those onto the support.
    nan_mask = np.isnan(B).any(axis=1)
    if nan_mask.any():
        r_fixed = np.where(np.isclose(r, knots[-1]), knots[-1] * (1 - 1e-15), r)
        B = spline(r_fixed)
        dB = spline.derivative(1)(r_fixed)
    return B, dB


def _gauss_legendre_per_cell(
    knots: np.ndarray, n_gauss: int
) -> tuple[np.ndarray, np.ndarray]:
    """Composite Gauss-Legendre quadrature over each non-empty knot cell."""
    breakpoints = np.unique(knots)
    nodes, weights = np.polynomial.legendre.leggauss(n_gauss)
    pts = []
    wts = []
    for left, right in zip(breakpoints[:-1], breakpoints[1:]):
        if right <= left:
            continue
        half = 0.5 * (right - left)
        mid = 0.5 * (right + left)
        pts.append(half * nodes + mid)
        wts.append(half * weights)
    return np.concatenate(pts), np.concatenate(wts)


def assemble_internal_region(
    V_func: Callable[[np.ndarray], np.ndarray],
    a_au: float,
    n_basis: int,
    l: int,
    k_spline_order: int = 6,
    n_gauss: int = 10,
) -> InternalRegionSolution:
    """Diagonalize the Bloch-augmented single-channel Hamiltonian on [0, a].

    Parameters
    ----------
    V_func :
        Local radial potential ``V(r)`` (Hartree, function of r in Bohr).
        For a hydrogenic target the static potential is ``-Z/r``; the test
        suite drives this with arbitrary short-range potentials.
    a_au :
        R-matrix boundary radius in Bohr.
    n_basis :
        Number of clamped B-splines (the first spline is dropped to enforce
        ``u(0) = 0``, so the effective basis has ``n_basis - 1`` columns).
    l :
        Orbital angular momentum quantum number.
    k_spline_order :
        Polynomial order of the B-splines (degree = order - 1).
    n_gauss :
        Gauss-Legendre points per knot cell.
    """
    knots = _make_clamped_knots(a_au, n_basis, k_spline_order)
    degree = k_spline_order - 1
    r_q, w_q = _gauss_legendre_per_cell(knots, n_gauss)
    B, dB = _bspline_design(r_q, knots, degree)

    # Drop the first B-spline to impose u(0) = 0.
    B = B[:, 1:]
    dB = dB[:, 1:]

    V_q = np.asarray(V_func(r_q), dtype=float)
    centrifugal = 0.5 * l * (l + 1) / (r_q * r_q)
    V_eff = V_q + centrifugal

    # Overlap, stiffness, and potential matrices.
    Bw = B * w_q[:, None]
    S = Bw.T @ B
    T = 0.5 * (dB * w_q[:, None]).T @ dB
    V = (Bw * V_eff[:, None]).T @ B

    H = T + V
    # Symmetrize to undo accumulated round-off before the generalized
    # eigensolve - eigh raises if H is even a few ULPs non-Hermitian.
    H = 0.5 * (H + H.T)
    S = 0.5 * (S + S.T)

    eigvals, eigvecs = eigh(H, S)

    B_at_a, _ = _bspline_design(np.array([a_au]), knots, degree)
    boundary_row = B_at_a[0, 1:]
    surface_amplitudes = eigvecs.T @ boundary_row

    return InternalRegionSolution(
        eigenvalues_au=eigvals,
        surface_amplitudes=surface_amplitudes,
        a_au=a_au,
        l=l,
        n_basis=n_basis,
        k_spline_order=k_spline_order,
    )
