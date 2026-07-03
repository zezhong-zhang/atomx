"""Slater integrals and two-electron Coulomb energies for LS-coupled terms.

The radial conventions match the rest of AtomX: ``P(r) = r R(r)`` and
``∫ P(r)^2 dr = 1``.  All energies are returned in Hartree.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def _gauss_legendre_grid(n_points: int, r_max: float) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights on ``[0, r_max]``."""
    x, w = np.polynomial.legendre.leggauss(n_points)
    # Map [-1, 1] -> [0, r_max]
    r = 0.5 * r_max * (x + 1.0)
    w = 0.5 * r_max * w
    return r, w


def _y_k(
    r: np.ndarray,
    w: np.ndarray,
    f: np.ndarray,
    k: int,
) -> np.ndarray:
    """Multipole function ``y^k(r_i) = ∫ f(r') r_<^k / r_>^{k+1} dr'``.

    Uses the cumulative (less/greater) decomposition evaluated with the
    supplied quadrature weights.
    """
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    r = np.asarray(r, dtype=float)
    w = np.asarray(w, dtype=float)
    f = np.asarray(f, dtype=float)
    n = r.shape[0]

    rk = r ** k
    inv_rk1 = 1.0 / (r ** (k + 1))

    f_rk = f * rk
    f_inv = f * inv_rk1

    # Contributions excluding the self-cell midpoint to match the trapezoidal
    # half-cell treatment used in cc_potentials.
    cum_lower = np.cumsum(w * f_rk)
    cum_upper_from_right = np.cumsum((w * f_inv)[::-1])[::-1]
    half_cell_lower = 0.5 * w * f_rk
    half_cell_upper = 0.5 * w * f_inv

    i_lt = cum_lower - half_cell_lower
    i_gt = cum_upper_from_right - half_cell_upper

    return inv_rk1 * i_lt + rk * i_gt


def slater_integral(
    P_a: Callable[[np.ndarray], np.ndarray],
    P_b: Callable[[np.ndarray], np.ndarray],
    P_c: Callable[[np.ndarray], np.ndarray],
    P_d: Callable[[np.ndarray], np.ndarray],
    k: int,
    *,
    n_points: int = 2000,
    r_max: float | None = None,
) -> float:
    """Compute the Slater integral ``R^k(ab, cd)``.

        R^k(ab, cd) = ∫∫ P_a(r1) P_b(r2)
                          (r_<^k / r_>^{k+1})
                          P_c(r1) P_d(r2) dr1 dr2

    Parameters
    ----------
    P_a, P_b, P_c, P_d
        Radial functions ``P(r) = r R(r)``.
    k
        Multipole order.
    n_points
        Number of Gauss-Legendre points in each dimension.
    r_max
        Outer integration limit.  If ``None``, an estimate based on the
        slowest-decaying hydrogenic orbital is used.

    Returns
    -------
    float
        Value in Hartree atomic units.
    """
    if r_max is None:
        # Hydrogenic 1s tail reaches ~1e-12 by r ~ 30/Z; use a conservative box.
        r_max = 50.0

    r, w = _gauss_legendre_grid(n_points, r_max)

    a_q = P_a(r)
    b_q = P_b(r)
    c_q = P_c(r)
    d_q = P_d(r)

    f_ab = a_q * b_q
    y_ab = _y_k(r, w, f_ab, k)
    integrand = y_ab * c_q * d_q
    return float(np.dot(w, integrand))


def term_energy_1s_nl(
    P_1s: Callable[[np.ndarray], np.ndarray],
    P_nl: Callable[[np.ndarray], np.ndarray],
    e_1s: float,
    e_nl: float,
    l: int,
    S: int,
    *,
    n_points: int = 2000,
    r_max: float | None = None,
) -> float:
    """Energy of a ``1s n l`` LS term.

    For inequivalent electrons the direct term has coefficient 1 and the
    exchange term has coefficient ``±1/(2l+1)`` (``+`` for singlet, ``-`` for
    triplet).  This follows from the Cowan/Burke LS-coupling reduction; the
    angular coefficients were cross-checked against
    ``angular_cc.direct_terms`` and ``angular_cc.exchange_terms``.
    """
    direct = slater_integral(
        P_1s, P_nl, P_1s, P_nl, k=0,
        n_points=n_points, r_max=r_max,
    )
    exchange = slater_integral(
        P_1s, P_nl, P_nl, P_1s, k=l,
        n_points=n_points, r_max=r_max,
    )
    exchange_sign = 1.0 if S == 0 else -1.0
    return e_1s + e_nl + direct + exchange_sign * exchange / (2 * l + 1)


def term_energy_1s2(
    P_1s: Callable[[np.ndarray], np.ndarray],
    e_1s: float,
    *,
    n_points: int = 2000,
    r_max: float | None = None,
) -> float:
    """Energy of the ``1s^2 ^1S`` closed-shell term."""
    coulomb = slater_integral(
        P_1s, P_1s, P_1s, P_1s, k=0,
        n_points=n_points, r_max=r_max,
    )
    return 2.0 * e_1s + coulomb
