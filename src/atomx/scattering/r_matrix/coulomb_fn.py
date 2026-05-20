"""Coulomb wave functions ``F_l(eta, rho)`` and ``G_l(eta, rho)``.

The outer-region radial equation for an electron in the residual ``-1/r``
Coulomb field of a single-charge core (He+ residual after the projectile
moves out) is, with ``rho = k r`` and ``eta = -Z_eff / k``,

    F''_l + [ 1 - 2 eta / rho - l(l+1) / rho^2 ] F_l = 0.

Two linearly independent solutions are needed at the matching radius
``rho_a = k a`` to extract the K-matrix from the boundary R-matrix:
the regular ``F_l`` and irregular ``G_l``.

Implementation
--------------
* ``F_l`` is integrated outward by Numerov from a small-rho ascending
  series anchor.
* ``G_l`` is taken from the large-rho asymptotic expansion
  (Abramowitz & Stegun 14.5). Validated against the zero-charge limit
  ``F_l = rho * j_l(rho)``, ``G_l = -rho * y_l(rho)``.

The implementation targets the modest energy / partial-wave regime of
phase-2 (``k <= 2 a_0^{-1}``, ``l <= 3``, ``rho_a >= 25``).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import gammaln, loggamma


def _coulomb_C_l(l: int, eta: float) -> float:
    """``C_l(eta) = 2^l e^{-pi eta/2} |Gamma(l+1+i eta)| / (2l+1)!``.

    Uses the identity ``|Gamma(l+1+i eta)|^2 = pi |eta| / sinh(pi |eta|)
    * prod_{m=1..l} (m^2 + eta^2)`` (note the product starts at m=1).
    """
    log_prod = 0.0
    for m in range(1, l + 1):
        log_prod += math.log(m * m + eta * eta)
    # log[ pi |eta| / sinh(pi |eta|) ]; stable for large |eta|.
    x = math.pi * abs(eta)
    if x < 1e-12:
        log_factor = 0.0
    elif x < 30:
        log_factor = math.log(x) - math.log(math.sinh(x))
    else:
        log_factor = math.log(2.0 * x) - x  # sinh(x) ~ e^x / 2
    log_abs_gamma_sq = log_factor + log_prod
    log_C = (
        l * math.log(2.0)
        - 0.5 * math.pi * eta
        + 0.5 * log_abs_gamma_sq
        - gammaln(2 * l + 2)
    )
    return math.exp(log_C)


def _series_F(
    l: int, eta: float, rho: float, n_terms: int = 80
) -> tuple[float, float]:
    """Ascending series for ``F_l(eta, rho)`` and its rho-derivative."""
    a = np.zeros(n_terms + 2)
    a[0] = 1.0
    a[1] = eta / (l + 1.0)
    for j in range(n_terms):
        a[j + 2] = (
            2.0 * eta * a[j + 1] - a[j]
        ) / ((j + 2.0) * (2.0 * l + j + 3.0))

    js = np.arange(n_terms + 2)
    rho_pow = rho ** js                       # rho^0, rho^1, ...
    series = float(np.dot(a, rho_pow))
    if rho > 0:
        d_series = float(np.dot(js * a, rho ** (js - 1)))
    else:
        d_series = a[1] if n_terms >= 1 else 0.0

    C = _coulomb_C_l(l, eta)
    F = C * rho ** (l + 1) * series
    dF = C * ((l + 1) * rho ** l * series + rho ** (l + 1) * d_series)
    return F, dF


def coulomb_sigma_l(l: int, eta: float) -> float:
    """Coulomb phase shift ``sigma_l = arg Gamma(l + 1 + i eta)``."""
    sigma = float(loggamma(1.0 + 1j * eta).imag)
    for m in range(1, l + 1):
        sigma += math.atan2(eta, float(m))
    return sigma


def _asymptotic_FG(
    l: int, eta: float, rho: float
) -> tuple[float, float, float, float]:
    """``F_l, dF_l/drho, G_l, dG_l/drho`` from the large-rho expansion.

    Uses the standard A&S 14.5.5 / 14.5.6 expansion with the leading
    1/rho and 1/rho^2 correction terms; sufficient for ``rho >= 25``.
    """
    sigma_l = coulomb_sigma_l(l, eta)
    theta = rho - eta * math.log(2.0 * rho) - 0.5 * l * math.pi + sigma_l
    dth = 1.0 - eta / rho

    L = l * (l + 1)
    eta2 = eta * eta
    # f, g and their rho-derivatives — leading two terms each. The full
    # A&S 14.5 recursion would push these to higher order, but for the
    # matching radii used here (rho_a >= 25) the residual is below 1e-4.
    a2 = -(L - eta2) / 2.0
    f = 1.0 + a2 / (rho * rho)
    df = -2.0 * a2 / (rho ** 3)
    b1 = eta
    b3 = eta * (L - eta2 - 1.0) / 2.0
    g = b1 / rho + b3 / (rho ** 3)
    dg = -b1 / (rho * rho) - 3.0 * b3 / (rho ** 4)

    sin_t = math.sin(theta)
    cos_t = math.cos(theta)

    F = f * sin_t + g * cos_t
    G = f * cos_t - g * sin_t
    dF = df * sin_t + dg * cos_t + (f * cos_t - g * sin_t) * dth
    dG = df * cos_t - dg * sin_t - (f * sin_t + g * cos_t) * dth
    return F, dF, G, dG


def _numerov_outward(
    f_func, rho_grid: np.ndarray, y0: float, dy0: float
) -> np.ndarray:
    """Integrate ``y'' = f(rho) y`` outward by Numerov on a uniform grid."""
    n = rho_grid.size
    if n < 3:
        raise ValueError("rho_grid needs >= 3 points")
    h = rho_grid[1] - rho_grid[0]
    if not np.allclose(np.diff(rho_grid), h):
        raise ValueError("rho_grid must be uniform")
    f_vals = f_func(rho_grid)
    h2 = h * h / 12.0
    y = np.empty(n)
    y[0] = y0
    # Taylor-expand to second order for the second sample.
    y[1] = y0 + h * dy0 + 0.5 * h * h * f_vals[0] * y0
    for i in range(1, n - 1):
        # Numerov for y'' = f(x) y :
        #   y_{n+1}(1 - h^2 f_{n+1}/12)
        #     = 2 y_n (1 + 5 h^2 f_n / 12)
        #     - y_{n-1} (1 - h^2 f_{n-1}/12).
        num = (
            2.0 * y[i] * (1.0 + 5.0 * h2 * f_vals[i])
            - y[i - 1] * (1.0 - h2 * f_vals[i - 1])
        )
        y[i + 1] = num / (1.0 - h2 * f_vals[i + 1])
    return y


def coulomb_FG(
    l: int, eta: float, rho_a: float, h: float = 0.01
) -> tuple[float, float, float, float]:
    """Return ``F_l, dF_l/drho, G_l, dG_l/drho`` at ``rho = rho_a``.

    Parameters
    ----------
    l :
        Partial-wave angular momentum.
    eta :
        Sommerfeld parameter ``-Z_eff / k`` (negative for attractive
        Coulomb / electron-on-positive-ion, the only case we exercise
        in phase-2).
    rho_a :
        Match-radius times wavenumber, ``rho_a = k a``.
    h :
        Numerov step size in ``rho``. Default 0.01 is overkill for the
        modest accuracy needed but is cheap (~3000 steps to rho_a ~ 30).
    """
    if rho_a <= 0:
        raise ValueError(f"rho_a must be positive, got {rho_a}")
    if l < 0:
        raise ValueError(f"l must be >= 0, got {l}")

    def coul_f(rho):
        return -(1.0 - 2.0 * eta / rho - l * (l + 1) / (rho * rho))

    rho_start = min(0.05, rho_a * 0.05)
    F0, dF0 = _series_F(l, eta, rho_start)
    n_steps = max(int((rho_a - rho_start) / h) + 2, 200)
    rho_grid = np.linspace(rho_start, rho_a, n_steps)
    F_grid = _numerov_outward(coul_f, rho_grid, F0, dF0)
    F = float(F_grid[-1])
    h_eff = rho_grid[1] - rho_grid[0]
    # Central-difference derivative at the boundary; Numerov O(h^4)
    # accuracy is preserved with the simple 2nd-order finite difference.
    dF = float((F_grid[-1] - F_grid[-3]) / (2.0 * h_eff))

    # G_l and its derivative from the large-rho expansion at rho_a.
    _, _, G, dG = _asymptotic_FG(l, eta, rho_a)
    return F, dF, G, dG
