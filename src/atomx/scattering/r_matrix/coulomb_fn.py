"""Coulomb wave functions ``F_l(eta, rho)`` and ``G_l(eta, rho)``.

The outer-region radial equation for an electron in the residual ``-1/r``
Coulomb field of a single-charge core (He+ residual after the projectile
moves out) is, with ``rho = k r`` and ``eta = -Z_eff / k``,

    F''_l + [ 1 - 2 eta / rho - l(l+1) / rho^2 ] F_l = 0.

Two linearly independent solutions are needed at the matching radius
``rho_a = k a`` to extract the K-matrix from the boundary R-matrix:
the regular ``F_l`` and irregular ``G_l``.

The neutral limit uses exact Riccati–Bessel functions. Charged references
start from a converged outgoing-wave asymptotic series and, when needed,
are propagated inward with adaptive DOP853 integration. In a forbidden
region the regular solution is integrated outward separately to avoid
loss of its small amplitude by cancellation. The Wronskian is checked
before a bounded cache stores the reference pair. Closed-channel radial
propagation uses a compiled Numerov recurrence.

"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from numba import njit
from scipy.integrate import solve_ivp
from scipy.special import gammaln, loggamma, spherical_jn, spherical_yn


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
    """Coulomb waves from the outgoing-wave series (DLMF 33.11.1).

    Substitution of H+=exp(i theta) sum_n a_n/rho^n into the radial
    equation gives a_(n+1)/a_n = [(n+i eta)(n+1+i eta)-l(l+1)]/[2i(n+1)].
    An asymptotic series must be truncated before its least term. Refuse
    if that term is too large; the caller moves the anchor out and
    propagates inward instead of accepting an inaccurate expansion.
    """
    sigma_l = coulomb_sigma_l(l, eta)
    theta = rho - eta * math.log(2.0 * rho) - 0.5 * l * math.pi + sigma_l
    dth = 1.0 - eta / rho

    term = total = complex(1.0)
    derivative = complex(0.0)
    for n in range(160):
        following = term * ((n + 1j * eta) * (n + 1 + 1j * eta) - l * (l + 1))
        following /= 2j * (n + 1) * rho
        if abs(following) < 2e-15 * abs(total):
            total += following
            derivative -= (n + 1) * following / rho
            phase = complex(math.cos(theta), math.sin(theta))
            value = phase * total
            slope = phase * (derivative + 1j * dth * total)
            return value.imag, slope.imag, value.real, slope.real
        if n > 0 and abs(following) > abs(term):
            break
        total += following
        derivative -= (n + 1) * following / rho
        term = following
    raise ArithmeticError("Coulomb asymptotic series needs a larger radius")


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
    return _numerov_values(f_vals, h, y0, dy0)


@njit(cache=True)
def _numerov_values(f_vals, h, y0, dy0):
    n = len(f_vals)
    h2 = h * h / 12.0
    y = np.empty(n)
    y[0] = y0
    fp = (-3 * f_vals[0] + 4 * f_vals[1] - f_vals[2]) / (2 * h)
    fpp = (f_vals[0] - 2 * f_vals[1] + f_vals[2]) / h**2
    y[1] = (y0 + h * dy0 + 0.5 * h**2 * f_vals[0] * y0
            + h**3 / 6 * (fp * y0 + f_vals[0] * dy0)
            + h**4 / 24 * ((fpp + f_vals[0]**2) * y0 + 2 * fp * dy0))
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


@lru_cache(maxsize=8192)
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
        Maximum rho step for adaptive inward propagation, if the
        asymptotic series cannot attain accuracy at the requested point.
        Zero-charge waves use exact spherical Bessel functions. Most
        asymptotic-radius calls require no ODE integration at all.
    """
    if rho_a <= 0:
        raise ValueError(f"rho_a must be positive, got {rho_a}")
    if l < 0:
        raise ValueError(f"l must be >= 0, got {l}")
    if h <= 0 or not np.isfinite([eta, rho_a, h]).all():
        raise ValueError("Coulomb parameters must be finite and h positive")
    if eta == 0.0:
        j, y = spherical_jn(l, rho_a), spherical_yn(l, rho_a)
        return (rho_a * j, j + rho_a * spherical_jn(l, rho_a, derivative=True),
                -rho_a * y, -y - rho_a * spherical_yn(l, rho_a, derivative=True))
    anchor = rho_a
    for _ in range(24):
        try:
            initial = _asymptotic_FG(l, eta, anchor)
            break
        except ArithmeticError:
            anchor = max(32.0, 2.0 * anchor)
    else:
        raise ArithmeticError("could not establish an accurate Coulomb asymptote")
    if anchor == rho_a:
        return initial

    def rhs(rho, v):
        f = -(1.0 - 2.0 * eta / rho - l * (l + 1) / rho**2)
        return np.array([v[1], f * v[0], v[3], f * v[2]])

    result = solve_ivp(rhs, (anchor, rho_a), initial, method="DOP853",
                       rtol=3e-12, atol=3e-14, max_step=h, t_eval=[rho_a])
    if not result.success:
        raise ArithmeticError(f"Coulomb inward integration failed: {result.message}")
    F, dF, G, dG = result.y[:, -1]
    # In a centrifugal/Coulomb forbidden region, contamination by the
    # irregular solution spoils the tiny F during inward propagation.
    # The regular power series has its own absolute normalization.
    if 1 - 2 * eta / rho_a - l * (l + 1) / rho_a**2 < 0:
        start = min(0.01, rho_a * 0.01, 0.01 / max(abs(eta), 1))
        regular = solve_ivp(lambda r, v: rhs(r, [*v, 0., 0.])[:2],
                            (start, rho_a), _series_F(l, eta, start),
                            method="DOP853", rtol=3e-12, atol=1e-30,
                            max_step=h, t_eval=[rho_a])
        if not regular.success:
            raise ArithmeticError("Coulomb regular integration failed")
        F, dF = regular.y[:, -1]
    wronskian = dF * G - F * dG
    if abs(wronskian - 1) > 1e-8:
        raise ArithmeticError(f"Coulomb Wronskian defect: {wronskian - 1:.3e}")
    return float(F), float(dF), float(G), float(dG)
