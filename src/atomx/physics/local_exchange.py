"""Local exchange functionals for self-consistent Dirac-Fock-Slater.

All functions take the radial charge density ρ(r) — electrons per unit
volume, NOT the radial density 4πr²ρ — and return V_x(r) on the same
grid in atomic units (Hartree). Callers (``DFSolver``,
``ShootingSolver``) are responsible for adding V_nuc + V_Hartree + V_x
and applying Latter's tail correction.

The three functionals implemented:

* :func:`slater_exchange` — plain Slater with user-chosen α; default
  α = 2/3 reproduces the original Hartree-Fock-Slater form.
* :func:`kohn_sham_exchange` — α = 1 (Kohn-Sham exchange).
* :func:`fac_gu_exchange` — Gu's empirically-tuned α_xc(Z, N) per
  ``faclib/radial.c:654 GetHXS``; the default FAC local-exchange form.

The FAC helper :func:`get_hxs` is exposed at module level so users can
inspect the α_xc that will be applied for a given ``(Z, N, ihx, hxs)``
without running a full SCF.
"""

from __future__ import annotations

from typing import Union

import numpy as np

from atomx.numerics.local_dfs.hxs_tables import (
    NHXN,
    SLATER_ALPHA_LDA,
    _HX0,
    _HX1,
)

ArrayLike = Union[np.ndarray, float]


# ---------------------------------------------------------------------------
# Primitive functionals
# ---------------------------------------------------------------------------


def slater_exchange(
    rho: ArrayLike, r: ArrayLike, alpha: float = 2.0 / 3.0
) -> np.ndarray:
    r"""Slater local-exchange potential.

    .. math::

        V_x(r) = -\alpha \left( \frac{3\rho(r)}{\pi} \right)^{1/3}

    Parameters
    ----------
    rho : array-like
        Charge density ρ(r) (electrons / Bohr³).
    r : array-like
        Radial grid (Bohr). Included for interface symmetry with
        grid-dependent functionals; ignored here.
    alpha : float, default 2/3
        Slater exchange coefficient.

    Returns
    -------
    V_x : ndarray
        Local exchange potential (Hartree) on the input grid.
    """
    _ = r  # intentionally unused; keeps a uniform call signature
    rho_arr = np.asarray(rho, dtype=float)
    out = np.zeros_like(rho_arr)
    mask = rho_arr > 1.0e-30
    out[mask] = -alpha * (3.0 * rho_arr[mask] / np.pi) ** (1.0 / 3.0)
    return out


def kohn_sham_exchange(rho: ArrayLike, r: ArrayLike) -> np.ndarray:
    """Kohn-Sham local-exchange potential (Slater with α = 1)."""
    return slater_exchange(rho, r, alpha=1.0)


# ---------------------------------------------------------------------------
# FAC/Gu empirical α_xc(Z, N)
# ---------------------------------------------------------------------------


def get_hxs(
    *,
    Z: float,
    N: float,
    ihx: float = -1.0,
    hxs: float = 1.0,
) -> float:
    """Gu's empirical α_xc(Z, N) — port of ``faclib/radial.c:654``.

    ``hxs`` > 0 and ≠ 1 selects an explicit user override, ``hxs`` <= −5
    selects the free-electron-gas LDA value ``(3/(4π²))^(1/3)`` , and
    the default ``hxs = 1`` uses the tabulated Gu fit.  ``ihx`` selects
    (or blends) the two branches of the tables:

    * ``|ihx| ≤ 0.001`` → k=0 branch only
    * ``|ihx| ≥ 0.999`` → k=1 branch only  (FAC default, ``ihx = -1``)
    * otherwise         → smooth blend between the two

    Parameters
    ----------
    Z : float
        Nuclear charge (used as Z² in the fit).
    N : float
        Electron count; rounded to the nearest integer for the table
        index ``n = round(N)``.  For ``n < 2`` the function returns
        0.5 (FAC's fallback).
    ihx : float, default -1.0
        FAC's ``_ihx`` parameter; default matches FAC's default.
    hxs : float, default 1.0
        FAC's ``_hxs`` parameter; default matches FAC's default.
    """
    # User-override branch (FAC: p->hx0 > 0)
    if hxs > 0 and hxs != 1.0:
        r0 = hxs + ihx * Z * Z
        return float(np.clip(r0, 0.1, 0.9))

    # LDA-sentinel branch (FAC: p->hx0 <= -5)
    if hxs <= -5.0:
        return SLATER_ALPHA_LDA

    n = int(round(N))
    if n < 2:
        return 0.5

    hs = abs(ihx)
    if hs <= 0.001:
        hs = 0.0
    if hs >= 0.999:
        hs = 1.0

    idx = min(n - 2, NHXN - 1)

    r0 = 0.5
    r1 = 0.5

    if hs < 0.999:
        r0 = float(_HX0[0, idx] + _HX1[0, idx] * Z * Z)
        r0 = float(np.clip(r0, 0.1, 0.9))
        if hs < 0.001:
            return r0

    if hs > 0.001:
        r1 = float(_HX0[1, idx] + _HX1[1, idx] * Z * Z)
        r1 = float(np.clip(r1, 0.1, 0.9))
        if hs > 0.999:
            return r1

    return r1 * hs + r0 * (1.0 - hs)


def fac_gu_exchange(
    rho: ArrayLike,
    r: ArrayLike,
    *,
    Z: float,
    N_total: float,
    ihx: float = -1.0,
    hxs: float = 1.0,
    slater_alpha: float = 2.0 / 3.0,
) -> np.ndarray:
    """FAC/Gu local-exchange potential with Z- and N-dependent α_xc.

    .. math::

        V_x(r) = -\\alpha_{\\rm xc}(Z, N) \\cdot
                 \\left( \\frac{3\\rho(r)}{\\pi} \\right)^{1/3}

    where :math:`\\alpha_{\\rm xc}` is the Gu empirical fit returned
    by :func:`get_hxs`. When ``hxs = 0`` the function degenerates to
    :func:`slater_exchange` with the user-supplied ``slater_alpha``;
    this lets callers toggle between the Gu fit and plain Slater
    without changing code paths.
    """
    if hxs == 0.0:
        return slater_exchange(rho, r, alpha=slater_alpha)
    alpha = get_hxs(Z=Z, N=N_total, ihx=ihx, hxs=hxs)
    return slater_exchange(rho, r, alpha=alpha)
