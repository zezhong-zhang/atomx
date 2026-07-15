"""Arbitrary-grid Slater/Y^k quadrature — the canonical kernels.

This is the validated pairing that fixed the 2026-07 quadrature bug
(docs/scattering_fac_verification_2026-07.md): a composite radial grid
that resolves BOTH the inner bound structure and the continuum
oscillation, together with trapezoidal cumulative sums (second order on
non-uniform grids).  Rectangle-rule cumulatives on a structured grid are
O(h)-biased and produced a uniform ~15% error in Auger radial integrals
before the fix — new consumers must import from here rather than
re-deriving either half.

Numerical contract (all functions):
* grids may be arbitrary/non-uniform; spacings are taken from ``r``;
* Y^k and R^k are exact trapezoidal quadratures of
  Y^k(r) = Int (r_<^k / r_>^{k+1}) rho(r') dr' and
  R^k    = Int rho_ac(r) Y^k[rho_bd](r) dr;
* bound-density support truncation (threshold 1e-16 of the density max)
  drops ~1e-13 relative — far below quadrature accuracy;
* kernels are serial fused numba by design: they are memory-bound, and
  batched/vectorized variants measurably lose (three separate attempts
  documented in the verification campaign).

The B-spline Gauss quadrature (numerics/bspline) and the Salvat SLAG6
machinery (numerics/shooting) remain the right tools ON THEIR OWN grids;
this module is for integrals of interpolated waves on scattering-style
composite grids.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numba as nb
import numpy as np

C_SQ = 137.0359990836 ** 2


def p_rel(E_au):
    """Relativistic momentum p = sqrt(2E + (E/c)^2) in a.u."""
    return math.sqrt(2.0 * E_au + (E_au * E_au) / C_SQ)


@lru_cache(maxsize=64)
def _fine_grid_cached(r_max: float, k_max_au: float, z_inner: float):
    p2 = max(k_max_au, 1.0) ** 2
    pts = [max(1.0e-5 / max(z_inner, 1.0), 1.0e-7)]
    r = pts[0]
    while r < r_max:
        p_loc = math.sqrt(p2 + 2.0 * z_inner / r)
        dr = min(0.15 / p_loc, r / 48.0)
        r += dr
        pts.append(min(r, r_max))
        if len(pts) > 2_000_000:
            break
    arr = np.asarray(pts)
    arr.setflags(write=False)
    return arr


def make_fine_grid(r_max, k_max_au, z_inner=30.0):
    """Composite radial grid resolving inner bound structure AND the
    continuum oscillation.

    A uniform grid sized by the continuum wavelength under-resolves deep
    K/L-shell orbitals (Fe 1s: ~5 points across the peak at the old
    h = r_max/n).  The step obeys two local constraints:

        dr <= 0.15 / p_loc(r),  p_loc = sqrt(k_max^2 + 2 z_inner / r)
        dr <= r / 48            (>= 48 points per radial e-fold)

    Pass the actual nuclear Z as ``z_inner`` for exactness at high Z.
    The momentum is rounded up to 0.25 a.u. so nearby energies share a
    cached grid (and its power tables); the criterion has ~30% headroom.
    """
    k_key = float(np.ceil(max(float(k_max_au), 1.0) * 4.0) / 4.0)
    return _fine_grid_cached(float(r_max), k_key, float(z_inner))


# Power tables r^k / r^(k+1) / r^-(k+1) per (grid, k): recomputing them
# was a large share of every Slater integral.  Keyed by the grid's id()
# — fine grids are cached (read-only) and reused for thousands of
# integrals; the reciprocal table keeps the numba inner loops
# division-free.
_POW_CACHE: dict = {}


def pow_tables(r_grid, k):
    key = (id(r_grid), k)
    val = _POW_CACHE.get(key)
    if val is None:
        if len(_POW_CACHE) > 256:
            _POW_CACHE.clear()
        rk1 = r_grid ** (k + 1)
        val = (r_grid ** k, rk1, 1.0 / rk1)
        _POW_CACHE[key] = val
    return val


def yk_on_grid(rho, r, k, i_cut=None):
    """Hartree Y^k(r) = Int (r_<^k / r_>^{k+1}) rho(r') dr'.

    Trapezoidal cumulatives, second order on any grid.  ``i_cut``:
    return Y only on ``r[:i_cut+1]`` (the suffix integral still covers
    the full grid) — for consumers whose weighting density has compact
    support.
    """
    rk, rk1, irk1 = pow_tables(r, k)
    n = rho.shape[0]
    ic = n - 1 if i_cut is None else min(int(i_cut), n - 1)
    return _yk_kernel_rho(rho, r, rk, irk1, ic)


@nb.njit(cache=True, nogil=True)
def _yk_kernel_rho(rho, r, rk, irk1, i_cut):
    """Fused Y^k with trapezoidal cumulative sums (2nd order on any grid).

    ``irk1`` = 1/r^{k+1} (division-free inner loops).  The suffix
    (outer) integral spans the full grid; Y is materialized only on
    [0, i_cut].
    """
    n = rho.shape[0]
    y = np.empty(i_cut + 1)
    acc = 0.0
    prev_b = rho[n - 1] * irk1[n - 1]
    if i_cut == n - 1:
        y[n - 1] = 0.0
    for i in range(n - 2, -1, -1):
        b = rho[i] * irk1[i]
        acc += 0.5 * (b + prev_b) * (r[i + 1] - r[i])
        prev_b = b
        if i <= i_cut:
            y[i] = acc                  # suffix (outer) integrals
    inner = 0.0
    prev_a = rho[0] * rk[0]
    y[0] = inner * irk1[0] + rk[0] * y[0]
    for i in range(1, i_cut + 1):
        a = rho[i] * rk[i]
        inner += 0.5 * (a + prev_a) * (r[i] - r[i - 1])
        prev_a = a
        y[i] = inner * irk1[i] + rk[i] * y[i]
    return y


@nb.njit(cache=True, nogil=True)
def _slater_rk_kernel(P_a, Q_a, P_b, Q_b, P_c, Q_c, P_d, Q_d,
                      r, rk, irk1):
    """Fused R^k = Int rho_ac Y^k[rho_bd] dr with trapezoidal cumulatives.

    Second-order on arbitrary (non-uniform) grids; serial by design —
    memory-bound arrays, thread fork/join would dominate (same lesson as
    the SCF convolution kernels).  The forward (accumulation) pass runs
    only over the support of rho_ac: beyond it every contribution is
    exactly zero, and in the scattering integrals rho_ac always contains
    at least one bound orbital (support r <~ few a0, ~15% of the grid).
    """
    n = r.shape[0]
    # support of rho_ac: |rho| > 1e-16 * max|rho| (interpolated bound
    # tails are tiny-but-nonzero floats; the neglected contribution is
    # ~1e-13 relative, far below the quadrature accuracy)
    rho_max = 0.0
    for i in range(n):
        v = abs(P_a[i] * P_c[i] + Q_a[i] * Q_c[i])
        if v > rho_max:
            rho_max = v
    thr = 1.0e-16 * rho_max
    i_cut = 0
    for i in range(n - 1, -1, -1):
        if abs(P_a[i] * P_c[i] + Q_a[i] * Q_c[i]) > thr:
            i_cut = i
            break
    suffix = np.empty(i_cut + 1)
    acc = 0.0
    prev_b = (P_b[n - 1] * P_d[n - 1] + Q_b[n - 1] * Q_d[n - 1]) * irk1[n - 1]
    for i in range(n - 2, -1, -1):
        b = (P_b[i] * P_d[i] + Q_b[i] * Q_d[i]) * irk1[i]
        acc += 0.5 * (b + prev_b) * (r[i + 1] - r[i])
        prev_b = b
        if i <= i_cut:
            suffix[i] = acc
    total = 0.0
    inner = 0.0
    prev_a = (P_b[0] * P_d[0] + Q_b[0] * Q_d[0]) * rk[0]
    y0 = inner * irk1[0] + rk[0] * suffix[0]
    prev_g = (P_a[0] * P_c[0] + Q_a[0] * Q_c[0]) * y0
    for i in range(1, i_cut + 1):
        a = (P_b[i] * P_d[i] + Q_b[i] * Q_d[i]) * rk[i]
        inner += 0.5 * (a + prev_a) * (r[i] - r[i - 1])
        prev_a = a
        y = inner * irk1[i] + rk[i] * suffix[i]
        g = (P_a[i] * P_c[i] + Q_a[i] * Q_c[i]) * y
        total += 0.5 * (g + prev_g) * (r[i] - r[i - 1])
        prev_g = g
    return total


def grid_slater_rk(P_a, Q_a, P_b, Q_b, P_c, Q_c, P_d, Q_d, r_grid, k):
    """R^k(a,b,c,d) = Int rho_ac(r) Y^k[rho_bd](r) dr on an arbitrary grid.

    Argument order as ``compute_slater_rk_cell`` (rho13 = a*c,
    rho24 = b*d).  The fused numba kernel reproduces the reference
    ``yk_on_grid`` + dot implementation exactly up to floating-point
    summation order.
    """
    rk, rk1, irk1 = pow_tables(r_grid, k)
    return float(_slater_rk_kernel(P_a, Q_a, P_b, Q_b, P_c, Q_c, P_d, Q_d,
                                   r_grid, rk, irk1))


__all__ = [
    "make_fine_grid",
    "grid_slater_rk",
    "yk_on_grid",
    "pow_tables",
    "p_rel",
]
