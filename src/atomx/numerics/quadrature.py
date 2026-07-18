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


class GridResourceError(RuntimeError):
    """Raised when a requested scattering quadrature grid exceeds its cap."""


def p_rel(E_au):
    """Relativistic momentum p = sqrt(2E + (E/c)^2) in a.u."""
    return math.sqrt(2.0 * E_au + (E_au * E_au) / C_SQ)


@lru_cache(maxsize=64)
def _fine_grid_cached(
    r_max: float,
    k_max_au: float,
    z_inner: float,
    max_points: int,
):
    if not (math.isfinite(r_max) and r_max > 0.0):
        raise ValueError(f"make_fine_grid: r_max must be finite and > 0, got {r_max!r}")
    if not (math.isfinite(k_max_au) and k_max_au > 0.0):
        raise ValueError(
            f"make_fine_grid: k_max_au must be finite and > 0, got {k_max_au!r}"
        )
    if not (math.isfinite(z_inner) and z_inner >= 0.0):
        raise ValueError(
            f"make_fine_grid: z_inner must be finite and >= 0, got {z_inner!r}"
        )
    if int(max_points) < 2:
        raise ValueError("make_fine_grid: max_points must be >= 2")

    p2 = max(k_max_au, 1.0) ** 2
    pts = [max(1.0e-5 / max(z_inner, 1.0), 1.0e-7)]
    r = pts[0]
    while r < r_max:
        p_loc = math.sqrt(p2 + 2.0 * z_inner / r)
        dr = min(0.15 / p_loc, r / 48.0)
        r += dr
        pts.append(min(r, r_max))
        if len(pts) > max_points:
            remaining = max(r_max - r, 0.0)
            required = len(pts) + int(math.ceil(remaining / max(dr, 1.0e-300)))
            raise GridResourceError(
                "make_fine_grid exceeded radial grid cap: "
                f"required_points>={required}, cap={max_points}, "
                f"p={k_max_au:.12g} a.u., r_max={r_max:.12g} a0, "
                f"z_inner={z_inner:.12g}"
            )
    arr = np.asarray(pts)
    arr.setflags(write=False)
    return arr


def make_fine_grid(r_max, k_max_au, z_inner=30.0, *, max_points=2_000_000):
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
    return _fine_grid_cached(float(r_max), k_key, float(z_inner), int(max_points))


# Power tables r^k / r^(k+1) / r^-(k+1) per (grid, k): recomputing them
# was a large share of every Slater integral.  Keyed by the grid's id(),
# BUT the cache value holds a strong reference to the keyed array: a bare
# id() key goes stale when the grid is garbage-collected (the fine-grid
# lru evicts at 64 entries) and CPython reuses the address for a new
# array — observed as order-dependent 1e-4 omega drift and intermittent
# NaN before the fix.  Holding the reference makes id-reuse impossible
# while the entry lives; the reciprocal table keeps the numba inner
# loops division-free.
_POW_CACHE: dict = {}


def pow_tables(r_grid, k):
    key = (id(r_grid), k)
    val = _POW_CACHE.get(key)
    if val is None:
        if len(_POW_CACHE) > 256:
            _POW_CACHE.clear()
        rk1 = r_grid ** (k + 1)
        val = (r_grid, r_grid ** k, rk1, 1.0 / rk1)
        _POW_CACHE[key] = val
    return val[1:]


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


# Stacked r^k / 1/r^{k+1} tables for the batch kernel, one row per k.
# Rows are the pow_tables entries verbatim; keyed like _POW_CACHE.
_POW2D_CACHE: dict = {}


def _pow_tables_2d(r_grid, k_hi):
    # same strong-reference convention as _POW_CACHE (id-reuse safety)
    key = (id(r_grid), k_hi)
    val = _POW2D_CACHE.get(key)
    if val is None:
        if len(_POW2D_CACHE) > 64:
            _POW2D_CACHE.clear()
        n = r_grid.shape[0]
        RK = np.empty((k_hi + 1, n))
        IRK1 = np.empty((k_hi + 1, n))
        for k in range(k_hi + 1):
            rk, _rk1, irk1 = pow_tables(r_grid, k)
            RK[k] = rk
            IRK1[k] = irk1
        val = (r_grid, RK, IRK1)
        _POW2D_CACHE[key] = val
    return val[1:]


def yk_pair_dot_batch_on_grid(PA, QA, PB, QB, rows_a, rows_b, rows_k,
                              rows_ok, RhoA, cons_ptr, cons_rho, r,
                              i_cut=None):
    """Batch of ``Int rho_w(r) Y^k[rho](r) dr`` dot products where row m
    integrates rho = PA[a]·PB[b] + QA[a]·QB[b] with k = rows_k[m] and is
    dotted against the weight rows ``RhoA[cons_rho[s]]`` for its CSR
    consumer slots s in ``[cons_ptr[m], cons_ptr[m+1])``.

    Each row is the exact serial ``_yk_kernel_rho`` recurrence (the
    density is formed elementwise on the fly — same IEEE op order), so
    this is thread-parallel over ROWS only; the Y kernel itself stays
    serial per the module contract, and Y is never materialized (a
    725-row batch would be a ~75 MB matrix).  The fused dot accumulates
    in grid order (a BLAS ``rho @ y`` reorders the same sum; both are
    within one trapezoid-weight ulp).  Rows with ``rows_ok`` false
    (missing waves) leave their slots zero.
    """
    n = r.shape[0]
    ic = n - 1 if i_cut is None else min(int(i_cut), n - 1)
    val = np.zeros(cons_ptr[-1] if cons_ptr.shape[0] else 0)
    if rows_a.shape[0] == 0:
        return val
    RK, IRK1 = _pow_tables_2d(r, int(rows_k.max()))
    _yk_kernel_pair_dot_batch(PA, QA, PB, QB, rows_a, rows_b, rows_k,
                              rows_ok, RhoA, cons_ptr, cons_rho,
                              r, RK, IRK1, ic, val)
    return val


@nb.njit(cache=True, nogil=True, parallel=True)
def _yk_kernel_pair_dot_batch(PA, QA, PB, QB, rows_a, rows_b, rows_k,
                              rows_ok, RhoA, cons_ptr, cons_rho,
                              r, RK, IRK1, i_cut, val):
    """prange-over-rows fusion of the ``_yk_kernel_rho`` recurrence with
    the consumer dot products.

    Per-row Y arithmetic must stay identical to ``_yk_kernel_rho``
    (value invariance): no fastmath, no FMA contraction.  CSR slot
    ranges are disjoint per row, so the parallel writes never race.
    """
    n = r.shape[0]
    for m in nb.prange(rows_a.shape[0]):
        if not rows_ok[m]:
            continue
        s0 = cons_ptr[m]
        s1 = cons_ptr[m + 1]
        if s1 == s0:
            continue
        ia = rows_a[m]
        ib = rows_b[m]
        rk = RK[rows_k[m]]
        irk1 = IRK1[rows_k[m]]
        suf = np.empty(i_cut + 1)
        acc = 0.0
        prev_b = (PA[ia, n - 1] * PB[ib, n - 1]
                  + QA[ia, n - 1] * QB[ib, n - 1]) * irk1[n - 1]
        if i_cut == n - 1:
            suf[n - 1] = 0.0
        for i in range(n - 2, -1, -1):
            b = (PA[ia, i] * PB[ib, i] + QA[ia, i] * QB[ib, i]) * irk1[i]
            acc += 0.5 * (b + prev_b) * (r[i + 1] - r[i])
            prev_b = b
            if i <= i_cut:
                suf[i] = acc                  # suffix (outer) integrals
        inner = 0.0
        prev_a = (PA[ia, 0] * PB[ib, 0] + QA[ia, 0] * QB[ib, 0]) * rk[0]
        y = inner * irk1[0] + rk[0] * suf[0]
        for s in range(s0, s1):
            val[s] += RhoA[cons_rho[s], 0] * y
        for i in range(1, i_cut + 1):
            a = (PA[ia, i] * PB[ib, i] + QA[ia, i] * QB[ib, i]) * rk[i]
            inner += 0.5 * (a + prev_a) * (r[i] - r[i - 1])
            prev_a = a
            y = inner * irk1[i] + rk[i] * suf[i]
            for s in range(s0, s1):
                val[s] += RhoA[cons_rho[s], i] * y
    return val


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
    if i_cut == n - 1:
        suffix[n - 1] = 0.0  # backward loop starts at n-2: without this the
        # last entry is uninitialized np.empty memory whenever rho_ac is
        # above threshold at the box edge (bound-continuum exchange
        # densities) — seen as first-call-vs-rest omega drift and
        # intermittent NaN.  Mirrors _yk_kernel_rho/_yk_kernel_pair_dot_batch.
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
    "GridResourceError",
    "make_fine_grid",
    "grid_slater_rk",
    "yk_on_grid",
    "yk_pair_dot_batch_on_grid",
    "pow_tables",
    "p_rel",
]
