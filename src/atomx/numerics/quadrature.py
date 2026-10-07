"""Finite-interval rules and arbitrary-grid Slater/Y^k quadrature.

``sinh_squared_quadrature`` supplies a general positive-interval rule with
square-root endpoint resolution and a characteristic scale. Physics callers
choose the interval, scale and quadrature order.

This is the validated pairing that fixed the 2026-07 quadrature bug
(docs/scattering_fac_verification_2026-07.md): a composite radial grid
that resolves BOTH the inner bound structure and the continuum
oscillation, together with trapezoidal cumulative sums (second order on
non-uniform grids).  Rectangle-rule cumulatives on a structured grid are
O(h)-biased and produced a uniform ~15% error in Auger radial integrals
before the fix — new consumers must import from here rather than
re-deriving either half.

Numerical contract for the radial Slater/Y^k functions:
* grids may be arbitrary/non-uniform; spacings are taken from ``r``;
* Y^k and R^k are exact trapezoidal quadratures of
  Y^k(r) = Int (r_<^k / r_>^{k+1}) rho(r') dr' and
  R^k    = Int rho_ac(r) Y^k[rho_bd](r) dr;
* no pointwise relative-density tail cutoff is imposed;
* source rows may run in parallel; consumer dots retain radial order;
* powers use bounded local radius ratios and exact block rescaling.
The B-spline Gauss quadrature (numerics/bspline) and the Salvat SLAG6
machinery (numerics/shooting) remain the right tools ON THEIR OWN grids;
this module integrates samples of native bound expansions and continuum
dense output on scattering-style composite grids.
"""

from __future__ import annotations

import math
import operator
from functools import lru_cache

import numba as nb
import numpy as np
from scipy.special import roots_legendre

from atomx.constants import C_SQ  # CODATA-2022 c² (was a stale 2018 literal)


class GridResourceError(RuntimeError):
    """Raised when a requested scattering quadrature grid exceeds its cap."""


@lru_cache(maxsize=32)
def _gauss_legendre_rule(order):
    nodes, weights = roots_legendre(order)
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return nodes, weights


def gauss_legendre_rule(order):
    """Read-only cached Gauss-Legendre rule on [-1, 1].

    Cache identity is the positive integer order; no interval or physical
    inputs enter this canonical rule. Copy before modifying its arrays.
    SciPy constructs the roots through a banded Jacobi eigensolve rather
    than NumPy's dense companion eigensolve at large orders.
    """
    if isinstance(order, (bool, np.bool_)):
        raise ValueError("quadrature order must be a positive integer")
    try:
        n = operator.index(order)
    except TypeError as exc:
        raise ValueError("quadrature order must be a positive integer") from exc
    if n < 1:
        raise ValueError("quadrature order must be a positive integer")
    return _gauss_legendre_rule(n)


def sinh_squared_quadrature(n, upper, scale):
    """Nodes and weights for an integral on ``[0, upper]``.

    Gauss-Legendre integration in ``x`` under ``E = scale*sinh(x)**2``
    resolves square-root endpoint behavior and a positive characteristic
    scale even when the interval is much larger. All nodes lie strictly
    inside the interval; no endpoint cutoff is introduced. Weights include
    the full Jacobian ``scale*sinh(2*x)``.
    """
    if isinstance(n, bool) or not np.isfinite(n) or int(n) != n or n < 1:
        raise ValueError('quadrature order must be a positive integer')
    if not (np.isfinite(upper) and upper > 0 and np.isfinite(scale) and scale > 0):
        raise ValueError('quadrature upper limit and scale must be finite and positive')
    t, w = gauss_legendre_rule(int(n))
    extent = np.arcsinh(np.sqrt(upper / scale))
    x = .5 * extent * (t + 1.)
    nodes = scale * np.sinh(x)**2
    weights = .5 * extent * scale * np.sinh(2*x) * w
    return nodes, weights


def p_rel(E_au):
    """Relativistic momentum p = sqrt(2E + (E/c)^2) in a.u."""
    return math.sqrt(2.0 * E_au + (E_au * E_au) / C_SQ)


@lru_cache(maxsize=64)
def _fine_grid_cached(
    r_max: float,
    k_max_au: float,
    z_inner: float,
    max_points: int,
    refinement: int,
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
    if refinement > 1:
        required = (len(arr)-1)*refinement+1
        if required > max_points:
            raise GridResourceError(
                'make_fine_grid refinement exceeded radial grid cap: '
                f'required_points={required}, cap={max_points}, '
                f'p={k_max_au:.12g} a.u., r_max={r_max:.12g} a0, '
                f'z_inner={z_inner:.12g}, refinement={refinement}')
        # Nested subdivision keeps the original domain and every coarse
        # node; fresh wave samples, not interpolated integrands, are used.
        arr = np.r_[(arr[:-1,None] + np.diff(arr)[:,None]
                      * (np.arange(refinement)/refinement)).ravel(), arr[-1]]
    arr.setflags(write=False)
    return arr


def make_fine_grid(r_max, k_max_au, z_inner=30.0, *, max_points=2_000_000,
                   refinement=1):
    """Composite radial grid resolving inner bound structure AND the
    continuum oscillation.

    A uniform grid sized by the continuum wavelength under-resolves deep
    K/L-shell orbitals (Fe 1s: ~5 points across the peak at the old
    h = r_max/n).  The step obeys two local constraints:

        dr <= 0.15 / p_loc(r),  p_loc = sqrt(k_max^2 + 2 z_inner / r)
        dr <= r / 48            (>= 48 points per radial e-fold)

    Pass the actual nuclear Z as ``z_inner`` to resolve the inner Coulomb field.
    The momentum is rounded up to 0.25 a.u. so nearby energies share a
    cached grid. These step constraints are heuristics, not error bounds.
    ``refinement`` is a positive integer: subdivide every interval into that
    many equal pieces, independently of the continuum propagation grid.
    This supplies nested meshes for observable quadrature convergence.
    """
    for name, value, minimum in [('refinement', refinement, 1), ('max_points', max_points, 2)]:
        if (isinstance(value, (bool, np.bool_)) or not np.isfinite(value)
                or int(value) != value or value < minimum):
            raise ValueError(f'{name} must be an integer >= {minimum}')
    if not (np.isfinite(k_max_au) and k_max_au > 0):
        raise ValueError('k_max_au must be finite and positive')
    k_key = float(np.ceil(max(float(k_max_au), 1.0) * 4.0) / 4.0)
    return _fine_grid_cached(float(r_max), k_key, float(z_inner), int(max_points),
                             int(refinement))


# Cache only immutable grids, keeping a strong reference against id reuse.
# Writable caller grids are rebuilt so in-place edits cannot stale geometry.
_COULOMB_CACHE: dict = {}


def _coulomb_tables(r, k_max):
    """Scaled powers in radial blocks, with bounded frame-to-frame transfers.

    Within a block x=r/r_anchor and (k_max+1)*log(x) <= 128. This keeps
    powers far from floating overflow/underflow while retaining the fast
    additive cumulative quadrature. Block changes multiply carried moments
    by the k-th power of the smaller/larger anchor ratio in either direction.
    Both transfer factors are at most one. The physical Green kernel is
    unchanged; no source or multipole is discarded.
    """
    if isinstance(k_max, (bool, np.bool_)) or int(k_max) != k_max or k_max < 0:
        raise ValueError('Coulomb multipole rank must be a nonnegative integer')
    key = id(r)
    cached = _COULOMB_CACHE.get(key) if not r.flags.writeable else None
    if cached is not None and cached[1].shape[0] >= k_max+1:
        return cached[1:]
    if (r.ndim != 1 or not r.size or not np.isfinite(r).all()
            or np.any(r <= 0) or np.any(np.diff(r) <= 0)):
        raise ValueError('Coulomb grid must be finite, positive and strictly increasing')
    logs, cuts = np.log(r), [0]
    span = 128. / (k_max+1)
    while cuts[-1] < len(r):
        stop = int(np.searchsorted(logs, logs[cuts[-1]]+span, side='right'))
        cuts.append(max(cuts[-1]+1, stop))
    cuts = np.asarray(cuts, dtype=np.int64)
    anchors = r[cuts[:-1]]
    rk = np.empty((k_max+1, len(r)))
    inverse = np.empty_like(rk)
    orders = np.arange(k_max+1)[:, None]
    for block in range(len(anchors)):
        lo, hi = cuts[block:block+2]
        x = r[lo:hi] / anchors[block]
        rk[:, lo:hi] = x[None,:] ** orders
        inverse[:, lo:hi] = x[None,:] ** (-orders-1) / anchors[block]
    transfers = ((anchors[:-1]/anchors[1:])[None,:]
                 ** np.arange(k_max+1)[:,None])
    values = (rk, inverse, cuts, transfers)
    if not r.flags.writeable:
        if key not in _COULOMB_CACHE and len(_COULOMB_CACHE) >= 64:
            _COULOMB_CACHE.clear()
        _COULOMB_CACHE[key] = (r, *values)
    return values


def yk_on_grid(rho, r, k, i_cut=None):
    """Trapezoidal action of the Coulomb Green kernel on a radial density.

    ``i_cut`` restricts returned target radii; the source integral always
    spans the full grid. Bounded local radius ratios avoid absolute r**k.
    """
    r = np.asarray(r, dtype=float)
    rk, inverse, cuts, transfers = _coulomb_tables(r, k)
    ic = len(r)-1 if i_cut is None else min(int(i_cut), len(r)-1)
    return _yk_kernel_rho(rho, r, rk[k], inverse[k], cuts, transfers[k], ic)


@nb.njit(cache=True, nogil=True)
def _yk_kernel_rho(rho, r, rk, inverse, cuts, transfer, i_cut):
    """Block-scaled cumulative trapezoids; no change to the quadrature rule."""
    n = len(r)
    y = np.empty(i_cut+1)
    acc, prev = 0., rho[n-1]*inverse[n-1]
    if i_cut == n-1:
        y[n-1] = 0.
    for block in range(len(cuts)-2, -1, -1):
        if block < len(cuts)-2:
            acc *= transfer[block]
            prev *= transfer[block]
        for index in range(min(cuts[block+1]-1, n-2), cuts[block]-1, -1):
            i = np.uint64(index)
            b = rho[i]*inverse[i]
            acc += .5*(b+prev)*(r[i+np.uint64(1)]-r[i])
            if i <= i_cut:
                y[i] = acc
            prev = b
    inner, prev = 0., rho[0]*rk[0]
    y[0] *= rk[0]
    for block in range(len(cuts)-1):
        if cuts[block] > i_cut:
            break
        if block:
            inner *= transfer[block-1]
            prev *= transfer[block-1]
        for index in range(max(1, cuts[block]), min(cuts[block+1], i_cut+1)):
            i = np.uint64(index)
            a = rho[i]*rk[i]
            inner += .5*(a+prev)*(r[i]-r[i-np.uint64(1)])
            y[i] = (inner*inverse[i] + rk[i]*y[i])
            prev = a
    return y


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
    serial per the module contract. Each worker reuses its suffix scratch
    array for one Y row; no rows-by-grid matrix is allocated. Consumer dots
    traverse contiguous radial rows in grid order, avoiding a strided
    consumer loop at every radius. Rows with ``rows_ok`` false
    (missing waves) leave their slots zero.
    """
    r = np.asarray(r, dtype=float)
    n = r.shape[0]
    ic = n - 1 if i_cut is None else min(int(i_cut), n - 1)
    val = np.zeros(cons_ptr[-1] if cons_ptr.shape[0] else 0)
    if rows_a.shape[0] == 0:
        return val
    RK, inverse, cuts, transfers = _coulomb_tables(r, int(rows_k.max()))
    _yk_kernel_pair_dot_batch(PA, QA, PB, QB, rows_a, rows_b, rows_k,
                              rows_ok, RhoA, cons_ptr, cons_rho,
                              r, RK, inverse, cuts, transfers, ic, val)
    return val


@nb.njit(cache=True, nogil=True, inline="always")
def _pair_yk(P_a, Q_a, P_b, Q_b, r, rk, inverse, cuts, transfer, i_cut):
    """Fused transition density and block-scaled Green action.

    Validated block limits give nonnegative indices. Explicit unsigned
    indices avoid negative-index adjustments on every wave-array load.
    """
    n = len(r)
    y = np.empty(i_cut+1)
    acc = 0.
    prev = (P_a[n-1]*P_b[n-1]+Q_a[n-1]*Q_b[n-1])*inverse[n-1]
    if i_cut == n-1:
        y[n-1] = 0.
    for block in range(len(cuts)-2, -1, -1):
        if block < len(cuts)-2:
            acc *= transfer[block]
            prev *= transfer[block]
        for index in range(min(cuts[block+1]-1,n-2), cuts[block]-1, -1):
            i = np.uint64(index)
            b = (P_a[i]*P_b[i]+Q_a[i]*Q_b[i])*inverse[i]
            acc += .5*(b+prev)*(r[i+np.uint64(1)]-r[i])
            if i <= i_cut:
                y[i] = acc
            prev = b
    inner = 0.
    prev = (P_a[0]*P_b[0]+Q_a[0]*Q_b[0])*rk[0]
    y[0] *= rk[0]
    for block in range(len(cuts)-1):
        if cuts[block] > i_cut:
            break
        if block:
            inner *= transfer[block-1]
            prev *= transfer[block-1]
        for index in range(max(1,cuts[block]), min(cuts[block+1],i_cut+1)):
            i = np.uint64(index)
            a = (P_a[i]*P_b[i]+Q_a[i]*Q_b[i])*rk[i]
            inner += .5*(a+prev)*(r[i]-r[i-np.uint64(1)])
            y[i] = (inner*inverse[i]+rk[i]*y[i])
            prev = a
    return y


@nb.njit(cache=True, nogil=True, parallel=True)
def _yk_kernel_pair_dot_batch(PA, QA, PB, QB, rows_a, rows_b, rows_k,
                              rows_ok, RhoA, cons_ptr, cons_rho,
                              r, RK, inverse, cuts, transfers, i_cut, val):
    """Parallel source rows, shared Green action, contiguous consumer dots."""
    for m in nb.prange(len(rows_a)):
        s0, s1 = cons_ptr[m], cons_ptr[m+1]
        if not rows_ok[m] or s0 == s1:
            continue
        ia, ib, k = rows_a[m], rows_b[m], rows_k[m]
        y = _pair_yk(PA[ia], QA[ia], PB[ib], QB[ib], r, RK[k], inverse[k],
                     cuts, transfers[k], i_cut)
        for s in range(s0,s1):
            total = 0.
            consumer = cons_rho[s]
            for i in range(i_cut+1):
                total += RhoA[consumer,i]*y[i]
            val[s] = total


@nb.njit(cache=True, nogil=True)
def _slater_rk_kernel(P_a, Q_a, P_b, Q_b, P_c, Q_c, P_d, Q_d,
                      r, rk, inverse, cuts, transfer):
    """Full-interval trapezoidal Slater integral; skip only exact zero support."""
    i_cut = 0
    for i in range(len(r)-1,-1,-1):
        if P_a[i]*P_c[i]+Q_a[i]*Q_c[i] != 0.:
            # Include the following endpoint and its last half-cell weight.
            i_cut = min(i+1,len(r)-1)
            break
    y = _pair_yk(P_b,Q_b,P_d,Q_d,r,rk,inverse,cuts,transfer,i_cut)
    prev = (P_a[0]*P_c[0]+Q_a[0]*Q_c[0])*y[0]
    total = 0.
    for i in range(1,i_cut+1):
        g = (P_a[i]*P_c[i]+Q_a[i]*Q_c[i])*y[i]
        total += .5*(g+prev)*(r[i]-r[i-1])
        prev = g
    return total


def grid_slater_rk(P_a, Q_a, P_b, Q_b, P_c, Q_c, P_d, Q_d, r_grid, k):
    """R^k(a,b,c,d) = Int rho_ac(r) Y^k[rho_bd](r) dr on an arbitrary grid.

    Argument order as ``compute_slater_rk_cell`` (rho13 = a*c,
    rho24 = b*d).  The fused numba kernel reproduces the reference
    ``yk_on_grid`` + dot implementation exactly up to floating-point
    summation order.
    """
    r_grid = np.asarray(r_grid, dtype=float)
    rk, inverse, cuts, transfers = _coulomb_tables(r_grid, k)
    return float(_slater_rk_kernel(P_a, Q_a, P_b, Q_b, P_c, Q_c, P_d, Q_d,
                                   r_grid, rk[k], inverse[k], cuts, transfers[k]))


__all__ = [
    "GridResourceError",
    "sinh_squared_quadrature",
    "make_fine_grid",
    "grid_slater_rk",
    "yk_on_grid",
    "yk_pair_dot_batch_on_grid",
    "p_rel",
]
