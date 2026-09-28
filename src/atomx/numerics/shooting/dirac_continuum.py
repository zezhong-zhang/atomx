"""Dirac continuum wavefunction by direct adaptive ODE integration.

Replaces the B-spline pseudospectrum approach
(:func:`atomx.numerics.bspline.continuum.solve_continuum`) with the
**FAC-style direct integration** of the radial Dirac equation at the
*exact* requested energy.  This is what FAC does in
``recombination.c:RRRadialQkTable`` via ``OrbitalIndex(0, kappa, e)``
and what Salvat's RADIAL package does for its ``DBOUNDC`` /
``DOUTC`` continuum routines.

Approach
--------
1. Build a non-uniform radial grid via :func:`atomx.numerics.shooting.grid.sgrid`.
2. Tabulate r·V(r) on that grid and cubic-spline-interpolate via
   :func:`atomx.numerics.shooting.spline.vint`.
3. Integrate the radial Dirac equation outward from r → 0 (using
   the existing ``dir0`` short-distance power series for the seed
   and ``dir_integrate`` per interval) all the way to ``r_max``.
   The integrator is the same adaptive-Numerov-like step-doubling
   kernel that FAC and Salvat use — no eigenvalue selection.
4. Match the outer-region wavefunction to the relativistic-energy
   normalized Coulomb form with asymptotic amplitude

       A² = (2 + E/c²) / (π k),

   on the outermost ~10 % of the grid to extract the phase shift δ
   and renormalise the wave to the standard delta-energy convention.

Public API
----------
:func:`solve_dirac_continuum_shooting(E_au, kappa, V_func, …)`
    Returns ``(r, P, Q, phase_shift)``.

Reference
---------
* M. F. Gu, *Astrophys. J.* **579**, 1077 (2002), §3 — FAC's direct-
  integration continuum approach.
* F. Salvat, R. Mayol, *Comp. Phys. Commun.* **74**, 358 (1993) —
  RADIAL package, original Dirac shooting routines.
* F. Salvat, *Comp. Phys. Commun.* **240**, 165 (2019) — updated
  RADIAL with the same DIR / DOUTC kernels we use here.
"""

from __future__ import annotations

import contextlib
import math
from functools import lru_cache
from typing import Callable, Tuple

import numpy as np
from scipy.special import gamma as _sp_gamma

from atomx.constants import C_SQ as _C_SQ  # CODATA-2022 c² (M12)
from atomx.numerics.shooting.dirac_integrator import doutw, doutw_coulomb_tail
from atomx.numerics.shooting.grid import sgrid
from atomx.numerics.shooting.spline import vint

#: Wide-step Coulomb-tail continuation for e > 0 (doutw_coulomb_tail).
#: The tail of a frozen-core potential approaches r*V = -Z_res, so the
#: continuation replaces the spline by the exact asymptote; disabling this
#: falls back to the per-grid-interval integration (kept as the reference
#: path for tests).
_TAIL_WALKER = True

#: Phase budget of the Coulomb-tail continuation (radians).  The tail
#: start i_c(k) is chosen so the integrated deviation of the sampled
#: potential from the -Z_res/r asymptote contributes at most this phase
#: (and a comparable relative amplitude error) — two orders of magnitude
#: inside the 1.5 % FAC-parity gate.  The criterion is per-wave: slow
#: waves (small k) feel a given deviation more, so their i_c moves out.
_TAIL_PHASE_BUDGET = 1.0e-4

#: Pointwise guard on the tail start: no grid point in the walked region
#: may deviate from the Coulomb asymptote by more than this *relative*
#: amount.  This keeps single sharp features (which the constant-
#: coefficient wide steps would step over) on the spline-integrated path.
_TAIL_DEV_TOL = 1.0e-4

#: Minimum number of tail grid points for the continuation to engage;
#: shorter tails stay on the reference path.
_TAIL_MIN_POINTS = 32

#: Fitted spline intervals whose cubic misses the callable at the four
#: Gauss-Lobatto nodes by more than this *relative* amount (times
#: max(|Z_res|, 1)) are re-interpolated.  Natural-spline cubics deviate
#: between widely spaced outer knots — measured up to ~7e-3 in r*V on
#: the Fe SCF bench, where the tail never satisfies vint's freeze
#: tolerance — while the four-node interpolant of a smooth tail is exact
#: to roundoff, so the threshold only ever fires where the spline is
#: genuinely off.
_SPLINE_FIX_TOL = 1.0e-8

#: Gauss-Lobatto 4 nodes on [0, 1] (endpoint-inclusive, so adjacent
#: interval interpolants meet exactly at the shared knot) and Gauss 2
#: nodes (fresh check points for the strict-improvement guard).
_LOBATTO4 = np.array([0.0, 0.5 - 0.5 / np.sqrt(3.0),
                      0.5 + 0.5 / np.sqrt(3.0), 1.0])
_GAUSS2 = np.array([0.5 - 0.5 / np.sqrt(3.0), 0.5 + 0.5 / np.sqrt(3.0)])

#: Inverse Vandermonde of _LOBATTO4 with ascending power columns
#: (1, u, u^2, u^3): coefficients of the interpolating cubic in
#: u = (r - r_a)/h, as c = values @ _V4INV.T.
_V4INV = np.linalg.inv(np.vander(_LOBATTO4, 4)[:, ::-1])


@lru_cache(maxsize=64)
def _sgrid_cached(rn: float, r2: float, drn: float, n: int):
    """Cache the fast-inverted Salvat grid shared by partial waves/energies.

    Arrays are read-only; the original bisection remains the default for
    bound-state grids, while the continuum uses the equivalent Newton grid.
    """
    r, dr, n_out = sgrid(rn=rn, r2=r2, drn=drn, n=n, fast=True)
    r = np.ascontiguousarray(r[:n_out])
    r.setflags(write=False)
    return r, n_out


def _refit_deviating_intervals(V_func, r, rv, va, vb, vc, vd, ind, nvt):
    """Replace fitted-interval cubics that miss the callable at the four
    Gauss-Lobatto nodes by more than ``_SPLINE_FIX_TOL * max(|Z_res|, 1)``.

    The natural spline from :func:`vint` interpolates the tabulated r·V,
    but between widely spaced outer knots its cubics can deviate from the
    true callable (measured up to ~7e-3 in r·V on the Fe SCF bench) — a
    real error of the reference path, independent of the Coulomb-tail
    walker.  Each offending *fitted* interval (``ind < nvt-1``) is
    re-interpolated by the cubic through the four Lobatto nodes, which for
    smooth tails is exact to roundoff; adjacent intervals meet exactly at
    the shared knot because the node set includes both endpoints.

    The replacement is applied only when it is *strictly better* at the
    two Gauss nodes (a fresh check set, not the fit points), so a
    tabulated or noisy callable keeps the global spline.  Intervals in
    vint's frozen constant tail (``ind == nvt-1``) are exact by design
    and are never touched.
    """
    m = len(r) - 1
    h = np.diff(r)
    nodes = r[:m, None] + h[:, None] * _LOBATTO4[None, :]
    rn = np.where(nodes > 0.0, nodes, 1.0e-30)
    rvn = rn * np.asarray(V_func(rn), dtype=float)
    a = va[ind[:m]][:, None]
    b = vb[ind[:m]][:, None]
    c = vc[ind[:m]][:, None]
    d = vd[ind[:m]][:, None]
    sp = a + nodes * (b + nodes * (c + nodes * d))
    dev = np.abs(sp - rvn).max(axis=1)
    z_eff = max(abs(rv[-1]), 1.0)
    fitted = ind[:m] < nvt - 1
    # Safety valve: the refit is for the smooth outer tail.  Intervals
    # with non-finite or divergently large r*V (exotic model cores where
    # no cubic is a faithful representation and the historical spline is
    # the reference behaviour anyway) are left untouched.
    calm = np.isfinite(rvn).all(axis=1) \
        & (np.abs(rvn).max(axis=1) <= 1.0e6 * z_eff)
    bad = np.where(fitted & calm & np.isfinite(dev)
                   & (dev > _SPLINE_FIX_TOL * z_eff))[0]
    if bad.size == 0:
        return va, vb, vc, vd
    # coefficients of the interpolant in u = (r - r_a)/h (shared 4x4 solve)
    c0, c1, c2, c3 = (rvn[bad] @ _V4INV.T).T
    ra = r[bad]
    hh = h[bad]
    ih = 1.0 / hh
    ih2 = ih * ih
    ih3 = ih2 * ih
    # change of variables u = (x - r_a)/h to power form in x
    ra2 = ra * ra
    ra3 = ra2 * ra
    nA = c0 - c1 * ra * ih + c2 * ra2 * ih2 - c3 * ra3 * ih3
    nB = c1 * ih - 2.0 * c2 * ra * ih2 + 3.0 * c3 * ra2 * ih3
    nC = c2 * ih2 - 3.0 * c3 * ra * ih3
    nD = c3 * ih3
    # strict-improvement guard at fresh Gauss nodes
    g = ra[:, None] + hh[:, None] * _GAUSS2[None, :]
    rg_ = np.where(g > 0.0, g, 1.0e-30)
    rv_g = rg_ * np.asarray(V_func(rg_), dtype=float)
    sp_old = a[bad] + g * (b[bad] + g * (c[bad] + g * d[bad]))
    sp_new = nA[:, None] + g * (nB[:, None] + g * (nC[:, None] + g * nD[:, None]))
    improve = (np.abs(sp_new - rv_g).max(axis=1)
               < np.abs(sp_old - rv_g).max(axis=1))
    if not np.any(improve):
        return va, vb, vc, vd
    ib = ind[bad[improve]]
    va[ib] = nA[improve]
    vb[ib] = nB[improve]
    vc[ib] = nC[improve]
    vd[ib] = nD[improve]
    return va, vb, vc, vd


def _prepare_potential(V_func, r_max: float, n_grid: int, Z_residual: float):
    """Grid + potential spline shared by every partial wave and energy.

    The spline of r*V(r) is independent of (kappa, E). Check its sampled
    values on every call: a callable may close over a mutable potential,
    and reusing its former spline would silently solve the wrong atom.

    Also precomputes the Coulomb-tail criterion arrays.  The sampled
    deviation ``dev = |r*V + Z_res|`` of the asymptote and its cumulative
    action ``S[j] = ∫_{r[j]}^{r_max} dev/r' dr'`` let each wave pick the
    tail start ``i_c(k)``: the smallest radius past which the deviation
    stays pointwise below ``_TAIL_DEV_TOL * |Z_res|`` *and* its integrated
    action stays below ``_TAIL_PHASE_BUDGET * k`` — i.e. the continuation's
    phase error is bounded by ``_TAIL_PHASE_BUDGET`` radians, two orders
    inside the FAC-parity gate.  Potentials with no usable tail yield
    ``i_c = -1`` and take the historical per-interval path.
    """
    key = (float(r_max), int(n_grid), float(Z_residual))
    Z_eff = max(Z_residual, 1.0)
    r2 = 1.0e-3 / Z_eff
    drn = max(0.5, r_max / max(n_grid, 200))
    r, n_out = _sgrid_cached(float(r_max), float(r2), float(drn), int(n_grid))
    r_eval = np.where(r > 0, r, 1.0e-30)
    rv = np.ascontiguousarray(r_eval * np.asarray(V_func(r_eval), dtype=float))
    if rv.shape != r.shape or not np.all(np.isfinite(rv)):
        raise ValueError("continuum potential must be finite on the shooting grid")
    cache = getattr(V_func, "_contsolve_prep", None)
    if (cache is not None and cache.get("key") == key
            and np.array_equal(cache["prep"][7], rv)):
        return cache["prep"]
    rg, rvg, va, vb, vc, vd, nvt = vint(r, rv, n_out)
    ind = np.clip(
        np.searchsorted(rg[:nvt], r, side="right") - 1, 0, nvt - 2
    ).astype(np.int64)
    # Grid points past vint's last fitted knot belong to the frozen
    # constant-asymptote entry (va = zinf, vb = vc = vd = 0), built for
    # exactly this region.  The clipped mapping above would instead
    # extrapolate the last fitted cubic over the whole outer box — the
    # measured ~5e-4..7e-3 drift of the reference path's implied r*V.
    ind[r > rg[nvt - 1]] = nvt - 1
    va, vb, vc, vd = _refit_deviating_intervals(
        V_func, r, rv, va, vb, vc, vd, ind, nvt)
    for arr in (va, vb, vc, vd, ind):
        with contextlib.suppress(Exception):
            arr.setflags(write=False)
    rv.setflags(write=False)
    z_res = -float(rv[-1])
    dev_mask, dev_action = _coulomb_tail_criteria(r, rv, z_res)
    prep = (r, n_out, va, vb, vc, vd, ind, rv, z_res, dev_mask, dev_action)
    with contextlib.suppress(AttributeError, TypeError):
        V_func._contsolve_prep = {"key": key, "prep": prep}
    return prep


def _coulomb_tail_criteria(r, rv, z_res: float):
    """Pointwise mask and cumulative action of the Coulomb deviation.

    ``mask[j]`` is True where ``|r*V + Z_res| <= _TAIL_DEV_TOL * max(|Z_res|, 1)``
    and ``action[j] = ∫_{r[j]}^{r_max} |r*V + Z_res| / r' dr'`` (trapezoid).
    Both are length-n arrays; the walker region must satisfy the mask
    everywhere and ``action <= _TAIL_PHASE_BUDGET * k``.
    """
    dev = np.abs(rv + z_res)
    tol = _TAIL_DEV_TOL * max(abs(z_res), 1.0)
    mask = dev <= tol
    # The walked region must satisfy the mask everywhere (the caller takes
    # the last violator), so only masked deviation enters the action —
    # this also keeps near-origin singularities of exotic model potentials
    # (r*V diverging like a power of 1/r) out of the trapezoid integral.
    dev_m = np.where(mask, dev, 0.0)
    d_v = dev_m / np.maximum(r, 1.0)
    seg = 0.5 * (d_v[:-1] + d_v[1:]) * np.diff(r)
    cum = np.concatenate(([0.0], np.cumsum(seg)))
    total = cum[-1]
    action = total - cum
    with contextlib.suppress(Exception):
        mask.setflags(write=False)
        action.setflags(write=False)
    return mask, action


def _coulomb_tail_start(prep, k_inf: float) -> int:
    """Tail start index for one wave, or -1 to keep the reference path.

    The combined validity (mask everywhere, action within the phase
    budget) must hold from ``i_c`` through the last point, so ``i_c`` sits
    past the *last* violating point (plus one grid-cell margin).
    """
    _, n_out, _, _, _, _, _, _, _, mask, action = prep
    budget = _TAIL_PHASE_BUDGET * k_inf
    valid = mask & (action <= budget)
    bad = np.nonzero(~valid)[0]
    i_c = int(bad[-1]) + 2 if len(bad) else 2
    if n_out - i_c < _TAIL_MIN_POINTS:
        return -1
    return i_c


def _solve_prepared(prep, E_au: float, kappa: int, eps_tol: float,
                    *, match_phase: bool = True):
    """One partial-wave solve against a prepared potential (thread-safe)."""
    r, n_out, va, vb, vc, vd, ind, rv, z_res, _mask, _action = prep
    P = np.zeros(n_out)
    Q = np.zeros(n_out)
    eps_rel = E_au / _C_SQ
    k_inf = math.sqrt(2.0 * E_au * (1.0 + 0.5 * eps_rel))
    if E_au > 0.0:
        i_c = _coulomb_tail_start(prep, k_inf) if _TAIL_WALKER else -1
    else:
        i_c = -1
    if i_c > 0:
        # Core: per-interval integration through the Coulomb-tail start.
        # doutw's loop fills pt[0:iotp_in], so iotp_in = i_c + 1 leaves
        # pt[i_c] as the walker's seed.  Tail: wide-step continuation with
        # the converged power series as dense output — bounded phase
        # error (_TAIL_PHASE_BUDGET), same grid, same normalization.
        doutw(
            E_au, eps_tol, float(kappa), 100, i_c + 1, 0,
            r, ind, n_out, va, vb, vc, vd, P, Q,
        )
        if P[i_c] != 0.0 or Q[i_c] != 0.0:
            doutw_coulomb_tail(
                E_au, eps_tol, float(kappa), r, i_c, n_out, z_res, P, Q,
            )
    else:
        doutw(
            E_au, eps_tol, float(kappa), 100, n_out, 0,
            r, ind, n_out, va, vb, vc, vd, P, Q,
        )

    l = kappa if kappa > 0 else -kappa - 1
    # relativistic momentum and Grant's delta(E)-normalized amplitude:
    # A^2 = (E + 2c^2)/(pi c^2 k) -> 2/(pi k) as E/c^2 -> 0.  The old
    # nonrelativistic k = sqrt(2E) target left amplitude^2 low by
    # sqrt(1 + E/2c^2) per wave (9% at 200 keV) — visible as a
    # high-energy deficit in Z=54 K-shell cross sections.
    # c² comes from atomx.constants (module-level _C_SQ import), M12:
    # the local literal was the CODATA-2018 1/α², 1.4e-9 low.
    # (eps_rel and k_inf were computed above for the tail-start budget.)

    n_match = max(20, n_out // 5)
    r_match = r[-n_match:]
    P_match = P[-n_match:]
    Q_match = Q[-n_match:]
    # WKB envelope: for a slow wave over a residual ion the box edge is
    # far from asymptotic (Z_res/(k r) can exceed 1 at high Z), where the
    # local momentum k(r) >> k_inf.  The invariant is amplitude^2 * k:
    #     P = A_inf sqrt(k_inf/k(r)) sin(Int k dr)
    # so A_inf^2 = (P^2 + (P'/k(r))^2) * k(r)/k_inf.  Estimating with
    # k_inf instead under-measured A_inf for slow ejected waves in deep
    # potentials — a Z- and threshold-dependent overshoot of ionization
    # cross sections (Xe K-shell +80% at 1.5x threshold).
    V_m = rv[-n_match:] / r_match
    b_loc = E_au - V_m
    # The radial Dirac equation gives dP/dr exactly from the integrated
    # pair (P,Q). Finite differences damp an oscillatory derivative when
    # the wavelength is only moderately sampled and bias the continuum
    # normalization by about 0.7% in the FAC high-energy wave panel.
    dP = (-kappa * P_match / r_match
          + (b_loc + 2.0 * _C_SQ) / math.sqrt(_C_SQ) * Q_match)
    k_loc2 = (2.0 * b_loc * (1.0 + 0.5 * b_loc / _C_SQ)
              - l * (l + 1.0) / (r_match * r_match))
    k_loc = np.sqrt(np.maximum(k_loc2, k_inf * k_inf))
    A_sq_arr = (P_match ** 2 + (dP / k_loc) ** 2) * (k_loc / k_inf)
    A_meas = math.sqrt(float(np.median(A_sq_arr)))
    if A_meas <= 0 or not math.isfinite(A_meas):
        raise RuntimeError("continuum matching failed: nonfinite or zero asymptotic amplitude")

    # Asymptotic matching must use the *Coulomb* wave, not a plane wave:
    # the outgoing electron still feels the residual ion, so
    #     P(r) ~ A sin(k r - l pi/2 - eta ln(2 k r) + sigma_l + delta)
    # with eta = -Z_res/k and sigma_l = arg Gamma(l + 1 + i eta).  Dropping
    # the two Coulomb terms (the previous behaviour) made `delta` a
    # function of the box radius rather than a phase shift: -eta ln(2 k r)
    # alone is 12.2 rad for Z_res = 1, k = 0.271, r_max = 50.
    # See docs/PHYSICS_NUMERICS_REVIEW_2026-09-19.md finding M15.
    z_res = -float(rv[-1])          # r*V(r) -> -Z_res as r -> infinity
    delta = 0.0
    if match_phase:
        # FAC radial.c uses y=(1+alpha² E) Z/k for the relativistic
        # Coulomb logarithm. Match across the whole asymptotic window;
        # a single endpoint arcsin is ill-conditioned at extrema.
        eta = -(1.0 + eps_rel) * z_res / k_inf
        _gam = _sp_gamma(complex(l + 1.0, eta))
        sigma_l = math.atan2(_gam.imag, _gam.real)
        sample = np.linspace(0, n_match - 1, min(512, n_match), dtype=int)
        rs = r_match[sample]
        phase = np.arctan2(P_match[sample], dP[sample] / k_loc[sample])
        reference = (k_inf * rs - 0.5 * l * math.pi
                     - eta * np.log(2.0 * k_inf * rs) + sigma_l)
        circle = np.mean(np.exp(1j * (phase - reference)))
        if not np.isfinite(circle) or abs(circle) < 0.99:
            raise RuntimeError(
                "continuum phase matching is unconverged; refine the radial "
                "grid or increase the asymptotic matching radius")
        delta = float(np.angle(circle))

    norm = math.sqrt((2.0 + eps_rel) / (math.pi * k_inf)) / A_meas
    P = P * norm
    Q = Q * norm
    return r, P, Q, delta


def solve_dirac_continuum_shooting(
    E_au: float,
    kappa: int,
    V_func: Callable[[np.ndarray], np.ndarray],
    *,
    r_max: float = 50.0,
    n_grid: int = 1000,
    eps_tol: float = 1.0e-10,
    Z_residual: float = 0.0,
    match_phase: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Adaptive-Numerov Dirac continuum at exact energy ``E_au``.

    Solves the radial Dirac equations for the large (P) and small (Q)
    components on a Salvat-style non-uniform grid, then phase-matches to
    the Coulomb asymptote and renormalises to the delta-energy convention.
    The squared large-component asymptotic amplitude is
    ``(2 + E/c²)/(pi*k)``; the nonrelativistic limit is ``2/(pi*k)``.

    ``delta`` is referenced to the Coulomb logarithm with relativistic
    Sommerfeld parameter and the conventional ``arg Gamma(l+1+i*eta)``
    constant. FAC's ``PhaseRDependent`` uses a different constant, so its
    printed phase is not directly comparable. An unconverged radial phase
    fit raises; the batch API omits the phase fit for wave-only consumers.
    Callers that only use P/Q may pass ``match_phase=False``; the returned
    phase is then NaN rather than a falsely precise number.

    Returns ``(r, P, Q, delta)``.  The grid and the r*V(r) spline are
    shared across partial waves and energies (cached on ``V_func``); see
    :func:`solve_dirac_continuum_many` for the thread-parallel batch API.
    """
    if not math.isfinite(E_au) or E_au <= 0:
        raise ValueError("solve_dirac_continuum_shooting: E_au must be finite and > 0")
    if kappa == 0:
        raise ValueError("kappa must be non-zero")
    if not math.isfinite(r_max) or r_max <= 0 or n_grid < 20:
        raise ValueError("continuum shooting requires r_max > 0 and n_grid >= 20")
    if not math.isfinite(eps_tol) or eps_tol <= 0:
        raise ValueError("eps_tol must be finite and positive")
    prep = _prepare_potential(V_func, r_max, n_grid, Z_residual)
    r, P, Q, delta = _solve_prepared(
        prep, E_au, int(kappa), eps_tol, match_phase=match_phase)
    return r, P, Q, delta if match_phase else float("nan")


def solve_dirac_continuum_many(
    requests,
    V_func,
    *,
    r_max: float = 50.0,
    n_grid: int = 1000,
    eps_tol: float = 1.0e-10,
    Z_residual: float = 0.0,
    workers: int | None = None,
    interp_grid=None,
):
    """Batch of partial-wave solves ``requests = [(kappa, E_au), ...]``.

    Thread-parallel: ``doutw`` is nogil-jitted and each solve is ~0.4-4 ms
    of numerics — the right granularity for threads (unlike the tiny
    per-row SCF kernels where fork/join dominates).  Returns
    ``(r, waves)`` with ``waves[(kappa, E_au)] = (P, Q)``.  When
    ``interp_grid`` is given, the waves are linearly interpolated onto it
    inside the worker threads (np.interp releases the GIL).
    """
    import os
    from concurrent.futures import ThreadPoolExecutor

    prep = _prepare_potential(V_func, r_max, n_grid, Z_residual)
    r = prep[0]
    reqs = list(dict.fromkeys((int(k), float(E)) for (k, E) in requests))
    if any(k == 0 or not math.isfinite(E) or E <= 0.0 for k, E in reqs):
        raise ValueError("continuum requests require nonzero kappa and finite positive energy")
    if not reqs:
        return r, {}
    if workers is None:
        # A dozen tiny radial solves do not repay 12–16 worker startups.
        # The measured 24-wave panel was fastest with 2–4 workers; leave
        # eight for larger batches where each thread has enough work.
        workers = min(len(reqs), os.cpu_count() or 1,
                      4 if len(reqs) < 32 else 8)
    if workers < 1:
        raise ValueError("workers must be positive")

    def _run(req):
        kappa, E = req
        try:
            _r, P, Q, _d = _solve_prepared(
                prep, E, kappa, eps_tol, match_phase=False)
            if interp_grid is not None:
                return req, (np.interp(interp_grid, r, P),
                             np.interp(interp_grid, r, Q))
            return req, (P, Q)
        except Exception:
            return req, None

    waves = {}
    if workers <= 1 or len(reqs) <= 2:
        for req in reqs:
            key, val = _run(req)
            waves[key] = val
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for key, val in pool.map(_run, reqs):
                waves[key] = val
    return r, waves


__all__ = ["solve_dirac_continuum_shooting", "solve_dirac_continuum_many"]
