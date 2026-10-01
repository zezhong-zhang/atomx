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
   Salvat's power-series integrator adapts its substeps to series
   convergence. This initial-value solve requires no eigenvalue search.
4. Match the integrated (P,Q) pair to regular/irregular Dirac–Coulomb
   solutions in the constant-rV tail. Their exact two-component determinant
   fixes amplitude and phase, without a WKB or plane-wave approximation.
   Normalize the large-component amplitude at infinity to

       A² = (2 + E/c²) / (π k).

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

from atomx.constants import C_SQ as _C_SQ  # CODATA-2022 c² (M12)
from atomx.numerics.shooting.dirac_coulomb import dirac_coulomb_outgoing
from atomx.numerics.shooting.dirac_integrator import doutw
from atomx.numerics.shooting.dirac_sampling import (
    sample_dirac_solution,
    validate_output_grid,
)
from atomx.numerics.shooting.grid import sgrid
from atomx.numerics.shooting.spline import vint


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


def _prepare_potential(V_func, r_max: float, n_grid: int, Z_residual: float):
    """Grid + potential spline shared by every partial wave and energy.

    The spline of r*V(r) is independent of (kappa, E). Check its sampled
    values on every call: a callable may close over a mutable potential,
    and reusing its former spline would silently solve the wrong atom.
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
            and np.array_equal(cache["prep"][-1], rv)):
        return cache["prep"]
    rg, rvg, va, vb, vc, vd, nvt = vint(r, rv, n_out)
    # VINT's final entry is the constant-rV Coulomb tail, valid beyond
    # its final knot. Clipping to nvt-2 extrapolates the preceding cubic
    # through the entire outer box, changing the field and forcing tiny
    # integration steps in an artificial rapidly growing potential.
    ind = np.clip(
        np.searchsorted(rg[:nvt], r, side="right") - 1, 0, nvt - 1
    ).astype(np.int64)
    for arr in (va, vb, vc, vd, ind):
        with contextlib.suppress(Exception):
            arr.setflags(write=False)
    rv.setflags(write=False)
    prep = (r, n_out, va, vb, vc, vd, ind, rv)
    with contextlib.suppress(AttributeError, TypeError):
        V_func._contsolve_prep = {"key": key, "prep": prep}
    return prep


def _solve_prepared(prep, E_au: float, kappa: int, eps_tol: float,
                    *, match_phase: bool = True):
    """One partial-wave solve against a prepared potential (thread-safe)."""
    r, n_out, va, vb, vc, vd, ind, rv = prep
    P = np.zeros(n_out)
    Q = np.zeros(n_out)
    doutw(
        E_au, eps_tol, float(kappa), 100, n_out, 0,
        r, ind, n_out, va, vb, vc, vd, P, Q,
    )

    eps_rel = E_au / _C_SQ
    momentum = math.sqrt(E_au*(2.0+eps_rel))
    small_ratio = math.sqrt(E_au/(E_au+2.0*_C_SQ))
    charge = -float(va[-1])
    tail = np.flatnonzero(ind == len(va)-1)
    if not len(tail):
        raise RuntimeError('continuum grid does not reach its Coulomb tail')
    checks = sorted({max(int(tail[0]), n_out-max(2, n_out//10)), n_out-1})
    coefficients = []
    for index in checks:
        reference = dirac_coulomb_outgoing(E_au, kappa, charge, float(r[index]))
        regular, irregular = reference.imag, reference.real
        # The exact Dirac determinant F_P G_Q - F_Q G_P is -small_ratio.
        coefficients.append(np.array([
            (P[index]*irregular[1]-Q[index]*irregular[0])/(-small_ratio),
            (regular[0]*Q[index]-regular[1]*P[index])/(-small_ratio)]))
    amplitude = float(np.linalg.norm(coefficients[-1]))
    if not math.isfinite(amplitude) or amplitude <= 0:
        raise RuntimeError('continuum matching failed: nonfinite or zero amplitude')
    if (len(coefficients) > 1 and np.linalg.norm(coefficients[0]-coefficients[-1])
            > max(100*eps_tol, 1e-8)*amplitude):
        raise RuntimeError('continuum Coulomb matching is unconverged; refine the potential/grid')
    delta = math.atan2(coefficients[-1][1], coefficients[-1][0]) if match_phase else float('nan')
    norm = math.sqrt((2.0+eps_rel)/(math.pi*momentum))/amplitude
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
    interp_grid=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Adaptive power-series Dirac continuum at exact energy ``E_au``.

    Solves the radial Dirac equations for the large (P) and small (Q)
    components on a Salvat-style non-uniform grid, then phase-matches to
    the Coulomb asymptote and renormalises to the delta-energy convention.
    The squared large-component asymptotic amplitude is
    ``(2 + E/c²)/(pi*k)``; the nonrelativistic limit is ``2/(pi*k)``.

    ``delta`` is relative to the regular Dirac–Coulomb solution in Salvat's
    DCOULF convention. It is zero for a pure attractive Coulomb potential,
    up to an overall real wavefunction sign. Matching coefficients are
    checked at two points in the Coulomb tail when available. This replaces
    the former WKB amplitude and nonrelativistic gamma-function phase.
    Callers that only use P/Q may pass ``match_phase=False``; the returned
    phase is then NaN. Normalization uses the same Coulomb reference either way.

    Returns ``(r, P, Q, delta)``.  The grid and the r*V(r) spline are
    shared across partial waves and energies (cached on ``V_func``); see
    :func:`solve_dirac_continuum_many` for the thread-parallel batch API.
    ``interp_grid`` requests dense power-series output at those radii,
    within the solved interval; the returned grid is then that output grid.
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
    target = None if interp_grid is None else validate_output_grid(interp_grid, prep[0])
    r, P, Q, delta = _solve_prepared(
        prep, E_au, int(kappa), eps_tol, match_phase=match_phase)
    if target is not None:
        _, _, va, vb, vc, vd, ind, _ = prep
        P, Q = sample_dirac_solution(target, r, P, Q, ind, va, vb, vc, vd,
                                    E_au, int(kappa), eps_tol)
        r = target
    return r, P, Q, delta if match_phase else float("nan")


class DiracContinuumBatch:
    """Own numerical controls and reusable workers for successive wave batches.

    Use as a context manager. Wave results are owned by the caller and are
    not retained here. The potential is checked on each call, just as in the
    standalone batch solver, so a mutated callable cannot reuse a stale field.
    """

    def __init__(self, V_func, *, r_max=50., n_grid=1000, eps_tol=1e-10,
                 Z_residual=0., workers=None, interp_grid=None):
        self._potential = V_func
        self._controls = dict(r_max=r_max, n_grid=n_grid, eps_tol=eps_tol,
                              Z_residual=Z_residual, workers=workers,
                              interp_grid=interp_grid)
        self._executor = None
        self._workers = None
        self._closed = False

    def __enter__(self):
        if self._closed:
            raise RuntimeError('continuum batch context is closed')
        return self

    def __exit__(self, *exception):
        self.close()

    def close(self):
        """Finish submitted work and release this context's worker threads."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        self._closed = True

    def _pool(self, workers):
        from concurrent.futures import ThreadPoolExecutor
        # Preserve the standalone solver's batch-size-dependent worker count.
        if self._executor is None or self._workers != workers:
            if self._executor is not None:
                self._executor.shutdown(wait=True)
            self._executor = ThreadPoolExecutor(max_workers=workers)
            self._workers = workers
        return self._executor

    def solve_many(self, requests):
        """Return the same grid and wave mapping as the standalone solver."""
        if self._closed:
            raise RuntimeError('continuum batch context is closed')
        return solve_dirac_continuum_many(
            requests, self._potential, **self._controls, _batch=self)


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
    _batch=None,
):
    """Batch of partial-wave solves ``requests = [(kappa, E_au), ...]``.

    Thread-parallel: ``doutw`` is nogil-jitted and each solve is ~0.4-4 ms
    of numerics — the right granularity for threads (unlike the tiny
    per-row SCF kernels where fork/join dominates).  Returns
    ``(r, waves)`` with ``waves[(kappa, E_au)] = (P, Q)``.  When
    ``interp_grid`` is given, the waves are evaluated there using the same
    Dirac power series as the propagator (nogil). The returned grid matches
    the wave arrays. Extrapolation outside the solved box is rejected.
    A failed wave raises with its energy/kappa and original exception;
    failure is not represented by a missing or zero partial-wave amplitude.
    """
    import os
    from concurrent.futures import ThreadPoolExecutor

    prep = _prepare_potential(V_func, r_max, n_grid, Z_residual)
    r = prep[0]
    target = None if interp_grid is None else validate_output_grid(interp_grid, r)
    reqs = list(dict.fromkeys((int(k), float(E)) for (k, E) in requests))
    if any(k == 0 or not math.isfinite(E) or E <= 0.0 for k, E in reqs):
        raise ValueError("continuum requests require nonzero kappa and finite positive energy")
    if not reqs:
        return r if target is None else target, {}
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
            if target is not None:
                _, _, va, vb, vc, vd, ind, _ = prep
                return req, sample_dirac_solution(target, r, P, Q, ind, va, vb, vc, vd,
                                                 E, kappa, eps_tol)
            return req, (P, Q)
        except Exception as exc:
            raise RuntimeError(
                f'continuum batch wave failed: kappa={kappa}, E={E} Ha') from exc

    waves = {}
    if workers <= 1 or len(reqs) <= 2:
        for req in reqs:
            key, val = _run(req)
            waves[key] = val
    elif _batch is not None:
        for key, val in _batch._pool(workers).map(_run, reqs):
            waves[key] = val
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for key, val in pool.map(_run, reqs):
                waves[key] = val
    return r if target is None else target, waves


__all__ = ["solve_dirac_continuum_shooting", "solve_dirac_continuum_many", "DiracContinuumBatch"]
