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
4. Match the outer-region wavefunction to the asymptotic
   free-particle / Coulomb form

       P(r → ∞)  →  √(2 / (π k))·sin(k r − l π/2 + δ)

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

import math
from typing import Callable, Tuple

import numpy as np

from functools import lru_cache

from atomx.numerics.shooting.dirac_integrator import doutw
from atomx.numerics.shooting.grid import sgrid
from atomx.numerics.shooting.spline import vint


class ContinuumBatchError(RuntimeError):
    """Aggregated failure from a batch of continuum partial-wave solves."""

    def __init__(self, failures):
        self.failures = tuple(failures)
        details = "; ".join(
            f"kappa={kappa}, E_au={energy:.12g}: {reason}"
            for kappa, energy, reason in self.failures[:8]
        )
        if len(self.failures) > 8:
            details += f"; ... {len(self.failures) - 8} more"
        super().__init__(f"continuum batch failed for {len(self.failures)} request(s): {details}")


@lru_cache(maxsize=64)
def _sgrid_cached(rn: float, r2: float, drn: float, n: int):
    """Cache the Salvat grid: construction is ~15x the cost of the actual
    Dirac integration, and every partial wave / energy in a scattering run
    shares the same grid parameters.  Arrays are marked read-only."""
    r, dr, n_out = sgrid(rn=rn, r2=r2, drn=drn, n=n)
    r = np.ascontiguousarray(r[:n_out])
    r.setflags(write=False)
    return r, n_out


def _prepare_potential(V_func, r_max: float, n_grid: int, Z_residual: float):
    """Grid + potential spline shared by every partial wave and energy.

    The spline of r*V(r) is independent of (kappa, E); rebuilding it per
    solve cost ~30% of each call.  Cached on the ``V_func`` object itself
    so all solves against the same potential reuse it.
    """
    key = (float(r_max), int(n_grid), float(Z_residual))
    cache = getattr(V_func, "_contsolve_prep", None)
    if cache is not None and cache.get("key") == key:
        return cache["prep"]
    Z_eff = max(Z_residual, 1.0)
    r2 = 1.0e-3 / Z_eff
    drn = max(0.5, r_max / max(n_grid, 200))
    r, n_out = _sgrid_cached(float(r_max), float(r2), float(drn), int(n_grid))
    r_eval = np.where(r > 0, r, 1.0e-30)
    rv = r_eval * np.asarray(V_func(r_eval), dtype=float)
    rg, rvg, va, vb, vc, vd, nvt = vint(r, rv, n_out)
    ind = np.clip(
        np.searchsorted(rg[:nvt], r, side="right") - 1, 0, nvt - 2
    ).astype(np.int64)
    for arr in (va, vb, vc, vd, ind):
        try:
            arr.setflags(write=False)
        except Exception:
            pass
    rv = np.ascontiguousarray(rv)
    rv.setflags(write=False)
    prep = (r, n_out, va, vb, vc, vd, ind, rv)
    try:
        V_func._contsolve_prep = {"key": key, "prep": prep}
    except (AttributeError, TypeError):
        pass
    return prep


def _solve_prepared(prep, E_au: float, kappa: int, eps_tol: float):
    """One partial-wave solve against a prepared potential (thread-safe)."""
    r, n_out, va, vb, vc, vd, ind, rv = prep
    P = np.zeros(n_out)
    Q = np.zeros(n_out)
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
    _C_SQ = 137.0359990836 ** 2
    eps_rel = E_au / _C_SQ
    k_inf = math.sqrt(2.0 * E_au * (1.0 + 0.5 * eps_rel))

    n_match = max(20, n_out // 5)
    r_match = r[-n_match:]
    P_match = P[-n_match:]
    dP = np.gradient(P_match, r_match)
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
    k_loc2 = (2.0 * b_loc * (1.0 + 0.5 * b_loc / _C_SQ)
              - l * (l + 1.0) / (r_match * r_match))
    k_loc = np.sqrt(np.maximum(k_loc2, k_inf * k_inf))
    A_sq_arr = (P_match ** 2 + (dP / k_loc) ** 2) * (k_loc / k_inf)
    A_meas = math.sqrt(float(np.median(A_sq_arr)))
    if A_meas <= 0 or not math.isfinite(A_meas):
        return r, P, Q, 0.0

    sin_arg = max(-1.0, min(1.0, P[-1] / A_meas))
    phase = math.asin(sin_arg)
    if dP[-1] < 0:
        phase = math.pi - phase
    delta = phase - k_inf * r[-1] + 0.5 * l * math.pi
    delta = (delta + math.pi) % (2 * math.pi) - math.pi

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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Adaptive-Numerov Dirac continuum at exact energy ``E_au``.

    Solves the radial Dirac equations for the large (P) and small (Q)
    components on a Salvat-style non-uniform grid, then phase-matches to
    the free-particle asymptote in the outer region and renormalises to
    the delta-energy convention

        P(r -> inf)  ->  sqrt(2 / (pi k)) * sin(k r - l pi/2 + delta).

    Returns ``(r, P, Q, delta)``.  The grid and the r*V(r) spline are
    shared across partial waves and energies (cached on ``V_func``); see
    :func:`solve_dirac_continuum_many` for the thread-parallel batch API.
    """
    if E_au <= 0:
        raise ValueError("solve_dirac_continuum_shooting: E_au must be > 0")
    if kappa == 0:
        raise ValueError("kappa must be non-zero")
    prep = _prepare_potential(V_func, r_max, n_grid, Z_residual)
    return _solve_prepared(prep, E_au, int(kappa), eps_tol)


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
    fail_closed: bool = False,
    return_failures: bool = False,
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
    reqs = [(int(k), float(E)) for (k, E) in requests]
    if workers is None:
        workers = min(len(reqs), os.cpu_count() or 1, 16)

    def _run(req):
        kappa, E = req
        try:
            _r, P, Q, _d = _solve_prepared(prep, E, kappa, eps_tol)
            if interp_grid is not None:
                return req, (np.interp(interp_grid, r, P),
                             np.interp(interp_grid, r, Q)), None
            return req, (P, Q), None
        except Exception as exc:  # noqa: BLE001 - surfaced below with context
            return req, None, f"{type(exc).__name__}: {exc}"

    waves = {}
    failures = []
    if workers <= 1 or len(reqs) <= 2:
        for req in reqs:
            key, val, failure = _run(req)
            waves[key] = val
            if failure is not None:
                failures.append((key[0], key[1], failure))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for key, val, failure in pool.map(_run, reqs):
                waves[key] = val
                if failure is not None:
                    failures.append((key[0], key[1], failure))
    if fail_closed and failures:
        raise ContinuumBatchError(failures)
    if return_failures:
        return r, waves, failures
    return r, waves


__all__ = [
    "ContinuumBatchError",
    "solve_dirac_continuum_shooting",
    "solve_dirac_continuum_many",
]
