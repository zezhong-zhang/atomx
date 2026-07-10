"""Build a numerical V(r) callable from any AtomicResultLike.

The continuum solvers in this package
(:func:`atomx.numerics.shooting.dirac_continuum.solve_dirac_continuum_shooting`)
take an arbitrary radial potential as a Python callable.  This module
ships factories for the most common potentials so a user can swap

    V = -Z/r                                   (bare nucleus)
    V = -Z/r + Σ_b q_b · V_d^{(0)}(b)          (static-Hartree)
    V = V_HF (full)                            (static-Hartree + EXCHANGE
                                                — non-local; for a local
                                                approximation use V_HFS)

without re-implementing the construction each time.

Decoupling philosophy
---------------------
* :mod:`atomx.numerics` provides numerical kernels (ODE integrators,
  splines, grids) that consume potentials as **callables**.
* :mod:`atomx.scattering` and :mod:`atomx.properties` build the appropriate
  callable from an ``AtomicResultLike`` (B-spline DiracFock, MCDHF,
  shooting result, mock backend) by composing the helpers below.
* Users can write their own callable for custom potentials (e.g.
  pseudopotentials, nuclear models with finite size) and pass it
  through unchanged.

Public API
----------
:func:`bare_nuclear_potential(Z)` — V(r) = -Z/r.

:func:`static_hartree_potential(scf_result, exclude_label=None)` —
    V(r) = -Z/r + Σ_b q_b' · V_d^{(0)}(b)(r), optionally plus
    local FAC/Slater exchange
    where q_b' = q_b − δ(b == exclude) gives the V^{N-1} flavour
    used by FAC and Pindzola-Griffin for distorted waves.

:func:`tabulated_potential(r_grid, V_grid)` — wrap a precomputed
    array on an arbitrary grid as a piecewise-linearly-interpolated
    callable.

Reference
---------
* M. F. Gu, *Astrophys. J.* **579**, 1077 (2002), §3 — FAC's V_HF
  static-Hartree distorted waves.
* M. S. Pindzola, D. C. Griffin, *Phys. Rev. A* **36**, 2628 (1987).
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np


def bare_nuclear_potential(Z: float) -> Callable[[np.ndarray], np.ndarray]:
    """Bare-nuclear Coulomb potential ``V(r) = -Z/r``.

    Returns a callable suitable for any ODE integrator that expects
    a radial potential.

    Parameters
    ----------
    Z : float
        Nuclear charge.

    Returns
    -------
    V_func : callable
        ``V_func(r)`` returns -Z/r at each r in the input array.
        Handles r=0 by clamping to a tiny positive value (the radial
        Dirac equation regularises the singularity through the
        small-r power-series anyway).
    """
    Z_f = float(Z)

    def V(r: np.ndarray) -> np.ndarray:
        r_arr = np.asarray(r, dtype=float)
        return -Z_f / np.where(r_arr > 0, r_arr, 1.0e-30)

    return V


def tabulated_potential(
    r_grid: np.ndarray, V_grid: np.ndarray,
    *,
    fill_outer: float = 0.0,
) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap a precomputed (r, V) table as a callable.

    Linear interpolation on the input grid; outside the grid the
    potential is extrapolated as ``fill_outer`` (default 0 — assumes
    the user has truncated to the asymptotic free-particle region).

    Parameters
    ----------
    r_grid, V_grid : array_like
        Sorted radial grid (a₀) and matching potential values (a.u.).
    fill_outer : float
        Constant extrapolation beyond ``r_grid[-1]``.

    Returns
    -------
    V_func : callable
    """
    r_arr = np.asarray(r_grid, dtype=float)
    V_arr = np.asarray(V_grid, dtype=float)
    if r_arr.shape != V_arr.shape:
        raise ValueError("r_grid and V_grid must have matching shape")
    if not np.all(np.diff(r_arr) >= 0):
        raise ValueError("r_grid must be non-decreasing")

    def V(r: np.ndarray) -> np.ndarray:
        r_in = np.asarray(r, dtype=float)
        return np.interp(r_in, r_arr, V_arr,
                          left=V_arr[0], right=fill_outer)

    return V


def static_hartree_potential(
    scf_result,
    *,
    exclude_label: Optional[str] = None,
    n_radial: int = 4000,
    r_max: Optional[float] = None,
    include_exchange: bool = False,
    exchange: str = "fac",
    alpha_x: float = 2.0 / 3.0,
    fac_ihx: float = -1.0,
    fac_hxs: float = 1.0,
    fac_csi: float = 0.55,
    fac_self_interaction_correction: bool = True,
    latter_correction: bool = True,
) -> Callable[[np.ndarray], np.ndarray]:
    """V_static-Hartree(r) = -Z/r + Σ_b q_b' · V_d^{(0)}(b)(r).

    Builds the *local* static-Hartree potential seen by a continuum
    electron from a converged SCF reference.  When ``include_exchange``
    is true, adds a local-density exchange approximation on the same
    residual-core density.  That gives a practical V_HFS/V_DFS
    distorted-wave potential for FAC parity while keeping the historic
    pure-Hartree default explicit.

    The bound-electron monopole Hartree potential at radius r is

        V_d(b)(r) = ∫_0^∞ ρ_b(r') / max(r, r') dr'

    where ρ_b(r) = P_b²(r) + Q_b²(r) is the radial density of orbital
    b.  Numerically integrated on a dense uniform grid.

    Parameters
    ----------
    scf_result
        Solved SCFResult / MCDHFResult satisfying the
        :class:`atomx.scattering.protocols.AtomicResultLike` contract.
    exclude_label : str, optional
        Subshell label whose contribution to the Hartree potential
        should be reduced by 1 electron — the standard V^{N-1}
        prescription for the photoelectron seeing the residual ion
        of one less electron of orbital ``a`` (the orbital that's
        being ionised).  Default: V^N (full SCF potential).
    n_radial : int
        Number of radial-grid points for tabulating V (default 4000).
    r_max : float, optional
        Outer radius (a₀).  Default: pick from the SCF result's
        B-spline knot vector if available, else 50 a₀.
    include_exchange : bool
        Add local exchange on the residual-core density.
    exchange : {"fac", "slater"}
        Local exchange flavour.  ``"fac"`` uses Gu's Z,N-dependent
        coefficient; ``"slater"`` uses ``alpha_x``.
    alpha_x : float
        Slater exchange coefficient when ``exchange="slater"``.
    fac_ihx, fac_hxs, fac_csi : float
        FAC/Gu local-exchange and screening-tail parameters.
    fac_self_interaction_correction, latter_correction : bool
        Apply FAC's default local-potential screening-tail correction and
        final asymptotic Latter cap when using FAC exchange.

    Returns
    -------
    V_func : callable
        ``V_func(r)`` evaluates V_HF(r) on any input r.
    """
    from atomx.scattering.protocols import grid_data, list_occupied_orbitals
    Z = float(scf_result.Z)

    # Determine outer radius
    bsp_data = grid_data(scf_result)
    if r_max is None:
        if bsp_data is not None and "t" in bsp_data:
            r_max = float(bsp_data["t"][-1])
        else:
            r_max = 50.0

    # Build a uniform radial grid (skip r = 0 to avoid the 1/r blow-up)
    r_h = np.linspace(r_max / n_radial, r_max, n_radial)

    # Evaluate each occupied orbital's radial density on r_h
    # We use the existing eval_P / eval_Q helpers if a B-spline grid
    # is available; otherwise interpolate from the orbital's stored
    # (P, Q) arrays directly.
    rho = np.zeros_like(r_h)
    occ_list = list_occupied_orbitals(scf_result)
    N_eff = 0.0
    if bsp_data is not None and "gr" in bsp_data and "bsp_p" in bsp_data:
        from atomx.numerics.bspline.evaluate import eval_P, eval_Q
        gr_b = bsp_data["gr"].ravel()
        for (label, kappa, _eps, q) in occ_list:
            sh = next(s for s in scf_result.subshells if s.label == label)
            same_kappa = [s for s in scf_result.subshells if s.kappa == kappa]
            P_b, Q_b, _ = scf_result.orbitals_raw[kappa][same_kappa.index(sh)]
            P_grid = eval_P(P_b, bsp_data).ravel()
            Q_grid = eval_Q(Q_b, bsp_data).ravel()
            rho_b_b = P_grid ** 2 + Q_grid ** 2
            # Interpolate the bound density onto our uniform grid
            sort_idx = np.argsort(gr_b)
            rho_on_h = np.interp(r_h, gr_b[sort_idx], rho_b_b[sort_idx])
            q_eff = q
            if exclude_label is not None and label == exclude_label:
                q_eff = max(q - 1.0, 0.0)
            N_eff += q_eff
            rho += q_eff * rho_on_h
    else:
        # Tabulated-orbital path (e.g. ShootingResult); orbital arrays
        # are already on a radial grid.  Loop and interpolate.
        # Skipped: not all backends carry the same grid metadata yet.
        # For now, fall back to bare nuclear if grid missing.
        return bare_nuclear_potential(Z)

    # Integrate the monopole Hartree potential
    #     V_H(r) = ∫_0^∞ ρ(r') / max(r, r') dr'
    #            = ∫_0^r ρ(r')/r dr'  +  ∫_r^∞ ρ(r')/r' dr'
    # using cumulative trapezoidal sums (vectorised over the grid).
    h = r_h[1] - r_h[0]
    # Inner: cumulative trapezoidal of ρ(r'), divided by r at each point
    inner_cum = np.cumsum(rho * h)
    inner_cum = np.maximum(inner_cum - 0.5 * rho * h, 0.0)  # trapezoid
    inner_term = inner_cum / r_h
    # Outer: ∫_r^∞ ρ(r')/r' dr' via reverse cumulative
    rho_over_r = rho / r_h
    outer_cum = np.cumsum(rho_over_r[::-1])[::-1] * h
    outer_term = outer_cum
    V_H = inner_term + outer_term
    V_total = -Z / r_h + V_H
    use_latter_tail = False
    if include_exchange:
        rho_vol = rho / (4.0 * np.pi * r_h ** 2)
        exchange_l = exchange.lower()
        if exchange_l == "fac":
            from atomx.physics.local_exchange import fac_gu_exchange
            V_x = fac_gu_exchange(
                rho_vol, r_h, Z=Z, N_total=N_eff,
                ihx=fac_ihx, hxs=fac_hxs, slater_alpha=alpha_x,
            )
            V_total = V_total + V_x
            use_latter_tail = True
            if fac_self_interaction_correction:
                from atomx.numerics.bspline.local_potential import _fac_csi_screening_correction

                screening = r_h * (V_H + V_x)
                corrected = _fac_csi_screening_correction(
                    r_h, screening, n_total=N_eff, ihx=fac_ihx, csi=fac_csi,
                )
                V_total = -Z / r_h + corrected / r_h
        elif exchange_l == "slater":
            from atomx.physics.local_exchange import slater_exchange
            V_total = V_total + slater_exchange(rho_vol, r_h, alpha=alpha_x)
        else:
            raise ValueError("exchange must be 'fac' or 'slater'")

    if latter_correction and use_latter_tail:
        rvinf = -(1.0 + Z - N_eff)
        V_total = np.where(r_h * V_total > rvinf, rvinf / r_h, V_total)

    # Wrap as an interpolating callable, with the Coulomb residual-core tail
    # outside the tabulated interval instead of a constant endpoint value.
    residual_charge = (1.0 + Z - N_eff) if (latter_correction and use_latter_tail) else (Z - N_eff)

    def V_func(r: np.ndarray) -> np.ndarray:
        r_arr = np.asarray(r, dtype=float)
        r_safe = np.where(r_arr > 0.0, r_arr, 1.0e-30)
        interp = np.interp(r_safe, r_h, V_total, left=V_total[0], right=np.nan)
        tail = -residual_charge / r_safe
        return np.where(np.isnan(interp), tail, interp)

    return V_func


__all__ = [
    "bare_nuclear_potential",
    "tabulated_potential",
    "static_hartree_potential",
]
