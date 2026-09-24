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

from typing import Callable, Optional

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
    b.  Integrated using the bound solver's native radial quadrature.

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
        Number of logarithmic radial-grid points for tabulating screening (default 4000).
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
    from atomx.scattering.protocols import grid_data
    Z = float(scf_result.Z)

    # Determine outer radius
    bsp_data = grid_data(scf_result)
    if r_max is None:
        if bsp_data is not None and "t" in bsp_data:
            r_max = float(bsp_data["t"][-1])
        else:
            r_max = 50.0

    if n_radial < 16 or r_max <= 0:
        raise ValueError("potential tabulation needs n_radial >= 16 and r_max > 0")
    # Integrate on the bound solver's native mesh. A uniform mesh starting
    # at R/n misses high-Z core charge and flattens the nuclear singularity.
    N_eff = 0.0
    if bsp_data is not None and "bsp_p" in bsp_data:
        from atomx.numerics.bspline.evaluate import compute_yk, eval_P, eval_Q
        gr = bsp_data["gr"]
        density = np.zeros_like(gr)
        sampled = {}
        for sh in scf_result.subshells:
            key = (sh.n, sh.kappa)
            if hasattr(scf_result, "orbitals_dict"):
                p, q = scf_result.orbitals_dict[key]
            else:
                same = [s for s in scf_result.subshells if s.kappa == sh.kappa]
                p, q, _ = scf_result.orbitals_raw[sh.kappa][same.index(sh)]
            p, q = eval_P(p, bsp_data), eval_Q(q, bsp_data)
            sampled[key] = (p, q)
            population = max(0., sh.occupation - float(sh.label == exclude_label))
            density += population * (p*p + q*q)
            N_eff += population
        if hasattr(scf_result, "one_body_density_matrix"):
            keys, matrix = scf_result.one_body_density_matrix()
            for i, key in enumerate(keys):
                for j in range(i):
                    other = keys[j]
                    if key[1] == other[1]:
                        p, q = sampled[key]
                        u, v = sampled[other]
                        density += 2 * matrix[i, j] * (p*u + q*v)
        direct = compute_yk(density, gr, bsp_data["grw"],
                            bsp_data["nv"], gr.shape[1], 0).ravel()
        radius, density = gr.ravel(), density.ravel()
    elif getattr(scf_result, "_grid", None) is not None:
        from atomx.numerics.shooting.solver import _yk_over_r
        radius = scf_result._grid
        density = np.zeros_like(radius)
        for i, sh in enumerate(scf_result.subshells):
            population = max(0., sh.occupation - float(sh.label == exclude_label))
            density += population * (scf_result._pa[i]**2 + scf_result._qa[i]**2)
            N_eff += population
        direct = _yk_over_r(density, 0, radius, scf_result._difr, len(radius))
        positive = radius > 0
        radius, density, direct = radius[positive], density[positive], direct[positive]
    else:
        raise ValueError("static Hartree potential requires bound radial orbitals and their mesh")

    r_h = np.geomspace(min(radius[0], 1e-8 / max(Z, 1.)), r_max, n_radial)
    rho = np.interp(r_h, radius, density, left=0., right=0.)
    # Interpolate the screening charge, which tends to an exact constant,
    # rather than interpolating 1/r in the empty exterior region.
    screening = np.interp(r_h, radius, radius * direct, right=N_eff)
    V_H = np.where(r_h < radius[0], direct[0], screening / r_h)
    nuclear = getattr(scf_result, "_nuclear_potential", None)
    nuclear_value = (bare_nuclear_potential(Z) if nuclear is None else
                     np.vectorize(nuclear, otypes=[float]))
    correction = V_H.copy()
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
            correction = correction + V_x
            use_latter_tail = True
            if fac_self_interaction_correction and exclude_label is None:
                from atomx.numerics.bspline.local_potential import (
                    _fac_csi_screening_correction,
                )

                screening = r_h * (V_H + V_x)
                corrected = _fac_csi_screening_correction(
                    r_h, screening, n_total=N_eff, ihx=fac_ihx, csi=fac_csi,
                )
                correction = corrected / r_h
        elif exchange_l == "slater":
            from atomx.physics.local_exchange import slater_exchange
            correction = correction + slater_exchange(rho_vol, r_h, alpha=alpha_x)
        else:
            raise ValueError("exchange must be 'fac' or 'slater'")

    # A hole has already removed the electron's Hartree self-field.
    # Applying the bound-orbital N -> N-1 correction again gives the wrong
    # ionic charge. With no hole, preserve the explicit bound FAC model.
    self_hole = float(exclude_label is None and latter_correction and use_latter_tail)
    residual_charge = Z - N_eff + self_hole
    if latter_correction and use_latter_tail:
        correction = np.minimum(correction, (N_eff - self_hole) / r_h)

    def V_func(r: np.ndarray) -> np.ndarray:
        r_arr = np.asarray(r, dtype=float)
        r_safe = np.where(r_arr > 0., r_arr, 1e-30)
        charge = np.interp(r_safe, r_h, r_h * correction,
                           right=N_eff - self_hole)
        field = np.where(r_safe < r_h[0], correction[0], charge / r_safe)
        return nuclear_value(r_safe) + field

    V_func.residual_charge = residual_charge

    return V_func


__all__ = [
    "bare_nuclear_potential",
    "tabulated_potential",
    "static_hartree_potential",
]
