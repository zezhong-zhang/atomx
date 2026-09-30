"""Distorted-wave Born approximation for electron-impact ionisation.

Public entry point: :func:`dw_eii_cross_section`.

Implementation
--------------
- Default local Hartree field with the complete target density scaled by
  (N-1)/N and the solved nuclear field retained; an explicit potential on
  ContinuumModel overrides this choice.
- Exact-energy Dirac shooting, regular/irregular Coulomb matching and
  delta-energy normalization; dense power-series samples for radial integrals.
- J-coupled antisymmetric direct/exchange Slater amplitudes between the bound
  orbital and three continuum waves.
- A full-interval outgoing-energy quadrature and finite partial-wave sum.
  A geometric estimate of the incident partial-wave tail is opt-in.

The orchestration kernel is `_eii_dw_total` in this package. Full high-energy
partial-wave convergence, correlation and channel coupling are separate
accuracy requirements; matching one radial solution does not establish them.

References
----------
- M. F. Gu, *Can. J. Phys.* **86**, 675 (2008) — FAC EII algorithm.
- M. S. Pindzola, D. C. Griffin, *Phys. Rev. A* **36**, 2628 (1987)
  — frozen-core DW prescription.
"""

from __future__ import annotations

import numpy as np


def dw_eii_cross_section(
    scf_result,
    energies_eV: np.ndarray,
    *,
    kappa_max: int = 4,
    n_outgoing_energies: int = 16,
    continuum_model=None,
    continuum_failure_mode: str = "raise",
    partial_wave_tail: str = "none",
    radial_refinement: int = 1,
):
    """Distorted-wave Born EII cross section for all occupied subshells.

    Direct entry to the DW kernel (without going through the
    process-selection API). Numerical continuum controls and an explicit
    potential can be supplied through ``continuum_model``.

    Parameters
    ----------
    scf_result
        Solved SCFResult / MCDHFResult (must satisfy
        :class:`atomx.state_protocols.AtomicResultLike`).
    energies_eV : array_like
        Incident electron energies in eV.
    kappa_max : int
        Largest orbital angular momentum l of the projectile and outgoing
        electrons (default 4); includes both Dirac partners, through kappa=-5.
    n_outgoing_energies : int
        Number of E_1 quadrature nodes for the outgoing-electron
        energy share (default 16). Gauss-Legendre integration under
        E_1 = I*sinh(x)^2 covers the full interval [0, (E-I)/2].
    radial_refinement : int
        Positive integer subdivision of the Slater integration mesh (default
        1). Refine separately from continuum_model.n_grid and energy sharing.
    continuum_model : ContinuumModel, optional
        ``ContinuumModel.shooting(r_max=...)`` sets both the continuum
        propagation and collision-integration radius. It may exceed the
        bound-orbital box: the native bound expansion is zero outside that
        box, while continuum-continuum Coulomb sources span the full domain.
        Increasing the radius does not refine the bound orbitals themselves.
        Radial-domain convergence must be checked independently of sampling.
    partial_wave_tail : {"none", "geometric"}
        Optional approximate incident-tail completion. Default returns the
        computed partial-wave sum; cutoff convergence remains necessary.

    Returns
    -------
    EIIResult
        ``method = "DW"``.

    Notes
    -----
    The collision operator is instantaneous Coulomb, with direct and exchange
    amplitudes. Atomic Breit/QED corrections do not add transverse photon
    exchange to this collision operator.
    """
    # Lazy import to avoid the module-level circular dependency
    # between dw.py and __init__.py.
    from atomx.scattering.electron_ionization import _eii_dw_total
    return _eii_dw_total(
        scf_result, np.asarray(energies_eV, dtype=float),
        kappa_max=kappa_max,
        n_outgoing_energies=n_outgoing_energies,
        continuum_model=continuum_model,
        continuum_failure_mode=continuum_failure_mode,
        partial_wave_tail=partial_wave_tail,
        radial_refinement=radial_refinement,
    )


def bound_optical_strength(
    scf_result, orbital_label: str,
):
    """Bound-free optical (dipole) oscillator strength for an orbital.

    Convenience wrapper around the package-private
    ``_bound_optical_strength`` helper, available for optical-strength
    diagnostics. The DW EII kernel's optional geometric incident-wave
    tail does not call this optical-strength helper.

    Returns ``f_a`` in atomic units (dimensionless).
    """
    from atomx.numerics.bspline.convol import ConvolCache
    from atomx.scattering.electron_ionization import (
        _bound_optical_strength,
    )
    sh = next((s for s in scf_result.subshells if s.label == orbital_label),
                None)
    if sh is None:
        raise ValueError(
            f"Subshell {orbital_label!r} not found in scf_result"
        )
    eps_a = scf_result.orbital_energies[orbital_label]
    same_kappa = [s for s in scf_result.subshells if s.kappa == sh.kappa]
    P_a, Q_a, _ = scf_result.orbitals_raw[sh.kappa][same_kappa.index(sh)]
    cache = ConvolCache(scf_result._bsp_data)
    static_potentials: dict = {}
    return _bound_optical_strength(
        P_a, Q_a, sh.kappa, eps_a, scf_result, cache, static_potentials,
        occupation=float(sh.occupation),
    )


__all__ = ["dw_eii_cross_section", "bound_optical_strength"]
