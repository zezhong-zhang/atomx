"""Distorted-wave Born approximation for electron-impact ionisation.

Public entry point: :func:`dw_eii_cross_section`.

Implementation
--------------
- Frozen-core V^{N-1} static-Hartree distorted waves for projectile
  + outgoing electrons (Pindzola-Griffin / FAC prescription).
- Asymptotic free-particle box DOS √(R/(π·k)) for delta-energy
  normalisation of continuum waves.
- J-coupled antisymmetric Slater integrals between bound + 3
  continuum waves; partial-wave sum to ``kappa_max``.
- Coulomb-Bethe high-ℓ top-up for diffuse valence orbitals
  (⟨r⟩_rms ≥ 1 a₀).
- E1 quadrature endpoints floored at 0.5 eV to keep the asymptotic
  DOS factor √(2 ρ/π) ∝ 1/√k finite near threshold.

The actual numerical kernel lives in the package's ``__init__.py``
module as ``_eii_dw_total`` plus the helpers ``_bound_optical_strength``,
``_build_static_hartree``, and ``_orbital_rms_radius``.  This file
re-exposes them under more user-facing names so callers can import

    from atomx.scattering.electron_ionization.dw import dw_eii_cross_section

without grepping the 900-line orchestrator module.

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
    n_outgoing_energies: int = 6,
):
    """Distorted-wave Born EII cross section for all occupied subshells.

    Direct entry to the DW kernel (without going through the
    auto-mode hybrid that gates DW against BEB).  Useful when you
    want full DW results for benchmarking and don't mind the
    threshold-region pseudospectrum spikes that the hybrid
    suppresses.

    Parameters
    ----------
    scf_result
        Solved SCFResult / MCDHFResult (must satisfy
        :class:`atomx.scattering.protocols.AtomicResultLike`).
    energies_eV : array_like
        Incident electron energies in eV.
    kappa_max : int
        Largest |κ| in the partial-wave expansion of the projectile
        and outgoing electrons (default 4).
    n_outgoing_energies : int
        Number of E_1 quadrature nodes for the outgoing-electron
        energy share (default 6).

    Returns
    -------
    EIIResult
        ``method = "DW"``.
    """
    # Lazy import to avoid the module-level circular dependency
    # between dw.py and __init__.py.
    from atomx.scattering.electron_ionization import _eii_dw_total
    return _eii_dw_total(
        scf_result, np.asarray(energies_eV, dtype=float),
        kappa_max=kappa_max,
        n_outgoing_energies=n_outgoing_energies,
    )


def bound_optical_strength(
    scf_result, orbital_label: str,
):
    """Bound-free optical (dipole) oscillator strength for an orbital.

    Convenience wrapper around the package-private
    ``_bound_optical_strength`` helper, used by the Coulomb-Bethe
    high-ℓ top-up in DW EII and by external sanity checks of σ_PI.

    Returns ``f_a`` in atomic units (dimensionless).
    """
    from atomx.scattering.electron_ionization import (
        _bound_optical_strength, _build_static_hartree,
    )
    from atomx.numerics.bspline.convol import ConvolCache
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
