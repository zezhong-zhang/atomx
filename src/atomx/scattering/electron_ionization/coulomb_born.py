"""Coulomb-Born / Bethe-Born high-energy ionisation cross sections.

Closed-form helpers covering the high-energy asymptote for electron-
impact ionisation, equivalent to FAC's ``SetCILCB`` Coulomb-Born
limit and ``SetCIPWOptions`` partial-wave-truncation regime.

Approach
--------
The Bethe-Born approximation (Bethe 1930) gives the leading high-E
behaviour of any dipole-allowed ionisation cross section:

    σ(E)   →   4 π a₀² · M_a² / E · [ ln(2 E / I_a) − C ]    (E ≫ I_a)

where M_a² = f_a / (2 I_a) is the orbital's dimensionless dipole
moment derived from the bound-free oscillator strength f_a, I_a is
the ionisation potential, and C ≈ 1 absorbs the inelastic-scattering
correction (Bethe 1930).  This is the asymptote that FAC's CB mode
falls back to once the partial-wave sum truncates.

Two flavours ship:

* :func:`bethe_born_ci_cross_section` — generic Bethe-Born tail
  evaluated from arbitrary `(M², I)` inputs.

* :func:`coulomb_born_ci_atom` — pulls per-orbital `(M², I)` from a
  solved AtomX SCF result (uses the existing
  :func:`atomx.scattering.electron_ionization._bound_optical_strength`
  helper) and returns σ_CB summed over occupied orbitals.

Reference
---------
* H. Bethe, *Ann. Phys.* **5**, 325 (1930).
* M. Inokuti, *Rev. Mod. Phys.* **43**, 297 (1971) — modern review.
* M. F. Gu, *Astrophys. J.* **579**, 1077 (2002) — FAC implementation
  (``SetCILCB`` knob in ``faclib/ionization.c``).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from atomx.constants import BOHR_RADIUS_M, HARTREE_EV

_A0_CM2 = (BOHR_RADIUS_M * 100.0) ** 2  # a₀² in cm²


def bethe_born_ci_cross_section(
    energies_eV: np.ndarray,
    M_sq: float,
    I_eV: float,
    *,
    bethe_C: float = 1.0,
) -> np.ndarray:
    """Bethe-Born high-energy ionisation cross section.

    σ(E) = (4 π a₀² · M² / E) · [ ln(2 E / I) − C ]   for E > I,
    σ(E) = 0                                          for E ≤ I.

    Parameters
    ----------
    energies_eV : array_like
        Incident electron energies in eV.
    M_sq : float
        Dimensionless squared dipole moment M² = f / (2 I) where f is
        the bound-free integrated oscillator strength of the orbital
        and I is its ionisation energy (both in atomic units).
    I_eV : float
        Ionisation potential of the orbital in eV (the ``I`` above
        in arbitrary energy units — the formula is unit-consistent
        as long as ``E`` and ``I`` share units).
    bethe_C : float
        Bethe inelastic constant; default 1.0 (textbook Bethe 1930).

    Returns
    -------
    np.ndarray
        σ_CB in cm² at each energy.
    """
    if M_sq < 0:
        raise ValueError("bethe_born_ci_cross_section: M_sq must be >= 0")
    if I_eV <= 0:
        raise ValueError("bethe_born_ci_cross_section: I_eV must be > 0")
    E = np.asarray(energies_eV, dtype=float)
    out = np.zeros_like(E)
    above = I_eV < E
    if not np.any(above):
        return out
    Ee = E[above]
    log_term = np.log(2.0 * Ee / I_eV) - bethe_C
    # 4π a₀² conversion to cm²; (M²/E) carries energy⁻¹, so multiply by
    # cm² and divide by E (eV) — the ratio is dimensionally a cross
    # section in cm² when M² is dimensionless and we use 4π·a₀²·HARTREE_EV
    # to put the prefactor in cm²·eV.  Specifically:
    #   σ[cm²] = 4π · a₀²[cm²] · (M²) · (HARTREE_EV / E[eV]) · log_term
    sigma = 4.0 * np.pi * _A0_CM2 * M_sq * (HARTREE_EV / Ee) * log_term
    sigma = np.where(sigma > 0, sigma, 0.0)  # log term may be < 0 just above I
    out[above] = sigma
    return out


def coulomb_born_ci_atom(
    scf_result,
    energies_eV: np.ndarray,
    *,
    bethe_C: float = 1.0,
    orbital_oscillator_strengths: Optional[Dict[str, float]] = None,
) -> Dict[str, np.ndarray]:
    """Coulomb-Born EII σ(E) summed over occupied orbitals of an atom.

    For each occupied subshell, computes M² = f_a / (2 I_a) using
    AtomX's existing :func:`_bound_optical_strength`, then evaluates
    the Bethe-Born asymptotic σ.  Useful as a high-E sanity check on
    the DW + BEB cross section returned by :func:`eii_cross_section`.

    Parameters
    ----------
    scf_result
        Solved SCFResult / MCDHFResult.
    energies_eV : array_like
        Incident electron energies in eV.
    bethe_C : float
        Bethe inelastic constant; default 1.0.
    orbital_oscillator_strengths : dict, optional
        Override `f_a` for any subshell label (e.g. when a precomputed
        oscillator strength is available).

    Returns
    -------
    dict
        ``{"sigma_total_cm2": np.ndarray, "by_orbital": {label: σ}}``.
    """
    from atomx.numerics.bspline.convol import ConvolCache
    from atomx.scattering.electron_ionization import (
        _bound_optical_strength,
    )

    bsp_data = scf_result._bsp_data
    cache = ConvolCache(bsp_data)
    static_potentials: Dict[str, tuple] = {}

    sigma_total = np.zeros(len(energies_eV))
    by_orbital: Dict[str, np.ndarray] = {}

    for sh in scf_result.subshells:
        eps = scf_result.orbital_energies.get(sh.label)
        if eps is None or eps >= 0:
            continue
        I_a_eV = -eps * HARTREE_EV

        # Orbital dipole strength
        if (orbital_oscillator_strengths is not None
                and sh.label in orbital_oscillator_strengths):
            f_a = orbital_oscillator_strengths[sh.label]
        else:
            orb_list = scf_result.orbitals_raw[sh.kappa]
            kappa_subs = [s for s in scf_result.subshells if s.kappa == sh.kappa]
            P, Q, _ = orb_list[kappa_subs.index(sh)]
            f_a = _bound_optical_strength(
                P, Q, sh.kappa, eps, scf_result, cache,
                static_potentials, occupation=float(sh.occupation),
            )

        M_sq = f_a / (2.0 * (-eps))
        sigma_orb = bethe_born_ci_cross_section(
            energies_eV, M_sq, I_a_eV, bethe_C=bethe_C,
        )
        # Multiply by the occupation (q_a electrons available)
        sigma_orb = sigma_orb * float(sh.occupation)
        by_orbital[sh.label] = sigma_orb
        sigma_total += sigma_orb

    return {"sigma_total_cm2": sigma_total, "by_orbital": by_orbital}


__all__ = [
    "bethe_born_ci_cross_section",
    "coulomb_born_ci_atom",
]
