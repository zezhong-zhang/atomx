"""Polarisation of line emission from electron-impact excitation.

When an atom is excited by a unidirectional electron beam, the
emitted line radiation is partially polarised because not all
magnetic substates ``M_J`` of the upper level are equally populated.
The degree of linear polarisation P (perpendicular to the beam
direction) is related to the alignment parameter A_2 of the upper
level:

    P = -3 A_2 / (h(2,J) − A_2)

with the level-anisotropy coefficient

    h(2, J) = 1 + (3/2)·δ_{J,1/2}·... ≃ 1   for J ≥ 1.

The alignment parameter A_2 is given by (Inal & Dubau 1989):

    A_2(J_u) = √(2/3) · Σ_{M_J} ⟨J_u M_J | T^2_0 | J_u M_J⟩
                                   · σ_{M_J}(J_u) / σ_total(J_u)

where σ_{M_J}(J_u) is the M-resolved excitation cross section.

This module provides:

- :func:`linear_polarisation` — convert alignment A_2 → P.
- :func:`alignment_from_msublevel_cross_sections` — compute A_2
  from a list of M-resolved cross sections.
- :func:`polarised_emission_intensity` — Stokes parameters for an
  observation geometry.

The M-sublevel cross sections themselves require an extension of
the distorted-wave CE solver (FAC ``CETableMSub``); a full
implementation is on the to-do list.

References
----------
- I. P. Grant, *Relativistic Quantum Theory of Atoms and Molecules*
  (Springer, 2007), §10.5.
- M. K. Inal, J. Dubau, *J. Phys. B* **20**, 4221 (1987); **22**,
  3329 (1989).
- D. L. Moores, M. S. Pindzola, *Phys. Rev. A* **41**, 3603 (1990).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from atomx.angular.wigner import wigner_6j


def _twice_j(value: float) -> int:
    return int(round(2.0 * value))


def _integer_phase(exponent: float) -> float:
    n = int(round(exponent))
    return -1.0 if n % 2 else 1.0


def steffen_alder_F2(J_u: float, J_l: float, L: int = 1) -> float:
    """Steffen-Alder ``F_2`` coefficient for a radiative multipole line.

    The coefficient converts upper-level alignment into emitted-line angular
    anisotropy/polarisation.  This implementation uses the standard 6j
    angular factor and the Steffen-Alder normalization for rank 2.  It returns
    zero for forbidden angular-momentum triangles and for upper levels that
    cannot support rank-2 alignment.
    """
    if L < 1:
        raise ValueError("Radiative multipole order L must be >= 1")
    if J_u < 1.0:
        return 0.0
    if abs(J_u - J_l) > L or J_u + J_l < L:
        return 0.0

    Ju2 = _twice_j(J_u)
    Jl2 = _twice_j(J_l)
    L2 = 2 * int(L)
    six = wigner_6j(Ju2, Ju2, 4, L2, L2, Jl2)
    if abs(six) < 1e-14:
        return 0.0
    norm = np.sqrt(3.0 * J_u * (2.0 * J_u + 1.0) / (J_u + 1.0))
    phase = _integer_phase(J_u + J_l + L)
    return float(phase * norm * six)


def linear_polarisation(alignment: float, J: float) -> float:
    """Linear polarisation P from alignment parameter A_2.

    P = -3 A_2 / (h(2, J) − A_2)

    where h(2, J) ≈ 1 for J ≥ 1; for J = 1/2 the polarisation
    vanishes by symmetry (no anisotropy possible).
    """
    if J <= 0.5 + 1e-9:
        return 0.0
    h2 = 1.0
    return -3.0 * alignment / (h2 - alignment)


def alignment_from_msublevel_cross_sections(
    sigma_M: Sequence[float],
    J: float,
) -> float:
    """Alignment parameter A_2 from M-sublevel cross sections.

    A_2 = Σ_M (3 M² − J(J+1)) σ_M / (J(J+1) Σ_M σ_M)
         · normalisation factor √(2/3)·…

    Inal-Dubau form:

    A_2(J) = √(5/(3 J (J+1) (2J−1)(2J+3))) · Σ_M (3M² − J(J+1)) σ_M / Σ_M σ_M

    Parameters
    ----------
    sigma_M
        Cross sections σ_{|M|} for ``|M| = 0, 1, …, J`` (length J+1
        for integer J, J+1/2 entries for half-integer J).  Each entry
        already accounts for ±M degeneracy.
    J
        Upper-level total angular momentum.

    Returns
    -------
    float
        Dimensionless alignment A_2.
    """
    sigma_M = np.asarray(sigma_M, dtype=float)
    if J <= 0.5 + 1e-9:
        return 0.0
    n_M = len(sigma_M)
    expected_n = int(round(J + 0.5)) if (2 * J) % 2 == 1 else int(J + 1)
    if n_M != expected_n:
        raise ValueError(
            f"sigma_M length {n_M} does not match J={J} "
            f"(expected {expected_n} entries for |M|=0…{J})."
        )

    # Build M values
    if (2 * J) % 2 == 0:  # integer J
        M_vals = np.arange(0, int(J) + 1)
        # Each entry counts ±M except M=0
        deg = np.where(M_vals == 0, 1.0, 2.0)
    else:  # half-integer J
        M_vals = np.arange(0, int(J + 1)) + 0.5
        deg = np.full_like(M_vals, 2.0)

    sigma_total = float(np.sum(sigma_M * deg))
    if sigma_total <= 0:
        return 0.0
    weighting = (3.0 * M_vals ** 2 - J * (J + 1.0))
    avg = float(np.sum(weighting * sigma_M * deg) / sigma_total)
    pref = np.sqrt(5.0 / (3.0 * J * (J + 1.0) *
                          (2.0 * J - 1.0) * (2.0 * J + 3.0)))
    return pref * avg


def polarised_emission_intensity(
    polarisation: float,
    observation_angle_rad: float,
) -> tuple[float, float]:
    """Intensity for parallel and perpendicular polariser orientations.

    For a transition with linear polarisation degree ``P`` at 90° to
    the beam, the angular-dependent intensity in directions parallel
    and perpendicular to the beam is

        I_||(θ) ∝ 1 − P · cos²θ
        I_⊥(θ) ∝ 1

    with θ the polar angle from beam axis.

    Returns
    -------
    (I_parallel, I_perpendicular)
        Relative intensities (normalised to perpendicular = 1).
    """
    cos2 = np.cos(observation_angle_rad) ** 2
    I_par = 1.0 - polarisation * cos2
    I_perp = 1.0
    return float(I_par), float(I_perp)


__all__ = [
    "steffen_alder_F2",
    "linear_polarisation",
    "alignment_from_msublevel_cross_sections",
    "polarised_emission_intensity",
]
