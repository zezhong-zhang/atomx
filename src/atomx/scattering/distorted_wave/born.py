"""Born-approximation electron-impact excitation cross section.

Closed-form helpers covering the high-energy plane-wave Born limit
of inelastic electron scattering, equivalent to FAC's ``SetCEBorn``
mode.  Replaces the more expensive distorted-wave partial-wave sum
when E ≫ ΔE — typically accurate to 10–20 % above 5 × threshold.

Approach
--------
The high-E Bethe-Born collision strength for a dipole-allowed
transition (a → b) is (Bethe 1930; van Regemorter 1962):

    Ω_dipole(E) ≈ (8 / √3) · g_a · f_ab · ⟨ḡ⟩(E/ΔE)

where f_ab is the absorption oscillator strength of the line,
ḡ ≈ 0.2–0.3 is the effective Gaunt factor, and g_a is the lower-state
weight 2 J_a + 1.  The cross section follows from the standard relation

    σ(E) = π / k_a²  ·  Ω(E) / g_a

with k_a² = 2 E in atomic units (non-relativistic).

For higher-multipole or forbidden lines, the Bethe limit reverts to
σ(E) = const·1/E with no log-term; we expose that via
:func:`born_excitation_forbidden`.

Reference
---------
* H. Bethe, *Ann. Phys.* **5**, 325 (1930).
* H. van Regemorter, *Astrophys. J.* **136**, 906 (1962).
* M. F. Gu, *Astrophys. J.* **579**, 1077 (2002) — FAC ``SetCEBorn``.
"""

from __future__ import annotations

import math

import numpy as np

from atomx.constants import HARTREE_EV, BOHR_RADIUS_M


_A0_CM2 = (BOHR_RADIUS_M * 100.0) ** 2  # a₀² in cm²


def born_excitation_dipole(
    energies_eV: np.ndarray,
    f_ab: float,
    delta_E_eV: float,
    *,
    g_lower: int = 1,
    gaunt_factor: float = 0.2,
) -> np.ndarray:
    """Plane-wave Born σ for a dipole-allowed E1 excitation.

    σ(E) = (8 π² · a₀² · f_ab / (√3 · ΔE)) · ḡ · ln(E / E_th)
                                        / (E / ΔE)             (E > E_th)

    with E_th = ΔE the threshold energy and ḡ the effective Gaunt
    factor (van Regemorter 1962 quotes ḡ ≈ 0.2 for ions, 0.3 for
    neutrals).

    Parameters
    ----------
    energies_eV : array_like
        Incident electron energies in eV.
    f_ab : float
        Absorption oscillator strength (dimensionless, computed from
        :func:`atomx.prop.transitions.rates.oscillator_strength`).
    delta_E_eV : float
        Excitation energy E_ab in eV.
    g_lower : int
        Statistical weight 2 J_a + 1 of the lower state. Default 1.
    gaunt_factor : float
        Effective ḡ factor; 0.2 for ions, 0.3 for neutrals.

    Returns
    -------
    σ_cm2 : np.ndarray
        Cross section in cm².
    """
    if f_ab < 0:
        raise ValueError("born_excitation_dipole: f_ab must be >= 0")
    if delta_E_eV <= 0:
        raise ValueError("born_excitation_dipole: ΔE must be > 0")
    E = np.asarray(energies_eV, dtype=float)
    out = np.zeros_like(E)
    above = E > delta_E_eV
    if not np.any(above):
        return out
    Ee = E[above]
    # van Regemorter ALN form: σ = 8π²·a₀² · f_ab · ḡ · ln(E/ΔE) / (√3 · E)
    sigma = (8.0 * math.pi ** 2 * _A0_CM2 * f_ab * gaunt_factor
             * np.log(Ee / delta_E_eV)
             / (math.sqrt(3.0) * (Ee / HARTREE_EV)))
    sigma = np.maximum(sigma, 0.0)
    out[above] = sigma
    return out


def born_excitation_forbidden(
    energies_eV: np.ndarray,
    line_strength_au: float,
    delta_E_eV: float,
    L: int = 2,
    *,
    g_lower: int = 1,
) -> np.ndarray:
    """Plane-wave Born σ for an L-pole forbidden excitation.

    σ(E) = (4 π · a₀² / E) · (S / (2 L + 1) · (ΔE)^(2L−2))   (very high E,
                                                              non-rel)

    More accurate at E ≫ ΔE for M1, E2, M2.  Falls off as 1/E without
    the log term that dipole-allowed transitions enjoy.

    Parameters
    ----------
    energies_eV : array_like
    line_strength_au : float
        Squared reduced matrix element S = |⟨f‖r^L C^L‖i⟩|² in atomic
        units (a₀^{2L}).
    delta_E_eV : float
    L : int
        Multipole rank ≥ 2 (E2 / M1 / M2 / E3 / …).
    g_lower : int
        Statistical weight of the lower state.

    Returns
    -------
    σ_cm2 : np.ndarray
    """
    if L < 1:
        raise ValueError("born_excitation_forbidden: L must be >= 1")
    if line_strength_au < 0:
        raise ValueError("line_strength_au must be >= 0")
    if delta_E_eV <= 0:
        raise ValueError("delta_E_eV must be > 0")
    E = np.asarray(energies_eV, dtype=float)
    out = np.zeros_like(E)
    above = E > delta_E_eV
    if not np.any(above):
        return out
    Ee_au = E[above] / HARTREE_EV
    dE_au = delta_E_eV / HARTREE_EV
    sigma_a0 = (4.0 * math.pi / Ee_au
                * (line_strength_au / (2 * L + 1))
                * dE_au ** (2 * L - 2))
    out[above] = sigma_a0 * _A0_CM2
    return out


__all__ = [
    "born_excitation_dipole",
    "born_excitation_forbidden",
]
