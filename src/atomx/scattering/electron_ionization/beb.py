"""Binary-Encounter-Bethe / Binary-Encounter-Dipole (BEB / BED).

Kim & Rudd 1994 single-orbital semi-empirical EII formula.  Needs
only the orbital binding energy U, mean kinetic energy ⟨T⟩, and an
optional dipole-Bethe parameter Q.  Accuracy ≈ 30 % for valence
shells; widely used in NIFS / NIST EII databases.

References
----------
* Y.-K. Kim, M. E. Rudd, *Phys. Rev. A* **50**, 3954 (1994).
* Y.-K. Kim, K. K. Irikura, M. A. Ali, *J. Res. NIST* **105**, 285
  (2000) — BEB extension to molecules.
"""

from __future__ import annotations

import numpy as np

from atomx.constants import HARTREE_EV, BOHR_RADIUS_M


# π·a₀² in cm² (atomic-physics cross-section unit)
PI_A0_SQ_CM2 = np.pi * (BOHR_RADIUS_M * 100.0) ** 2  # ≈ 8.797e-17 cm²
RYD_EV = HARTREE_EV / 2.0


def beb_cross_section(
    T_eV: np.ndarray,
    U_eV: float,
    avgT_eV: float,
    N_electrons: float,
    *,
    n_principal: int = 1,
    Q: float = 1.0,
) -> np.ndarray:
    """Binary-Encounter-Bethe (BEB) cross section for a single subshell.

    σ_BEB(T) = (S / (t + (u + 1)/n)) · [
                  (Q · ln t / 2) · (1 - 1/t²)
                + (2 - Q) · (1 - 1/t - ln t / (t + 1))
              ]

    with ``t = T/U``, ``u = ⟨T⟩/U``, ``S = 4π a₀² N · (R/U)²``,
    where ``R`` is the Rydberg energy and ``N`` the orbital
    occupation.  Below threshold (T < U) the cross section is zero.

    Parameters
    ----------
    T_eV
        Incident electron energy(s), eV.
    U_eV
        Orbital binding energy = -ε_a, eV.
    avgT_eV
        Mean kinetic energy of the bound orbital, eV.
    N_electrons
        Orbital occupation (q_a = 2j_a+1 for a closed subshell).
    n_principal
        Principal quantum number n (denominator correction;
        Kim-Rudd recommend n=1 unless using BEB-vM).
    Q
        Dipole-Bethe parameter.  ``Q = 1`` is the standard BEB; set
        ``Q = 2 M_dipole²`` if dipole oscillator-strength data are
        available (BED model).

    Returns
    -------
    np.ndarray
        Cross section in cm² on the same energy grid as ``T_eV``.
    """
    T = np.asarray(T_eV, dtype=float)
    out = np.zeros_like(T)
    above = T > U_eV
    if not np.any(above):
        return out

    t = T[above] / U_eV
    u = avgT_eV / U_eV
    S = 4.0 * PI_A0_SQ_CM2 * N_electrons * (RYD_EV / U_eV) ** 2

    log_t = np.log(t)
    bracket = (
        0.5 * Q * log_t * (1.0 - 1.0 / t ** 2)
        + (2.0 - Q) * (1.0 - 1.0 / t - log_t / (t + 1.0))
    )
    out[above] = S / (t + (u + 1.0) / n_principal) * bracket
    # Guard against tiny negative numerical noise just above threshold
    out = np.where(out < 0.0, 0.0, out)
    return out


__all__ = ["beb_cross_section", "PI_A0_SQ_CM2", "RYD_EV"]
