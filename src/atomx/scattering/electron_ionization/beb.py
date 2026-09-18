"""Binary-Encounter-Bethe / Binary-Encounter-Dipole (BEB / BED).

Kim & Rudd 1994 single-orbital semi-empirical EII formula.  Needs
only the orbital binding energy U, mean kinetic energy <T>, and an
optional dipole-Bethe parameter Q.  Accuracy ~ 30 % for valence
shells; widely used in NIFS / NIST EII databases.

References
----------
* Y.-K. Kim, M. E. Rudd, *Phys. Rev. A* **50**, 3954 (1994).
* Y.-K. Kim, K. K. Irikura, M. A. Ali, *J. Res. NIST* **105**, 285
  (2000) — BEB extension to molecules.
"""

from __future__ import annotations

import numba as nb
import numpy as np

from atomx.constants import BOHR_RADIUS_M, HARTREE_EV

# pi*a_0^2 in cm^2 (atomic-physics cross-section unit)
PI_A0_SQ_CM2 = np.pi * (BOHR_RADIUS_M * 100.0) ** 2  # ~ 8.797e-17 cm^2
RYD_EV = HARTREE_EV / 2.0


@nb.njit(cache=True)
def _beb_kernel(T: np.ndarray, U_eV: float, avgT_eV: float,
                N_electrons: float, n_principal: int, Q: float,
                PI_A0_SQ_CM2_val: float, RYD_EV_val: float) -> np.ndarray:
    """Numba-compiled BEB core computation."""
    out = np.zeros_like(T)
    S = 4.0 * PI_A0_SQ_CM2_val * N_electrons * (RYD_EV_val / U_eV) ** 2
    u = avgT_eV / U_eV
    denom_base = (u + 1.0) / n_principal
    for i in range(T.shape[0]):
        if T[i] <= U_eV:
            continue
        t = T[i] / U_eV
        log_t = np.log(t)
        bracket = (
            0.5 * Q * log_t * (1.0 - 1.0 / (t * t))
            + (2.0 - Q) * (1.0 - 1.0 / t - log_t / (t + 1.0))
        )
        val = S / (t + denom_base) * bracket
        if val < 0.0:
            val = 0.0
        out[i] = val
    return out


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

    sigma_BEB(T) = (S / (t + (u + 1)/n)) * [
                      (Q * ln t / 2) * (1 - 1/t^2)
                    + (2 - Q) * (1 - 1/t - ln t / (t + 1))
                  ]

    with ``t = T/U``, ``u = <T>/U``, ``S = 4pi a_0^2 N * (R/U)^2``,
    where ``R`` is the Rydberg energy and ``N`` the orbital
    occupation.  Below threshold (T < U) the cross section is zero.

    Parameters
    ----------
    T_eV
        Incident electron energy(s), eV.
    U_eV
        Orbital binding energy = -eps_a, eV.
    avgT_eV
        Mean kinetic energy of the bound orbital, eV.  Use
        |eps_a| (virial) if no explicit KE is available.
    N_electrons
        Orbital occupation (q_a = 2j_a+1 for a closed subshell).
    n_principal
        Principal quantum number n (denominator correction;
        Kim-Rudd recommend n=1 unless using BEB-vM).
    Q
        Dipole-Bethe parameter.  ``Q = 1`` is the standard BEB; set
        ``Q = 2 M_dipole^2`` if dipole oscillator-strength data are
        available (BED model).

    Returns
    -------
    np.ndarray
        Cross section in cm^2 on the same energy grid as ``T_eV``.
    """
    T = np.asarray(T_eV, dtype=np.float64)
    return _beb_kernel(T, float(U_eV), float(avgT_eV), float(N_electrons),
                       int(n_principal), float(Q),
                       float(PI_A0_SQ_CM2), float(RYD_EV))


__all__ = ["beb_cross_section", "PI_A0_SQ_CM2", "RYD_EV"]
