"""Closed-form analytic cross sections and rates for hydrogenic ions.

These are the FAC analogues of ``PICrossH``, ``RRCrossH``, and
``TRRateH`` — quick-reference formulas with no SCF / B-spline cost,
useful for sanity checks and for plasma-physics scoping where 5–10 %
accuracy is sufficient.

Public API
----------
:func:`stobbe_1s_pi_cross_section`
    Exact Stobbe (1930) hydrogenic 1s photoionisation σ_PI(ω).

:func:`kramers_pi_cross_section`
    Kramers (1923) semi-classical formula σ_PI ∝ Z⁴ / ω³, useful at
    ω ≫ I_n and all (n, l).

:func:`milne_rr_cross_section`
    Milne detailed-balance σ_RR(E_e) from a hydrogenic σ_PI at the
    same orbital.

:func:`hydrogenic_e1_rate`
    Closed-form A(n→n′; Z) for n,n′ ≤ 5 — the transitions that
    dominate Ly-α / He-α / Lyβ X-ray plasma diagnostics.

References
----------
* M. Stobbe, *Ann. Phys.* **399**, 661 (1930).
* H. A. Kramers, *Philos. Mag.* **46**, 836 (1923).
* H. A. Bethe & E. E. Salpeter, *Quantum Mechanics of One- and
  Two-Electron Atoms* (1957) — Eqs. (71.3) and (71.10).
* W. J. Karzas & R. Latter, *Astrophys. J. Suppl.* **6**, 167 (1961)
  — bound-free Gaunt factor tables.
* M. F. Gu, *Astrophys. J.* **582**, 1241 (2003) — FAC implementation.
"""

from __future__ import annotations

import math

import numpy as np

from atomx.constants import ALPHA, BOHR_RADIUS_M


# σ_K *prefactor* for the Stobbe formula (without the exp(−4) Coulomb
# enhancement that emerges from the formula itself at threshold):
#     A = σ_K^thr / e⁻⁴   →  6.3043 × 10⁻¹⁸ × e⁴ = 3.4422 × 10⁻¹⁶ cm²
# At threshold η → ∞, arccot(η) → 1/η and exp(−4 η · 1/η) = e⁻⁴, so
# σ_K^thr(Z = 1) = 6.3043 × 10⁻¹⁸ cm² as quoted in NIST tables.
_STOBBE_PREFACTOR_H_CM2 = 6.3043e-18 * math.exp(4.0)

# Bohr radius squared in 10⁻²⁰ cm² units (a₀² ≈ 2800 in those units).
_A0_SQ_CM2_X1E20 = (BOHR_RADIUS_M * 100.0) ** 2 * 1e20


def stobbe_1s_pi_cross_section(
    Z: int,
    photon_energies_au: np.ndarray,
) -> np.ndarray:
    """Exact hydrogenic 1s photoionisation cross section.

    Stobbe-Heitler closed form
        σ(ω) = σ_K^thr(Z) · (I/ω)⁴
                · exp(−4 η arccot η) / (1 − exp(−2π η))
    where I = Z²/2 (a.u.), η = Z/k = √(I/(ω−I)), and
    σ_K^thr(Z) = 6.30·10⁻¹⁸ cm² / Z² is the Z=∞ limit at threshold.

    Parameters
    ----------
    Z : int
        Nuclear charge (≥ 1).
    photon_energies_au : array_like
        Photon energies in atomic units (Hartree).  Below threshold
        ω < I the routine returns 0.

    Returns
    -------
    σ_cm2 : np.ndarray
        Cross section in cm².
    """
    if Z < 1:
        raise ValueError("stobbe_1s_pi_cross_section: Z must be >= 1")
    omega = np.asarray(photon_energies_au, dtype=float)
    I = 0.5 * Z * Z                       # threshold in Hartree
    out = np.zeros_like(omega)
    above = omega > I
    if not np.any(above):
        return out
    o = omega[above]
    k = np.sqrt(2.0 * (o - I))           # photoelectron k
    eta = Z / k                          # Coulomb parameter
    # Stobbe formula
    arc = np.arctan2(1.0, eta)            # arccot(η) = atan(1/η)
    expo = np.exp(-4.0 * eta * arc)
    denom = 1.0 - np.exp(-2.0 * math.pi * eta)
    sigma = (_STOBBE_PREFACTOR_H_CM2 / (Z * Z)) * (I / o) ** 4 * expo / denom
    out[above] = sigma
    return out


def kramers_pi_cross_section(
    Z: int, n: int,
    photon_energies_au: np.ndarray,
) -> np.ndarray:
    """Kramers semi-classical photoionisation cross section.

    Valid for ω ≫ I_n at any (n, l):

        σ(ω) = 64π α a₀² Z⁴ / (3 √3 · n⁵ · (ω/I_n)³)        [Kramers 1923]

    Off by ~25 % from the exact Stobbe value at threshold but
    asymptotically correct.  Useful as a quick high-energy estimate.

    Parameters
    ----------
    Z, n : int
    photon_energies_au : array_like

    Returns
    -------
    σ_cm2 : np.ndarray
        Cross section in cm².
    """
    if Z < 1 or n < 1:
        raise ValueError("kramers_pi_cross_section: Z, n must be >= 1")
    omega = np.asarray(photon_energies_au, dtype=float)
    I = 0.5 * Z * Z / (n * n)
    out = np.zeros_like(omega)
    above = omega > I
    o = omega[above]
    sigma_a0 = (64.0 * math.pi * ALPHA / (3.0 * math.sqrt(3.0) * n ** 5)
                * (Z ** 4) / (o / I) ** 3)
    out[above] = sigma_a0 * (BOHR_RADIUS_M * 100.0) ** 2
    return out


def milne_rr_cross_section(
    Z: int, n: int,
    electron_energies_au: np.ndarray,
    *,
    g_recombined: float = 2.0,
    g_recombining: float = 1.0,
) -> np.ndarray:
    """Milne detailed-balance σ_RR for hydrogenic capture into (n, *).

    Built on the exact Stobbe σ_PI for n=1, the Kramers approximation
    for n ≥ 2.  Sums over all l for each n.

    σ_RR(E_e) = σ_PI(ω) · (g_b / (2 g_a)) · α² ω² / (2 E_e)
        with ω = E_e + I_n.

    Parameters
    ----------
    Z, n : int
    electron_energies_au : array_like
        Free-electron kinetic energies in Hartree.
    g_recombined : float
        Statistical weight 2J_b+1 of the recombined ion.  Default 2 for
        the j=1/2 single-electron state.
    g_recombining : float
        Statistical weight 2J_a+1 of the recombining ion.  Default 1
        for a closed-shell or bare nucleus.

    Returns
    -------
    σ_cm2 : np.ndarray
    """
    if Z < 1 or n < 1:
        raise ValueError("milne_rr_cross_section: Z, n must be >= 1")
    Ee = np.asarray(electron_energies_au, dtype=float)
    if np.any(Ee <= 0):
        raise ValueError("milne_rr_cross_section: E_e must be positive")
    I = 0.5 * Z * Z / (n * n)
    omega = Ee + I
    if n == 1:
        sigma_pi_cm2 = stobbe_1s_pi_cross_section(Z, omega)
    else:
        sigma_pi_cm2 = kramers_pi_cross_section(Z, n, omega)
    milne = (ALPHA ** 2) * (omega ** 2) / (2.0 * Ee)
    return sigma_pi_cm2 * (g_recombined / (2.0 * g_recombining)) * milne


# -----------------------------------------------------------------
#  Hydrogenic E1 rates: closed forms for the lowest n→n' transitions
# -----------------------------------------------------------------

# Exact NR Lyman-α 2p → 1s rate at Z = 1 (Bethe & Salpeter 1957)
_LYMAN_ALPHA_2P_1S = 6.2649e8   # s⁻¹

# Hydrogenic E1 rate-constant table for low-n transitions, A(Z=1, ...)
# in s⁻¹.  Z⁴-scaling holds in the NR limit; relativistic corrections
# are O((αZ)²) and negligible for Z ≤ 30 at this precision.
_HYDROGENIC_E1_TABLE = {
    # (n_upper, n_lower, l_upper -> l_lower):  A(Z=1) in s⁻¹
    (2, 1, 1, 0): 6.2649e8,    # Ly-α   (2p → 1s)
    (3, 1, 1, 0): 1.6725e8,    # Ly-β   (3p → 1s)
    (3, 2, 0, 1): 6.3133e6,    # H-α s component (3s → 2p)
    (3, 2, 2, 1): 6.4651e7,    # H-α d component (3d → 2p)
    (4, 1, 1, 0): 6.8186e7,    # Ly-γ   (4p → 1s)
    (4, 2, 1, 0): 0.0,         # forbidden by parity within same n
    (4, 2, 0, 1): 2.5773e6,    # 4s → 2p
    (4, 2, 2, 1): 2.0625e7,    # 4d → 2p
    (4, 3, 1, 0): 0.0,
    (4, 3, 1, 2): 1.3781e7,    # 4p → 3d
    (4, 3, 0, 1): 1.8323e6,    # 4s → 3p
    (4, 3, 2, 1): 7.0426e6,    # 4d → 3p
    (4, 3, 3, 2): 1.3781e7,    # 4f → 3d
}


def hydrogenic_e1_rate(
    Z: int,
    n_upper: int, n_lower: int,
    l_upper: int, l_lower: int,
) -> float:
    """Hydrogenic E1 rate A(n_u, l_u → n_l, l_l) at given Z.

    Returns A in s⁻¹.  Implements the ``TRRateH`` table in FAC
    using closed-form values from Bethe & Salpeter Eq. 70.10 with the
    Z⁴ scaling.  Relativistic corrections are not included; the rate
    is accurate to (αZ)² for Z ≪ 137.

    Raises
    ------
    KeyError
        If the (n_u, n_l, l_u, l_l) combination is not in the
        tabulated set.
    """
    key = (n_upper, n_lower, l_upper, l_lower)
    A_z1 = _HYDROGENIC_E1_TABLE.get(key)
    if A_z1 is None:
        raise KeyError(
            f"hydrogenic_e1_rate: no table entry for "
            f"(n_u={n_upper}, n_l={n_lower}, l_u={l_upper}, l_l={l_lower}); "
            f"available: {sorted(_HYDROGENIC_E1_TABLE)}"
        )
    return A_z1 * (Z ** 4)


__all__ = [
    "stobbe_1s_pi_cross_section",
    "kramers_pi_cross_section",
    "milne_rr_cross_section",
    "hydrogenic_e1_rate",
]
