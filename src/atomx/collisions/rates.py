"""Rate coefficients from cross-sections.

Converts cross-sections to plasma-relevant rate coefficients by
integration over Maxwellian electron energy distributions.
"""

import numpy as np
from atomx.constants import HARTREE_EV, BOHR_RADIUS_M, BOLTZMANN_EV


def maxwellian_rate(cross_section_cm2, energies_eV, temperature_eV):
    """Maxwellian-averaged rate coefficient <sigma*v>.

    <sigma*v>(T) = sqrt(8/(pi*m_e)) * (1/kT)^{3/2}
                   * integral_0^inf sigma(E) * E * exp(-E/kT) dE

    Parameters
    ----------
    cross_section_cm2 : ndarray
        Cross-section in cm^2 at each energy.
    energies_eV : ndarray
        Electron energies in eV.
    temperature_eV : float
        Electron temperature in eV.

    Returns
    -------
    float
        Rate coefficient in cm^3/s.
    """
    kT = temperature_eV
    if kT <= 0:
        return 0.0

    # Boltzmann-weighted integrand: sigma(E) * E * exp(-E/kT)
    integrand = cross_section_cm2 * energies_eV * np.exp(-energies_eV / kT)

    # Trapezoidal integration
    integral = np.trapz(integrand, energies_eV)

    # Prefactor: sqrt(8/(pi*m_e)) * (1/kT)^{3/2}
    # In cgs: m_e = 9.109e-28 g, E in erg = eV * 1.602e-12
    m_e_cgs = 9.109_383_7015e-28  # g
    eV_to_erg = 1.602_176_634e-12
    kT_erg = kT * eV_to_erg

    prefactor = np.sqrt(8.0 / (np.pi * m_e_cgs)) * kT_erg ** (-1.5) * eV_to_erg
    return prefactor * integral


def effective_collision_rate(collision_strengths, energies_eV, temperature_eV,
                              g_i=1, threshold_eV=0.0):
    """Rate coefficient from collision strengths.

    q(T) = (8.629e-6 / (g_i * sqrt(T))) * integral Omega * exp(-E/kT) dE/kT

    Parameters
    ----------
    collision_strengths : ndarray
        Dimensionless collision strengths Omega(E).
    energies_eV : ndarray
        Electron energies above threshold in eV.
    temperature_eV : float
    g_i : int
        Statistical weight of initial state.
    threshold_eV : float
        Excitation threshold energy.

    Returns
    -------
    float
        Rate coefficient in cm^3/s.
    """
    kT = temperature_eV
    if kT <= 0:
        return 0.0

    # Upsilon: effective collision strength
    # Upsilon = integral Omega(E) * exp(-E/kT) d(E/kT)
    x = energies_eV / kT  # dimensionless energy
    integrand = collision_strengths * np.exp(-x)
    upsilon = np.trapz(integrand, x)

    # Rate: q = 8.629e-6 / (g_i * sqrt(T_eV)) * exp(-E_thresh/kT) * Upsilon
    rate = (8.629e-6 / (g_i * np.sqrt(kT))
            * np.exp(-threshold_eV / kT) * upsilon)
    return rate
