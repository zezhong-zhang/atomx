"""Rate coefficients from cross-sections.

Converts cross-sections to plasma-relevant rate coefficients by
integration over Maxwellian electron energy distributions.
"""

import numpy as np

from atomx.constants import BOLTZMANN_EV


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

    # Trapezoidal integration (np.trapezoid in numpy 2.x; np.trapz alias removed)
    _trap = getattr(np, "trapezoid", None) or np.trapz  # type: ignore[attr-defined]
    integral = _trap(integrand, energies_eV)

    # Prefactor: sqrt(8/(pi*m_e)) * (1/kT)^{3/2}
    # In cgs: m_e = 9.109e-28 g, E in erg = eV * 1.602e-12
    m_e_cgs = 9.109_383_7015e-28  # g
    eV_to_erg = 1.602_176_634e-12
    kT_erg = kT * eV_to_erg

    # eV_to_erg appears twice: one factor each for E in the integrand
    # and dE — the integral is built in eV but the prefactor is in
    # erg-consistent cgs.  (Previous version was missing one factor.)
    prefactor = np.sqrt(8.0 / (np.pi * m_e_cgs)) * kT_erg ** (-1.5) * eV_to_erg ** 2
    return prefactor * integral


def maxwellian_rate_direct(cross_section_cm2, threshold_eV, temperature_eV, *, order=16):
    """Nonrelativistic Maxwellian rate directly on threshold-shifted nodes.

    The callback consumes the complete incident-energy array in eV and returns
    matching nonnegative finite cross sections in cm². Gauss-Laguerre nodes
    sample the entire excess-energy half-line; no interpolated table or fitted
    tail is used. The caller must separately check quadrature order and the
    physical cross section's convergence/coverage. Return units are cm³/s.
    """
    from scipy.special import roots_laguerre

    from atomx.constants import CODATA, DEFAULT_CODATA_VERSION

    if (not callable(cross_section_cm2) or not np.isfinite(threshold_eV) or threshold_eV < 0
            or not np.isfinite(temperature_eV) or temperature_eV <= 0
            or isinstance(order, (bool, np.bool_)) or not isinstance(order, (int, np.integer))
            or not 2 <= order <= 128):
        raise ValueError('finite threshold>=0, temperature>0, callable cross section and order2..128 required')
    x, weights = roots_laguerre(int(order))
    energy = threshold_eV+temperature_eV*x
    sigma = np.asarray(cross_section_cm2(energy))
    if (sigma.shape != energy.shape or not np.isrealobj(sigma)
            or not np.isfinite(sigma).all() or np.any(sigma < 0)):
        raise ValueError('complete finite nonnegative cross-section array required')
    constants = CODATA[DEFAULT_CODATA_VERSION]
    factor = (100*np.sqrt(8*constants['_e']/(np.pi*constants['_me']))
              /np.sqrt(temperature_eV)*np.exp(-threshold_eV/temperature_eV))
    return float(factor*np.dot(weights, energy*sigma))


def effective_collision_rate(collision_strengths, energies_eV, temperature_eV,
                              g_i=1, threshold_eV=0.0):
    """Rate coefficient from collision strengths.

    q(T) = (8.629e-6 / (g_i * sqrt(T_K))) * exp(-dE/kT) * Upsilon(T),
    Upsilon = integral Omega(E_f) exp(-E_f/kT) d(E_f/kT),

    with the tabulated ``energies_eV`` being the FINAL (post-collision)
    electron energies above threshold, so the threshold enters only
    through ``exp(-threshold_eV/kT)``.

    The 8.629e-6 constant absorbs ``sqrt(8 k_B/(pi m_e)) pi a_0^2 Ry`` in
    cgs and is defined for T in KELVIN, not eV: the previous code passed
    ``sqrt(temperature_eV)``, which is a constant
    ``sqrt(1/BOLTZMANN_EV) = 107.72`` times too large at every
    temperature.

    Parameters
    ----------
    collision_strengths : ndarray
        Dimensionless collision strengths Omega(E_f).
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
    # Upsilon = integral Omega(E_f) * exp(-E_f/kT) d(E_f/kT)
    x = energies_eV / kT  # dimensionless energy
    integrand = collision_strengths * np.exp(-x)
    _trap = getattr(np, "trapezoid", None) or np.trapz  # type: ignore[attr-defined]
    upsilon = _trap(integrand, x)

    # Rate: q = 8.629e-6 / (g_i * sqrt(T_K)) * exp(-dE/kT) * Upsilon
    T_K = kT / BOLTZMANN_EV
    rate = (8.629e-6 / (g_i * np.sqrt(T_K))
            * np.exp(-threshold_eV / kT) * upsilon)
    return rate
