"""Continuum emission channels for spectral synthesis.

Free-free (bremsstrahlung) and free-bound (radiative-recombination
continuum) emissivities on a photon-energy grid, for folding into
:func:`atomx.spectrum.crm_synth.crm_emission_spectrum` alongside the line
spectrum.  Line-to-continuum ratios are standard hot-plasma diagnostics;
without these channels they cannot be formed at all.

Free-free
---------
Maxwell-averaged thermal bremsstrahlung on a hydrogenic ion of effective
charge ``Z_eff`` (Rybicki & Lightman eq. 5.14b, CGS):

    eps_nu = C_ff Z^2 n_e n_i T_K^(-1/2) exp(-h nu / k T) <g_ff>
    C_ff   = 2^5 pi e^6 / (3 m c^3) * sqrt(2 pi / (3 k m))
           = 6.842e-38 erg s^-1 cm^-3 Hz^-1   (verified from CODATA)

The thermally averaged Gaunt factor uses the exact closed form of the
Born (Bethe-Heitler) cross section averaged over a Maxwellian,

    <g_ff>(x) = (sqrt(3)/pi) e^(x/2) K_0(x/2),      x = h nu / k T,

(K_0 the modified Bessel function; e.g. Novikov & Thorne 1973).  Born is
accurate for k T >> Z^2 Ry and moderate x; for cold/high-Z plasmas an
Elwert-corrected or tabulated (van Hoof) Gaunt factor is the documented
upgrade path — the hook is the ``gaunt`` callable.

Free-bound
----------
Delegates to the first-principles Milne-relation RR machinery
(:func:`atomx.scattering.radiative_recombination.rrc_spectrum`, photons
cm^3 s^-1 eV^-1 per ion) and converts to the same energy-emissivity
units as the line spectrum.
"""

from __future__ import annotations

import numpy as np
from scipy.special import k0e

from atomx.constants import BOLTZMANN_EV

# 2^5 pi e^6 / (3 m_e c^3) sqrt(2 pi / (3 k_B m_e)) in CGS
# (erg s^-1 cm^-3 Hz^-1 per n_e n_i Z^2 / sqrt(T_K)); evaluated from
# CODATA constants, matches the textbook 6.8e-38 value.
_C_FF_ERG_CM3_HZ = 6.8419e-38
_ERG_PER_EV = 1.602176634e-12
_H_EV_S = 4.135667696e-15  # Planck constant, eV s


def gaunt_ff_born(x: np.ndarray) -> np.ndarray:
    """Maxwell-averaged free-free Gaunt factor, Born approximation.

    <g_ff>(x) = (sqrt(3)/pi) e^(x/2) K_0(x/2), x = h nu / k T.  Exact
    closed form of the thermally averaged Bethe-Heitler cross section.
    -> (sqrt(3)/pi) ln(4/(1.781 x)) as x -> 0; -> sqrt(3/(pi x)) as
    x -> inf.
    """
    x = np.asarray(x, dtype=float)
    if np.any(~np.isfinite(x)) or np.any(x <= 0):
        raise ValueError("photon energy / temperature must be finite and positive")
    out = np.sqrt(3.0) / np.pi * k0e(0.5 * x)
    return out


def free_free_emissivity(
    photon_energies_eV: np.ndarray,
    T_e_eV: float,
    Z_eff: float,
    *,
    n_e_cm3: float = 1.0,
    n_i_cm3: float = 1.0,
    gaunt=None,
) -> np.ndarray:
    """Thermal bremsstrahlung emissivity, eV s^-1 cm^-3 eV^-1.

    Energy radiated per unit volume, per unit photon energy, in eV/s
    units (divide by nothing to compare with line emissivities built as
    ``n_upper A_ul (h nu)`` with n in cm^-3 and h nu in eV).

    ``gaunt(x)`` may override the Maxwell-averaged Gaunt factor
    (default: Born closed form :func:`gaunt_ff_born`).
    """
    E = np.asarray(photon_energies_eV, dtype=float)
    T = float(T_e_eV)
    if T <= 0.0:
        raise ValueError("T_e_eV must be > 0")
    x = E / T
    eps_nu_erg = _free_free_prefactor(E, T, Z_eff, n_e_cm3, n_i_cm3, gaunt) * np.exp(-x)
    # per Hz -> per eV of photon energy; erg -> eV
    return eps_nu_erg / _H_EV_S / _ERG_PER_EV


def _free_free_prefactor(E_eV, T_eV, charge, ne_cm3, ni_cm3, gaunt):
    """Shared per-Hz CGS coefficient before the thermal exponential."""
    if T_eV <= 0 or not np.isfinite(T_eV):
        raise ValueError("T_e_eV must be finite and positive")
    g = (gaunt or gaunt_ff_born)(np.asarray(E_eV) / T_eV)
    return (_C_FF_ERG_CM3_HZ * charge**2 * ne_cm3 * ni_cm3
            / np.sqrt(T_eV / BOLTZMANN_EV) * g)


def free_free_absorption(photon_energies_eV, T_e_eV, Z_eff, *, n_e_cm3=1.0,
                         n_i_cm3=1.0, gaunt=None):
    """LTE net free-free absorption in cm^-1, including stimulated emission.

    Reuses the emission coefficient and cancels the Boltzmann exponential
    analytically, so high-energy absorption survives emissivity underflow.
    The default Born Gaunt factor has the same high-Z/low-T limitations as
    free_free_emissivity; callers can supply a tabulated Gaunt provider.
    """
    E = np.asarray(photon_energies_eV, dtype=float)
    if np.any(~np.isfinite(E)) or np.any(E <= 0):
        raise ValueError("positive finite photon energies required")
    eps0 = _free_free_prefactor(E, T_e_eV, Z_eff, n_e_cm3, n_i_cm3, gaunt)
    nu = E / _H_EV_S
    h_erg_s, c_cm_s = _H_EV_S * _ERG_PER_EV, 2.99792458e10
    return eps0 * -np.expm1(-E / T_e_eV) / (8 * np.pi * h_erg_s * nu**3 / c_cm_s**2)


def rr_continuum_emissivity(
    scf_result,
    photon_energies_eV: np.ndarray,
    T_e_eV: float,
    *,
    n_e_cm3: float = 1.0,
    n_i_cm3: float = 1.0,
    **rr_kwargs,
) -> np.ndarray:
    """Radiative-recombination continuum, eV s^-1 cm^-3 eV^-1.

    ``rrc_spectrum`` returns the photon emissivity coefficient
    (photons cm^3 s^-1 eV^-1 per recombining-ion, per electron);
    multiply by densities and photon energy for the energy emissivity.
    """
    from atomx.scattering.radiative_recombination import rrc_spectrum

    E = np.asarray(photon_energies_eV, dtype=float)
    coeff = rrc_spectrum(scf_result, T_e_eV, E, **rr_kwargs)
    return coeff * n_e_cm3 * n_i_cm3 * E


__all__ = [
    "free_free_emissivity",
    "free_free_absorption",
    "rr_continuum_emissivity",
    "gaunt_ff_born",
]
