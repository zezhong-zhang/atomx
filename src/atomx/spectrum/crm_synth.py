"""CRM-weighted synthetic emission spectrum.

Given a :class:`~atomx.plasma.crm.CollisionalRadiativeModel` with rates
and populations, build a :class:`~atomx.spectrum.Spectrum` where each
radiative transition contributes a normalised line-profile weighted by
``n_upper · A_{ul}``.  Line widths are computed from Doppler (thermal)
and natural-lifetime contributions; an optional static Stark width for
hydrogenic ions can be folded in.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from atomx.constants import HARTREE_EV, HARTREE_INV_ANGSTROM
from atomx.lineshapes import doppler_width, natural_width
from atomx.spectrum.spectrum import Spectrum


def _hartree_to_angstrom(dE_hartree: float) -> float:
    """ΔE (Hartree) → λ (Å), standard 1/HARTREE_INV_ANGSTROM inversion."""
    return 1.0 / (HARTREE_INV_ANGSTROM * dE_hartree)


def crm_emission_spectrum(
    crm,
    n_e_cm3: float,
    T_e_eV: float,
    *,
    lam_min_nm: float,
    lam_max_nm: float,
    n_points: int = 2001,
    ion_mass_amu: float = 1.0,
    T_ion_K: Optional[float] = None,
    profile: str = "voigt",
    stark_hwhm_nm: Optional[float] = None,
) -> Spectrum:
    """Render a CRM's radiative transitions into a :class:`Spectrum`.

    Line intensities are the *emissivity-proportional* product
    ``ε_{ul} = n_upper · A_{ul} · (hν)``  — appropriate for most
    optical-plasma applications.  Multiply by a normalisation factor
    externally if you need absolute emissivity in W·m⁻³·sr⁻¹·nm⁻¹.

    Parameters
    ----------
    crm : CollisionalRadiativeModel
        A CRM with ``.levels`` and ``.transitions`` populated.
    n_e_cm3, T_e_eV : float
        Plasma electron density (cm⁻³) and temperature (eV).
    lam_min_nm, lam_max_nm : float
        Output wavelength range in nm.
    n_points : int
        Grid resolution.
    ion_mass_amu : float
        Atomic mass of the emitting ion (for Doppler broadening).
    T_ion_K : float, optional
        Ion kinetic temperature in K.  Defaults to
        ``T_e_eV × 11604.52``  (ion-electron equilibrium assumption).
    profile : {"gaussian", "lorentzian", "voigt"}
        Line profile family.  Voigt is the default when both Doppler
        and natural broadening are non-zero.
    stark_hwhm_nm : float, optional
        Additional HWHM added in quadrature to the Lorentzian width
        (set from :func:`atomx.lineshapes.stark_hwhm_quasistatic`
        for hydrogenic plasmas).

    Returns
    -------
    Spectrum
        Emission intensity on a linear nm grid.
    """
    # Compute populations (steady-state, user may have solved time-dependent
    # elsewhere — for emission spectrum the steady-state is the usual choice).
    populations = crm.steady_state(n_e_cm3, T_e_eV)

    if T_ion_K is None:
        T_ion_K = T_e_eV * 11604.5250061657   # eV → K

    spec = Spectrum.linear(
        lam_min_nm, lam_max_nm, n_points,
        unit="emissivity (arb., n_u · A · hν)",
    )
    spec.metadata = {
        "n_e_cm3": n_e_cm3,
        "T_e_eV": T_e_eV,
        "T_ion_K": T_ion_K,
        "ion_mass_amu": ion_mass_amu,
        "profile": profile,
    }

    for tr in crm.transitions:
        n_upper = populations[tr.upper]
        if n_upper <= 0 or tr.A_rate <= 0:
            continue

        dE_eV = crm.levels[tr.upper].energy_eV - crm.levels[tr.lower].energy_eV
        if dE_eV <= 0:
            continue

        lam_A = _hartree_to_angstrom(dE_eV / HARTREE_EV)
        lam_nm = lam_A / 10.0
        if lam_nm < lam_min_nm or lam_nm > lam_max_nm:
            continue

        # Doppler (thermal) σ in nm
        sigma_nm = doppler_width(lam_nm, T_ion_K, ion_mass_amu)

        # Natural width in nm:  Γ_λ (HWHM) = A · λ²/(4πc)·... units
        # Easier:  Γ_nu = A/(4π) [Hz], convert to Γ_λ via  Δλ = λ²/c · Δν
        # c in nm/s = 2.998e17
        gamma_nu_Hz = natural_width(tr.A_rate)   # in Hz (HWHM)
        c_nm_per_s = 2.99792458e17
        gamma_nm = (lam_nm ** 2 / c_nm_per_s) * gamma_nu_Hz

        if stark_hwhm_nm is not None:
            gamma_nm = np.hypot(gamma_nm, stark_hwhm_nm)

        # Intensity ~ n_upper · A · hν  — use photons·s⁻¹ weighted by energy
        intensity = n_upper * tr.A_rate * dE_eV

        if profile == "gaussian":
            spec.add_line(lam_nm, intensity, sigma_nm=sigma_nm)
        elif profile == "lorentzian":
            spec.add_line(lam_nm, intensity, gamma_nm=gamma_nm)
        elif profile == "voigt":
            spec.add_line(lam_nm, intensity,
                          sigma_nm=sigma_nm, gamma_nm=gamma_nm)
        else:
            raise ValueError(
                f"Unknown profile {profile!r}; use 'gaussian', "
                f"'lorentzian', or 'voigt'."
            )

    return spec
