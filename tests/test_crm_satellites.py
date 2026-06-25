"""CRM dielectronic satellite line tests."""

import pytest

from atomx.constants import HARTREE_EV, HARTREE_INV_ANGSTROM
from atomx.plasma.crm import CRMLevel, CRMTransition, CollisionalRadiativeModel
from atomx.scattering.autoionization import AutoionizationState
from atomx.spectrum import crm_emission_spectrum


def _energy_from_angstrom(wavelength_A):
    return HARTREE_EV / (HARTREE_INV_ANGSTROM * wavelength_A)


def _fe17_base_crm():
    """Minimal Fe XVII CRM with a 15.01 A w-line."""
    e_w = _energy_from_angstrom(15.01)

    return CollisionalRadiativeModel(
        levels=[
            CRMLevel("Fe XVII ground", 0.0, g=1, config="2p6"),
            CRMLevel("Fe XVII 3C upper", e_w, g=3, config="2p5 3d"),
        ],
        transitions=[
            CRMTransition(
                upper=1,
                lower=0,
                A_rate=1.0e13,
                gf=1.0e-3,
            ),
        ],
        Z=26,
        charge=16,
    )


def _fe17_satellite_resonance():
    """Radiative-stabilising Fe XVII DR resonance near 15.26 A."""
    return AutoionizationState(
        label="Fe XVII KLL satellite",
        E_resonance_eV=_energy_from_angstrom(15.26),
        J=0.5,
        gamma_auger_per_s=4.0e16,
        a_radiative_per_s=4.0e16,
        channels={"Fe XVII ground": 4.0e16},
    )


def _fe17_fixture():
    """Minimal Fe XVII CRM with a 15.01 A w-line and one DR satellite."""
    crm = _fe17_base_crm()
    added = crm.add_dr_satellites([_fe17_satellite_resonance()], initial_level_idx=0)
    assert added == [2]
    return crm


def _satellite_to_w_ratio(spec):
    w_flux = spec.integrate(1.499, 1.503)
    satellite_flux = spec.integrate(1.522, 1.530)
    return satellite_flux / w_flux, w_flux, satellite_flux


@pytest.mark.parametrize("T_e_eV", [200.0, 400.0, 800.0])
def test_fe17_satellite_to_w_line_ratio(T_e_eV):
    """Fe XVII satellite / w-line emissivity ratio stays in textbook range.

    The expected integrated satellite-to-w ratio is approximately 0.05-0.20
    over this temperature range (Bely-Dubau et al. 1982; Doschek and
    Feldman 1978).  The acceptance band keeps the CRM pinned within a
    factor of two of that range.
    """
    crm = _fe17_fixture()

    spec = crm_emission_spectrum(
        crm,
        n_e_cm3=1.0e13,
        T_e_eV=T_e_eV,
        lam_min_nm=1.48,
        lam_max_nm=1.55,
        n_points=6001,
        ion_mass_amu=55.845,
        profile="gaussian",
    )

    ratio, w_flux, satellite_flux = _satellite_to_w_ratio(spec)

    assert spec.line_count == 2
    assert w_flux > 0.0
    assert satellite_flux > 0.0
    assert 0.025 < ratio < 0.40


def test_crm_synth_wires_dr_resonance_channels_without_mutating_input():
    crm = _fe17_base_crm()

    spec = crm_emission_spectrum(
        crm,
        n_e_cm3=1.0e13,
        T_e_eV=400.0,
        lam_min_nm=1.48,
        lam_max_nm=1.55,
        n_points=6001,
        ion_mass_amu=55.845,
        profile="gaussian",
        dr_resonances=[_fe17_satellite_resonance()],
        dr_initial_level_idx=0,
    )
    ratio, w_flux, satellite_flux = _satellite_to_w_ratio(spec)

    assert crm.n_levels == 2
    assert len(crm.transitions) == 1
    assert spec.metadata["dr_satellite_count"] == 1
    assert spec.line_count == 2
    assert w_flux > 0.0
    assert satellite_flux > 0.0
    assert 0.025 < ratio < 0.40
