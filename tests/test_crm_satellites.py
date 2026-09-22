"""CRM dielectronic satellite line tests.

The satellite/w-line emissivity ratio has a closed form, so it is gated
against one instead of against a hard-coded band.  With

* the DR capture coefficient from the Burgess formula for one isolated
  resonance,
      alpha_DR(T) = P T^{-3/2} (g_d/2g_i) * G_A G_r/(G_A+G_r) e^{-E_res/T}
  with P = (h^2 / 2 pi m_e kT_eV)^{3/2} in cm^3 (the eV-convention closed
  form pinned in
  ``test_plasma_physics.py::test_dr_prefactor_matches_the_ev_convention_closed_form``,
  the Saha detailed-balance identity at
  ``::test_dr_prefactor_is_consistent_with_saha_detailed_balance``, and one
  resonance against the same closed form at
  ``::test_dr_rate_coefficient_uses_the_corrected_prefactor``);
* the two-level collisional-radiative steady state of the w line,
  ``n1/n0 = n_e C_exc / (n_e C_deexc + A_w)`` with the van Regemorter
  rates written out in this file;
* the single radiative exit of the satellite level,
  ``n_sat/n_0 = n_e alpha_DR / A_sat``;

the line-integrated flux ratio is the analytic product ratio

    ratio = n_e alpha_DR(T) E_sat / [ (n1/n0) A_w E_w ],

which the measured ratio reproduces to 1e-15 relative at 200/400/800 eV.

History: this file used to accept ``_RATIO_BAND = (0.025, 0.40) *
RYDBERG_EV**1.5 = (1.2546, 20.074)`` — the pre-fix band rescaled by exactly
the 50.186x DR-prefactor correction, so it was fitted to the defect it was
meant to catch and left the DR prefactor free to move by x0.478...x1.817
with all four tests still green (the historical /50.186 value failed).
"""

import numpy as np
import pytest
from scipy import constants as sc

from atomx.constants import HARTREE_EV, HARTREE_INV_ANGSTROM
from atomx.plasma.crm import CollisionalRadiativeModel, CRMLevel, CRMTransition
from atomx.scattering.autoionization import AutoionizationState
from atomx.scattering.dielectronic_recombination.solver import (
    _DR_PREFACTOR_CM3_S_EV_32,
)
from atomx.spectrum import crm_emission_spectrum

# Closed form of the Burgess prefactor, i.e. the Saha detailed-balance
# relation P = T^{3/2} * 2 e^{-chi/T} / Saha evaluated for
# Saha = 2/lambda_th^3 with lambda_th = h/sqrt(2 pi m_e kT):
#     P = (h^2 / (2 pi m_e kT_eV))^{3/2}   [cm^3, kT in eV]
# (= (h^2/(2 pi m_e e))^{3/2} with e the elementary charge J/eV, then
# m^3 -> cm^3).  Written out here rather than imported so that an edit to
# the module constant cannot move both sides of the gate below.
DR_PREFACTOR_CLOSED_CM3_S_EV_32 = (
    sc.h ** 2 / (2.0 * np.pi * sc.m_e * sc.e)
) ** 1.5 * 1.0e6

N_E_CM3 = 1.0e13
_G_BOUND = 1
_G_W = 3
_A_W = 1.0e13
_GF_W = 1.0e-3
_LAMBDA_W_A = 15.01
_LAMBDA_SAT_A = 15.26


def _energy_from_angstrom(wavelength_A):
    return HARTREE_EV / (HARTREE_INV_ANGSTROM * wavelength_A)


E_W_EV = _energy_from_angstrom(_LAMBDA_W_A)
E_SAT_EV = _energy_from_angstrom(_LAMBDA_SAT_A)


class _DRResultLike:
    def __init__(self, resonances):
        self.resonances = list(resonances)


def _fe17_base_crm():
    """Minimal Fe XVII CRM with a 15.01 A w-line."""
    return CollisionalRadiativeModel(
        levels=[
            CRMLevel("Fe XVII ground", 0.0, g=_G_BOUND, config="2p6"),
            CRMLevel("Fe XVII 3C upper", E_W_EV, g=_G_W, config="2p5 3d"),
        ],
        transitions=[
            CRMTransition(
                upper=1,
                lower=0,
                A_rate=_A_W,
                gf=_GF_W,
            ),
        ],
        Z=26,
        charge=16,
    )


def _fe17_satellite_resonance():
    """Radiative-stabilising Fe XVII DR resonance near 15.26 A."""
    return AutoionizationState(
        label="Fe XVII KLL satellite",
        E_resonance_eV=E_SAT_EV,
        J=0.5,
        gamma_auger_per_s=4.0e16,
        a_radiative_per_s=4.0e16,
        channels={"Fe XVII ground": 4.0e16},
    )


def _fe17_fixture():
    """Minimal Fe XVII CRM with a 15.01 A w-line and one DR satellite."""
    crm = _fe17_base_crm()
    resonance = _fe17_satellite_resonance()
    return crm, _DRResultLike([resonance])


def _satellite_to_w_ratio(spec):
    w_flux = spec.integrate(1.499, 1.503)
    satellite_flux = spec.integrate(1.522, 1.530)
    return satellite_flux / w_flux, w_flux, satellite_flux


def _dr_alpha_closed_form(resonance, Te_eV, g_initial):
    """Burgess closed form for one isolated resonance (cm^3/s)."""
    branch = (
        resonance.gamma_auger_per_s * resonance.a_radiative_per_s
        / (resonance.gamma_auger_per_s + resonance.a_radiative_per_s)
    )
    g_d = 2.0 * resonance.J + 1.0
    return (
        DR_PREFACTOR_CLOSED_CM3_S_EV_32
        * Te_eV ** -1.5
        * (g_d / (2.0 * g_initial))
        * branch
        * np.exp(-resonance.E_resonance_eV / Te_eV)
    )


def test_module_dr_prefactor_is_the_ev_convention_closed_form():
    """(h^2/2 pi m_e kT_eV)^{3/2} in cm^3 — measured 2.8e-16 relative.

    Gate mirror of
    ``test_plasma_physics.py::test_dr_prefactor_matches_the_ev_convention_closed_form``;
    duplicated here (at 1e-9, with ``abs=0``) so that the ratio gate below
    cannot be satisfied by a module that changed this constant.
    """
    assert pytest.approx(DR_PREFACTOR_CLOSED_CM3_S_EV_32,
                         rel=1e-9, abs=0.0) == _DR_PREFACTOR_CM3_S_EV_32
    assert pytest.approx(3.312830e-22, rel=1e-6, abs=0.0) == _DR_PREFACTOR_CM3_S_EV_32


def _satellite_to_w_ratio_closed_form(Te_eV, ne_cm3, resonance):
    """Analytic satellite/w flux ratio (see the module docstring)."""
    x = E_W_EV / Te_eV
    # van Regemorter effective Gaunt factor, large-x asymptote used by
    # atomx.plasma.crm for dE/kT > 1 (the same expression the underflow
    # test derives at test_plasma_physics.py:250).
    gaunt = max(0.2, 0.276 * (x + 1.0) / (x * (x + 2.0)))
    base = 1.578e-5 * _GF_W / (_G_BOUND * np.sqrt(Te_eV))
    c_exc = base * gaunt * np.exp(-x)
    # detailed balance: C_deexc = C_exc (g_l/g_u) exp(+x)
    c_deexc = base * gaunt * (_G_BOUND / _G_W)
    n1_n0 = ne_cm3 * c_exc / (ne_cm3 * c_deexc + _A_W)
    alpha_dr = _dr_alpha_closed_form(resonance, Te_eV, _G_BOUND)
    return ne_cm3 * alpha_dr * E_SAT_EV / (n1_n0 * _A_W * E_W_EV)


@pytest.mark.parametrize("T_e_eV", [200.0, 400.0, 800.0])
def test_fe17_satellite_to_w_line_ratio(T_e_eV):
    """Satellite/w ratio against the analytic branching form.

    Measured ratio (and the closed form it must reproduce) at
    n_e = 1e13 cm^-3: 11.047738 / 5.340118 / 2.625274 for 200/400/800 eV,
    matched to 1e-15 relative.  The gate is the DR prefactor: x2 of
    ``_DR_PREFACTOR_CM3_S_EV_32`` moves the ratio by exactly x2.
    """
    crm, dr_rate = _fe17_fixture()

    spec = crm_emission_spectrum(
        crm,
        n_e_cm3=N_E_CM3,
        T_e_eV=T_e_eV,
        lam_min_nm=1.48,
        lam_max_nm=1.55,
        n_points=6001,
        ion_mass_amu=55.845,
        profile="gaussian",
        dr_satellites=dr_rate,
        dr_initial_level_idx=0,
    )

    ratio, w_flux, satellite_flux = _satellite_to_w_ratio(spec)
    closed = _satellite_to_w_ratio_closed_form(
        T_e_eV, N_E_CM3, _fe17_satellite_resonance())

    assert spec.line_count == 2
    assert crm.n_levels == 2
    assert len(crm.transitions) == 1
    assert spec.metadata["dr_satellite_count"] == 1
    assert w_flux > 0.0
    assert satellite_flux > 0.0
    assert pytest.approx(closed, rel=1e-9, abs=0.0) == ratio


def test_crm_synth_wires_dr_resonance_channels_without_mutating_input():
    crm = _fe17_base_crm()

    spec = crm_emission_spectrum(
        crm,
        n_e_cm3=N_E_CM3,
        T_e_eV=400.0,
        lam_min_nm=1.48,
        lam_max_nm=1.55,
        n_points=6001,
        ion_mass_amu=55.845,
        profile="gaussian",
        dr_satellites=[_fe17_satellite_resonance()],
        dr_initial_level_idx=0,
    )
    ratio, w_flux, satellite_flux = _satellite_to_w_ratio(spec)
    closed = _satellite_to_w_ratio_closed_form(
        400.0, N_E_CM3, _fe17_satellite_resonance())

    # the caller's CRM is untouched (the satellite levels go into a copy)
    assert crm.n_levels == 2
    assert len(crm.transitions) == 1
    assert spec.metadata["dr_satellite_count"] == 1
    assert spec.line_count == 2
    assert w_flux > 0.0
    assert satellite_flux > 0.0
    assert pytest.approx(closed, rel=1e-9, abs=0.0) == ratio
