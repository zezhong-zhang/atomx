"""Physics invariants covering the scattering/spectrum review repairs."""
from __future__ import annotations

from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import pytest
from scipy.special import k0e

from atomx import Atom, Basis
from atomx.constants import ALPHA, HARTREE_EV
from atomx.plasma.crm import CRMLevel, CRMTransition
from atomx.properties.transitions.two_photon import (
    two_photon_rate_h_like,
    two_photon_spectrum,
)
from atomx.scattering.distorted_wave import dw_excitation_cross_section
from atomx.scattering.electron_ionization import eii_cross_section
from atomx.scattering.msub import m_resolved_weights
from atomx.scattering.polarisation import (
    alignment_from_msublevel_cross_sections as alignment,
)
from atomx.scattering.polarisation import (
    linear_polarisation,
    polarised_emission_intensity,
)
from atomx.scattering.r_matrix import compute_multichannel, heplus_target_states
from atomx.scattering.r_matrix.cc_outer import (
    S_matrix_from_K,
    scattering_matrices_and_cross_sections,
)
from atomx.scattering.radiative_recombination import rr_cross_section
from atomx.spectrum import Spectrum, crm_emission_spectrum
from atomx.spectrum.continuum import free_free_emissivity, gaunt_ff_born


@pytest.fixture(scope="module")
def he_ground():
    atom = Atom(2, charge=1, config="1s1", corrections="none",
                engine="bspline", basis=Basis(ns=45))
    atom.solve()
    return atom


def _crm(rates):
    return NS(levels=[CRMLevel("lower0", 0), CRMLevel("lower1", 0),
                      CRMLevel("upper", 1239.8419843320026/100)],
              transitions=[CRMTransition(2, i, A) for i, A in enumerate(rates)],
              steady_state=lambda *a, **kw: np.array([0., 0., 1.]))


def _spectrum(crm, **kwargs):
    args = dict(lam_min_nm=99.9, lam_max_nm=100.1, n_points=2001, T_ion_K=1.)
    args.update(kwargs)
    return crm_emission_spectrum(crm, 1e10, 10., **args)


def test_polarization_is_positive_and_geometry_consistent():
    assert alignment([1, 2], 1) == pytest.approx(0, abs=1e-15)
    p0 = linear_polarisation(alignment([1, 0], 1), 1, 0)
    p1 = linear_polarisation(alignment([0, 1], 1), 1, 0)
    assert p0 == pytest.approx(1)
    assert p1 == pytest.approx(-1)
    assert polarised_emission_intensity(.5, np.pi/2) == pytest.approx((.75, .25))
    assert polarised_emission_intensity(.5, 0) == pytest.approx((.25, .25))
    for lower in (0, 1, 2):
        for populations in ([1., 0.], [0., 1.], [1., 2.]):
            assert abs(linear_polarisation(alignment(populations, 1), 1, lower)) <= 1+1e-12


def test_electron_impact_requires_collision_sublevel_data():
    with pytest.raises(ValueError, match="requires calculated"):
        m_resolved_weights(0, 1)
    assert m_resolved_weights(0, 1, geometry="photoabsorption") == pytest.approx({0.: 0., 1.: 1.})
    assert m_resolved_weights(0, 1, sublevel_weights={0.: 1., 1.: 2.}) == pytest.approx({0.: 1/3, 1.: 2/3})
    with pytest.raises(ValueError, match="Forbidden"):
        m_resolved_weights(0, 2, geometry="photoabsorption")


def test_natural_width_uses_total_lifetime_and_lorentzian_addition():
    calls = []
    with patch.object(Spectrum, "add_line", lambda self, center, power, **width: calls.append(width)):
        _spectrum(_crm([1e12, 9e12]), stark_hwhm_nm=.01)
    expected = 100.**2 / 2.99792458e17 * 1e13/(4*np.pi) + .01
    assert len(calls) == 2
    assert calls[0]["gamma_nm"] == pytest.approx(expected)
    assert calls[1]["gamma_nm"] == pytest.approx(expected)


def test_ci_width_sums_branches_across_configuration_pairs():
    upper = NS(config="up", index=0, two_J=2)

    def fake_lines(_source, upper_cfg, lower_cfg, **_kwargs):
        assert upper_cfg == "up"
        rate = {"low_a": 1e12, "low_b": 9e12}[lower_cfg]
        lower = NS(config=lower_cfg, index=0, two_J=0)
        line = NS(upper=upper, lower=lower, A_per_s=rate,
                  wavelength_angstrom=1000., omega_au=(1239.841984/100)/HARTREE_EV)
        return NS(levels_i=[upper], lines=[line])

    crm = NS(levels=[CRMLevel("low_a", 0, config="low_a"),
                     CRMLevel("low_b", 0, config="low_b"),
                     CRMLevel("up", 1239.841984/100, config="up")],
             transitions=[CRMTransition(2, 0, 1e12), CRMTransition(2, 1, 9e12)],
             steady_state=lambda *a, **kw: np.array([0., 0., 1.]))
    calls = []
    with (
        patch("atomx.properties.transitions.ci_lines.ci_line_strengths", fake_lines),
        patch.object(Spectrum, "add_line", lambda self, center, power, **width: calls.append(width)),
    ):
        _spectrum(crm, line_resolution="ci", ci_result=NS(subshells=[]),
                  stark_hwhm_nm=.01)
    expected = 100.**2 / 2.99792458e17 * 1e13/(4*np.pi) + .01
    assert len(calls) == 2
    assert all(width["gamma_nm"] == pytest.approx(expected) for width in calls)


def test_subgrid_profiles_conserve_area_without_changing_width():
    for n in (201, 2001):
        s = Spectrum.linear(99.9, 100.1, n)
        s.add_line(100.000037, 3., sigma_nm=1e-6)
        assert s.integrate() == pytest.approx(3., rel=1e-10)
        v = Spectrum.linear(99.9, 100.1, n)
        v.add_line(100.000037, 3., sigma_nm=1e-6, gamma_nm=1e-7)
        assert v.integrate() == pytest.approx(3., rel=3e-4)
    calls = []
    with patch.object(Spectrum, "add_line", lambda self, center, power, **width: calls.append(width)):
        _spectrum(_crm([1e8]), n_points=201)
        _spectrum(_crm([1e8]), n_points=2001)
    assert calls[0]["gamma_nm"] == pytest.approx(calls[1]["gamma_nm"])
    assert calls[0]["sigma_nm"] == pytest.approx(calls[1]["sigma_nm"])


def test_uta_delta_array_conserves_power_on_two_grids():
    from atomx.spectrum.uta import add_uta_array_to_spectrum
    for n in (101, 1001):
        s = Spectrum.linear(99., 101., n)
        add_uta_array_to_spectrum(s, [(1239.841984/100, 2.)],
                                  lam_min_nm=99., lam_max_nm=101.)
        assert s.integrate() == pytest.approx(2*1239.841984/100, rel=1e-10)


def test_free_free_tail_stays_finite():
    x = np.array([100., 1500., 2000.])
    g = gaunt_ff_born(x)
    np.testing.assert_allclose(g, np.sqrt(3)/np.pi*k0e(x/2), rtol=1e-15)
    assert np.all(np.isfinite(free_free_emissivity(x, 1., 1.)))


def test_rr_milne_relativistic_phase_space(he_ground):
    E = np.array([1e4, 1e5, 5e5])
    with patch("atomx.scattering.radiative_recombination.photoionization_cross_section",
               return_value=np.ones(3)):
        out = rr_cross_section(he_ground._result, E, g_initial=1, g_final=2,
                               multipoles=("E1",))
    e = E/HARTREE_EV
    binding = -he_ground._result.orbital_energies["1s1/2"]
    expected = 2*ALPHA**2*(e+binding)**2/(e*(2+ALPHA**2*e))
    np.testing.assert_allclose(out.sigma_total_a0_sq, expected, rtol=1e-12)


def test_eii_auto_continuous_at_model_switch(he_ground):
    eps = max(v for v in he_ground._result.orbital_energies.values() if v < 0)
    cutoff = -5*eps*HARTREE_EV
    E = cutoff*np.array([1-1e-6, 1+1e-6])
    out = eii_cross_section(he_ground._result, E)
    assert abs(out.sigma_total_cm2[1]/out.sigma_total_cm2[0]-1) < 1e-3
    assert out.provenance["blend"] == "C1 smoothstep DW/BEB"


def test_dw_uses_each_bound_orbitals_own_mesh(he_ground):
    excited = Atom(2, charge=1, config="2p1", corrections="none",
                   engine="bspline", basis=Basis(ns=30))
    excited.solve()
    sigma, omega = dw_excitation_cross_section(
        he_ground, excited, np.array([100.]), kappa_max=1, topup=False)
    assert np.all(np.isfinite(sigma)) and np.all(sigma >= 0)
    assert np.all(np.isfinite(omega)) and np.all(omega >= 0)


def test_corrected_r_matrix_observables_agree_with_k():
    targets = list(heplus_target_states(Z=2).values())
    out = compute_multichannel(targets, 0, 0, 1, np.array([-.8, -.78]),
                               n_basis=40, n_gauss=8, l_c_max=1,
                               include_l2_pseudostates=True)
    assert out.provenance["resonance_model"] == "empirical"
    for match in out.matchings:
        np.testing.assert_allclose(match.S_matrix, S_matrix_from_K(match.K_matrix), atol=1e-12)
        S, T, sigma, omega = scattering_matrices_and_cross_sections(
            out.channels, match.open_indices, match.k_diag, match.K_matrix,
            out.L_total, out.S_total)
        np.testing.assert_allclose(match.T_matrix, T, atol=1e-12)
        np.testing.assert_allclose(match.cross_section_matrix_a0, sigma, atol=1e-12)
        np.testing.assert_allclose(match.collision_strength_matrix, omega, atol=1e-12)


def test_two_photon_shape_has_no_central_node_and_correct_area():
    y = np.linspace(1e-4, 1-1e-4, 301)
    shape = two_photon_spectrum(y, 1)
    assert shape[150] > shape[100] > 0
    assert np.trapezoid(shape, y) == pytest.approx(2*two_photon_rate_h_like(1), rel=2e-3)


@pytest.fixture(scope="module")
def fe_ground():
    atom = Atom(26, charge=16, config="1s2 2s2 2p6", engine="shooting",
                exchange="gu", corrections="none")
    atom.solve()
    return atom


def test_ci_synthesis_uses_exact_line_set(fe_ground):
    crm = NS(levels=[CRMLevel("g", 0, 1, "2p6"),
                     CRMLevel("e", 820, 3, "2p5 3d1")],
             transitions=[CRMTransition(1, 0, 1e13)],
             steady_state=lambda *a, **kw: np.array([0., 1.]))
    kwargs = dict(lam_min_nm=1, lam_max_nm=2, n_points=2001, T_ion_K=1)
    spec = crm_emission_spectrum(crm, 1e13, 100, line_resolution="ci",
                                 ci_result=fe_ground, **kwargs)
    assert spec.line_count == 3
    assert spec.metadata["ci_groups"][0]["engine"] == "ci"
    assert spec.metadata["ci_groups"][0]["n_lines"] == 3
    with pytest.raises(ValueError, match="line_resolution"):
        crm_emission_spectrum(crm, 1e13, 100, line_resolution="invalid", **kwargs)
    with pytest.raises(ValueError, match="solved closed-shell"):
        crm_emission_spectrum(crm, 1e13, 100, line_resolution="ci", **kwargs)

    resolved_crm = NS(levels=[CRMLevel("g", 0), CRMLevel("e", 820)],
                      transitions=[CRMTransition(1, 0, 1e13)],
                      steady_state=lambda *a, **kw: np.array([0., 1.]))
    fallback = crm_emission_spectrum(resolved_crm, 1e13, 100,
                                     line_resolution="ci", ci_result=fe_ground,
                                     **kwargs)
    assert fallback.line_count == 1
    assert fallback.metadata["ci_groups"] == []
    assert fallback.metadata["ci_fallbacks"] == [{
        "transition_index": 0,
        "reason": "missing configuration labels; retained CRM transition",
    }]


def test_auger_continuum_failure_is_not_physical_zero():
    import runpy
    from pathlib import Path
    fixture = runpy.run_path(str(Path(__file__).with_name("test_auger_mscheme_ci.py")))["manifold_j72"].__wrapped__
    with (
        patch("atomx.scattering.autoionization.mscheme_ci._continuum_on_grid",
              side_effect=RuntimeError("injected failure")),
        pytest.raises(RuntimeError, match="Auger continuum failed"),
    ):
        fixture()
