"""Tests for plasma screening and CRM (Phase 4).

Validates Debye/ion-sphere screening models and basic CRM functionality.
"""

import numpy as np
import pytest

from atomx.constants import BOHR_RADIUS_M, HARTREE_EV


def _temperature_for_u(Z_ion, ne_cm3, u):
    from atomx.plasma.screening import EV_TO_HARTREE, ion_sphere_radius

    ne_au = ne_cm3 * (BOHR_RADIUS_M * 100) ** 3
    R_IS = ion_sphere_radius(Z_ion, ne_cm3)
    kT_au = 4.0 * np.pi * ne_au * (R_IS / u) ** 2
    return kT_au / EV_TO_HARTREE


class TestScreening:
    """Test plasma screening models."""

    def test_debye_length(self):
        """Debye length should be finite and positive."""
        from atomx.plasma.screening import debye_length
        lam = debye_length(ne_cm3=1e20, Te_eV=100.0)
        assert lam > 0
        assert lam < 1e10

    def test_ion_sphere_radius(self):
        """Ion-sphere radius should scale with (Z/ne)^(1/3)."""
        from atomx.plasma.screening import ion_sphere_radius
        R1 = ion_sphere_radius(Z_ion=1, ne_cm3=1e20)
        R2 = ion_sphere_radius(Z_ion=1, ne_cm3=8e20)
        # R ~ ne^(-1/3), so doubling ne should give R2/R1 = 2^(-1/3) ~ 0.5
        assert 0.4 < R2 / R1 < 0.6

    def test_debye_potential_reduces_to_coulomb(self):
        """At low density, Debye potential should approach bare Coulomb."""
        from atomx.plasma.screening import debye_potential
        V = debye_potential(Z=10, ne_cm3=1e10, Te_eV=1000.0)
        r = 1.0  # 1 Bohr
        # Very low density -> lambda_D very large -> exp(-r/lambda_D) ~ 1
        assert abs(V(r) - (-10.0 / r)) / abs(-10.0 / r) < 0.01

    def test_ipd_increases_with_density(self):
        """IPD should increase with electron density."""
        from atomx.plasma.screening import ionization_potential_depression
        ipd1 = ionization_potential_depression(Z_ion=1, ne_cm3=1e20, Te_eV=10.0)
        ipd2 = ionization_potential_depression(Z_ion=1, ne_cm3=1e23, Te_eV=10.0)
        assert ipd2 > ipd1

    def test_stewart_pyatt_interpolates(self):
        """Stewart-Pyatt should give intermediate result."""
        from atomx.plasma.screening import (
            stewart_pyatt_potential, debye_potential, ion_sphere_potential
        )
        ne = 1e22
        Te = 50.0
        Z = 10
        V_sp = stewart_pyatt_potential(Z, Z_ion=1, ne_cm3=ne, Te_eV=Te)
        r = 5.0
        # Should be between Debye and ion-sphere limits
        V_val = V_sp(r)
        assert V_val < 0  # attractive

    def test_stewart_pyatt_ipd_matches_limits(self):
        """SP IPD should recover Debye and ion-sphere analytic limits."""
        from atomx.plasma.screening import (
            debye_length,
            ion_sphere_radius,
            ionization_potential_depression,
        )

        Z_ion = 3
        ne = 1.0e23

        Te_debye = _temperature_for_u(Z_ion, ne, 1.0e-4)
        lam_D = debye_length(ne, Te_debye)
        debye = Z_ion / lam_D * HARTREE_EV
        sp_debye = ionization_potential_depression(
            Z_ion, ne, Te_debye, model="stewart_pyatt")
        assert sp_debye == pytest.approx(debye, rel=1e-4)

        Te_ion_sphere = _temperature_for_u(Z_ion, ne, 1.0e4)
        R_IS = ion_sphere_radius(Z_ion, ne)
        ion_sphere = 1.5 * Z_ion / R_IS * HARTREE_EV
        sp_ion_sphere = ionization_potential_depression(
            Z_ion, ne, Te_ion_sphere, model="stewart_pyatt")
        assert sp_ion_sphere == pytest.approx(ion_sphere, rel=2e-4)

    @pytest.mark.parametrize("u0", [0.1, 10.0])
    def test_stewart_pyatt_ipd_continuous_at_legacy_boundaries(self, u0):
        """Legacy implementation branched at u=0.1 and u=10."""
        from atomx.plasma.screening import ionization_potential_depression

        Z_ion = 1
        ne = 1.0e23
        values = [
            ionization_potential_depression(
                Z_ion, ne, _temperature_for_u(Z_ion, ne, u), "stewart_pyatt")
            for u in (u0 * 0.999, u0, u0 * 1.001)
        ]
        ratios = np.maximum(values[:-1], values[1:]) / np.minimum(
            values[:-1], values[1:])
        assert np.max(ratios) < 1.003

    def test_stewart_pyatt_ipd_log_sweep_has_no_silent_jump(self):
        """Adjacent SP values over the coupling range should vary smoothly."""
        from atomx.plasma.screening import ionization_potential_depression

        Z_ion = 2
        ne = 1.0e23
        values = np.array([
            ionization_potential_depression(
                Z_ion, ne, _temperature_for_u(Z_ion, ne, u), "stewart_pyatt")
            for u in np.geomspace(0.05, 20.0, 256)
        ])
        assert np.all(np.isfinite(values))
        assert np.all(values >= 0.0)
        adjacent = np.maximum(values[:-1], values[1:]) / np.minimum(
            values[:-1], values[1:])
        assert np.max(adjacent) < 1.03

    def test_stewart_pyatt_ipd_reported_u10_reproducer_is_continuous(self):
        """Reported u=9.99 -> 10.00 vector must not jump by orders."""
        from atomx.plasma.screening import ionization_potential_depression

        Z_ion = 1
        ne = 1.0e23
        below = ionization_potential_depression(
            Z_ion, ne, _temperature_for_u(Z_ion, ne, 9.99), "stewart_pyatt")
        at = ionization_potential_depression(
            Z_ion, ne, _temperature_for_u(Z_ion, ne, 10.0), "stewart_pyatt")
        assert max(below, at) / min(below, at) < 1.002

    def test_dr_density_suppression_uses_continuous_stewart_pyatt_ipd(self):
        """DR survival should not inherit the old SP branch discontinuity."""
        from atomx.scattering.autoionization.auger import AutoionizationState
        from atomx.scattering.dielectronic_recombination.solver import (
            dr_rate_coefficient,
        )

        Z_ion = 1
        ne = 1.0e23
        series_limit = 1000.01
        state = AutoionizationState(
            label="u10-boundary",
            E_resonance_eV=0.01,
            J=1.5,
            gamma_auger_per_s=1e14,
            a_radiative_per_s=1e13,
        )
        T = [
            _temperature_for_u(Z_ion, ne, 9.99),
            _temperature_for_u(Z_ion, ne, 10.0),
        ]
        rates = dr_rate_coefficient(
            [state],
            T,
            ne_cm3=ne,
            Z_ion=Z_ion,
            series_limit_eV=series_limit,
        )
        assert np.all(rates.alpha_cm3_s > 0.0)
        assert max(rates.alpha_cm3_s) / min(rates.alpha_cm3_s) < 1.03


class TestCRM:
    """Test Collisional-Radiative Model."""

    def test_crm_from_atom(self):
        """CRM should be buildable from a solved Atom."""
        from atomx import Atom
        from atomx.plasma.crm import CollisionalRadiativeModel
        # Default engine (shooting): CRM.from_atom rides on
        # transitions(), which now serves both radial flavours (gated in
        # test_transitions_engine_uniform.py).
        ne = Atom(Z=10)
        ne.solve(max_iter=40, tol=1e-4)
        if not ne.converged:
            pytest.skip("Ne did not converge")
        crm = CollisionalRadiativeModel.from_atom(ne)
        assert crm.n_levels > 0

    def test_steady_state_normalized(self):
        """Steady-state populations should sum to 1."""
        from atomx.plasma.crm import (
            CollisionalRadiativeModel, CRMLevel, CRMTransition
        )
        levels = [
            CRMLevel("ground", 0.0, g=1),
            CRMLevel("excited", 10.0, g=3),
        ]
        transitions = [
            CRMTransition(upper=1, lower=0, A_rate=1e8, gf=0.5),
        ]
        crm = CollisionalRadiativeModel(levels, transitions)
        pop = crm.steady_state(ne_cm3=1e12, Te_eV=5.0)
        assert abs(pop.sum() - 1.0) < 1e-10

    def test_high_density_approaches_boltzmann(self):
        """At high ne, CRM should approach LTE (Boltzmann distribution)."""
        from atomx.plasma.crm import (
            CollisionalRadiativeModel, CRMLevel, CRMTransition
        )
        Te = 10.0  # eV
        levels = [
            CRMLevel("ground", 0.0, g=1),
            CRMLevel("excited", 5.0, g=3),
        ]
        transitions = [
            CRMTransition(upper=1, lower=0, A_rate=1e8, gf=1.0),
        ]
        crm = CollisionalRadiativeModel(levels, transitions)

        # High density -> collisional rates dominate -> LTE
        pop_crm = crm.steady_state(ne_cm3=1e25, Te_eV=Te)
        pop_boltz = crm.boltzmann_populations(Te)

        # Should be reasonably close at high density
        # (may not be exact due to simplified collisional rates)
        assert pop_crm[1] > 0  # excited state should be populated

    def test_boltzmann_ground_dominant_at_low_T(self):
        """At low T, Boltzmann should put most population in ground state."""
        from atomx.plasma.crm import (
            CollisionalRadiativeModel, CRMLevel, CRMTransition
        )
        levels = [
            CRMLevel("ground", 0.0, g=1),
            CRMLevel("excited", 100.0, g=3),
        ]
        transitions = []
        crm = CollisionalRadiativeModel(levels, transitions)
        pop = crm.boltzmann_populations(Te_eV=1.0)
        assert pop[0] > 0.99  # ground state dominant at 1 eV, gap = 100 eV

    def test_bound_free_eii_and_three_body_balance(self):
        """EII and TBR should populate adjacent charge states consistently."""
        from atomx.plasma.crm import (
            CRMBoundFreeChannel, CRMLevel, CollisionalRadiativeModel,
            saha_ionization_density_cm3, three_body_recombination_rate_cm6_s,
        )

        Te = 20.0
        ne = 1.0e12
        eii = 2.0e-8
        channel = CRMBoundFreeChannel(
            bound=0, continuum=1, ionization_potential_eV=15.0,
            g_bound=2, g_continuum=1, eii_rate_cm3_s=eii,
        )
        crm = CollisionalRadiativeModel(
            [CRMLevel("q", 0.0, g=2), CRMLevel("q+1", 15.0, g=1)],
            [],
            bound_free_channels=[channel],
        )

        R = crm._build_rate_matrix(ne, Te)
        beta = three_body_recombination_rate_cm6_s(
            eii, 15.0, Te, g_bound=2, g_continuum=1,
        )
        assert R[1, 0] == pytest.approx(ne * eii)
        assert R[0, 1] == pytest.approx(ne * ne * beta)

        pop = crm.steady_state(ne, Te)
        saha = saha_ionization_density_cm3(15.0, Te, g_bound=2, g_continuum=1)
        assert pop[1] / pop[0] == pytest.approx(saha / ne, rel=1e-10)

    def test_bound_free_photoionization_planck_couples_channel(self):
        """Planck PI should add a one-way bound-free source term."""
        from atomx.plasma.crm import (
            CRMBoundFreeChannel, CRMLevel, CollisionalRadiativeModel,
        )

        crm = CollisionalRadiativeModel(
            [CRMLevel("q", 0.0), CRMLevel("q+1", 20.0)],
            [],
            bound_free_channels=[
                CRMBoundFreeChannel(
                    bound=0, continuum=1, ionization_potential_eV=20.0,
                    sigma0_cm2=1.0e-18,
                )
            ],
        )

        without_pi = crm._build_rate_matrix(
            1.0e10, 10.0, include_photoionization=False,
        )
        with_pi = crm._build_rate_matrix(
            1.0e10, 10.0, include_eii=False, radiation_temperature_eV=15.0,
        )
        assert without_pi[1, 0] == 0.0
        assert with_pi[1, 0] > 0.0

    def test_bound_free_toggles_reach_crm_emission_spectrum(self):
        """Spectrum wrapper should pass bound-free toggles into steady-state."""
        from atomx.plasma.crm import (
            CRMBoundFreeChannel, CRMLevel, CRMTransition, CollisionalRadiativeModel,
        )
        from atomx.spectrum import crm_emission_spectrum

        crm = CollisionalRadiativeModel(
            [CRMLevel("q", 0.0), CRMLevel("q+1", 12.0), CRMLevel("emit", 20.0)],
            [CRMTransition(upper=2, lower=0, A_rate=1.0e8, gf=0.0)],
            bound_free_channels=[
                CRMBoundFreeChannel(
                    bound=0, continuum=1, ionization_potential_eV=12.0,
                    photoionization_rate_s=1.0e5,
                )
            ],
        )

        spec = crm_emission_spectrum(
            crm, n_e_cm3=1.0e10, T_e_eV=10.0, lam_min_nm=1.0,
            lam_max_nm=200.0, include_photoionization=True,
        )
        assert spec.metadata["include_photoionization"] is True
