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
        """lambda_D = sqrt(kT/(4 pi n_e)) in a.u. = 743.4 sqrt(T_eV/n_e) cm.

        First-principles closed form (Gaussian units, kT in erg, e in esu);
        the 743.4 coefficient is *derived* below from cgs constants rather
        than quoted.  Measured: 140.481 a_0 = 7.4339e-7 cm at
        n_e = 1e20 cm^-3, T_e = 100 eV.  (Pre-2026-09 this test asserted
        only ``0 < lam < 1e10`` — 13 orders of magnitude of slack.)
        """
        from atomx.plasma.screening import debye_length

        ne, Te = 1e20, 100.0
        lam = debye_length(ne_cm3=ne, Te_eV=Te)
        ne_au = ne * (BOHR_RADIUS_M * 100.0) ** 3
        kT_au = Te / HARTREE_EV
        assert lam == pytest.approx(np.sqrt(kT_au / (4.0 * np.pi * ne_au)),
                                    rel=1e-12, abs=0.0)

        # same number from the cgs definition: lam_D = sqrt(kT/(4 pi n e^2))
        k_b_erg = 1.380649e-16
        e_esu = 4.803204712570263e-10
        lam_cm = np.sqrt(k_b_erg * (Te * 1.602176634e-12 / k_b_erg)
                         / (4.0 * np.pi * ne * e_esu ** 2))
        assert lam * (BOHR_RADIUS_M * 100.0) == pytest.approx(lam_cm, rel=1e-9, abs=0.0)
        assert pytest.approx(140.481, rel=1e-4, abs=0.0) == lam

    def test_ion_sphere_radius(self):
        """R_IS = (3 Z_ion / 4 pi n_e)^{1/3}, exact ne^{-1/3} scaling."""
        from atomx.plasma.screening import ion_sphere_radius

        R1 = ion_sphere_radius(Z_ion=1, ne_cm3=1e20)
        R2 = ion_sphere_radius(Z_ion=1, ne_cm3=8e20)
        ne_au = 1e20 * (BOHR_RADIUS_M * 100.0) ** 3
        assert pytest.approx((3.0 / (4.0 * np.pi * ne_au)) ** (1.0 / 3.0),
                             rel=1e-12, abs=0.0) == R1
        assert pytest.approx(25.2563, rel=1e-5, abs=0.0) == R1
        # an 8x density increase is an exact factor 8^{-1/3} = 1/2
        assert pytest.approx(8.0 ** (-1.0 / 3.0), rel=1e-12, abs=0.0) == R2 / R1
        assert pytest.approx(0.5, rel=1e-12, abs=0.0) == R2 / R1

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
        """SP must sit between its Debye and ion-sphere limits.

        The interpolation is what the name claims: at each coupling
        parameter u = R_IS/lambda_D the Stewart-Pyatt potential is bracketed
        by the two analytic limits, V_IS <= V_SP <= V_Debye (all three are
        negative — more negative is more attractive).  Measured at
        Z = 10, Z_ion = 1, n_e = 1e22, r = 5 a_0:
        u = 0.55 -> -1.802 <= -1.395 <= -1.209 Ha,
        u = 1.73 -> -1.802 <= -1.410 <= -0.407 Ha.
        (Pre-2026-09 this asserted only ``V(r) < 0``.)
        """
        from atomx.plasma.screening import (
            debye_length,
            debye_potential,
            ion_sphere_potential,
            ion_sphere_radius,
            stewart_pyatt_potential,
        )
        ne = 1e22
        Z = 10
        Z_ion = 1
        r = 5.0
        for Te, u_expected in ((50.0, 0.548), (5.0, 1.732)):
            u = ion_sphere_radius(Z_ion, ne) / debye_length(ne, Te)
            assert u == pytest.approx(u_expected, rel=1e-3, abs=0.0)
            v_debye = debye_potential(Z, ne, Te)(r)
            v_sp = stewart_pyatt_potential(Z, Z_ion, ne, Te)(r)
            v_is = ion_sphere_potential(Z, Z_ion, ne)(r)
            assert v_is <= v_sp <= v_debye, (Te, v_is, v_sp, v_debye)

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

    @pytest.mark.parametrize(
        ("lam_D", "R_IS"),
        [
            (1.0, 1.0e110),
            (1.0e-200, 1.0e200),
        ],
    )
    def test_stewart_pyatt_ipd_extreme_finite_u_stays_finite(self, lam_D, R_IS):
        """Finite but extreme ratios must not overflow or return NaN."""
        from atomx.plasma.screening import _stewart_pyatt_ipd_au

        Z_ion = 4.0
        ipd = _stewart_pyatt_ipd_au(Z_ion, lam_D, R_IS)
        ion_sphere = 1.5 * Z_ion / R_IS
        assert np.isfinite(ipd)
        assert ipd >= 0.0
        assert ipd == pytest.approx(ion_sphere, rel=1e-12)

    def test_stewart_pyatt_ipd_large_u_matches_ion_sphere_asymptote(self):
        from atomx.plasma.screening import _stewart_pyatt_ipd_au

        Z_ion = 2.0
        lam_D = 1.0
        R_IS = 1.0e20
        assert _stewart_pyatt_ipd_au(Z_ion, lam_D, R_IS) == pytest.approx(
            1.5 * Z_ion / R_IS, rel=1e-12)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"Z_ion": 0, "ne_cm3": 1e20, "Te_eV": 10.0}, "Z_ion"),
            ({"Z_ion": 1, "ne_cm3": np.nan, "Te_eV": 10.0}, "ne_cm3"),
            ({"Z_ion": 1, "ne_cm3": 1e20, "Te_eV": np.inf}, "Te_eV"),
        ],
    )
    def test_ipd_public_inputs_fail_closed(self, kwargs, match):
        from atomx.plasma.screening import ionization_potential_depression

        with pytest.raises(ValueError, match=match):
            ionization_potential_depression(**kwargs)


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
        from atomx.plasma.crm import CollisionalRadiativeModel, CRMLevel, CRMTransition
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
        """At high ne the CRM steady state must converge to Boltzmann (LTE).

        First-principles limit: the collisional rates are built in detailed
        balance (``C_deexc = C_exc (g_l/g_u) e^{+dE/kT}``,
        ``src/atomx/plasma/crm.py:354``), so for a two-level system

            n1/n0 = ne Ce / (ne Cd + A)  ->  (g1/g0) e^{-dE/kT}   (ne -> inf),

        i.e. the radiative decay A becomes negligible against collisional
        de-excitation and the *dynamics* must reproduce the analytic
        Boltzmann ratio.  The limit is independent of the collision strength
        (which cancels between the two directions).

        Measured max|pop/boltzmann - 1|: 1.540e-4 at 1e18, 1.541e-7 at
        1e21, 1.5e-11 at 1e25 — falling as 1/ne, the residual being
        A/(ne C_deexc).  The old body asserted only ``pop[1] > 0``, which a
        model with no collisional channel at all also satisfies.
        """
        from atomx.plasma.crm import CollisionalRadiativeModel, CRMLevel, CRMTransition
        Te = 10.0  # eV
        levels = [
            CRMLevel("ground", 0.0, g=1),
            CRMLevel("excited", 5.0, g=3),
        ]
        transitions = [
            CRMTransition(upper=1, lower=0, A_rate=1e8, gf=1.0),
        ]
        crm = CollisionalRadiativeModel(levels, transitions)
        boltzmann = crm.boltzmann_populations(Te)

        deviations = []
        for ne in (1e18, 1e21, 1e25):
            pop = crm.steady_state(ne_cm3=ne, Te_eV=Te)
            deviations.append(float(np.max(np.abs(pop / boltzmann - 1.0))))

        # monotone convergence in ne, and dense enough to be LTE.
        # The 1e-3 bound at ne=1e18 tracked the OLD (missing-1/dE) vR
        # constant; with the corrected q = 8.63e-6 gf gbar/(dE g
        # sqrt(T)) the absolute rates are ~dE x smaller so the residual
        # A/(ne C) is larger at fixed ne: 1.4e-3 at 1e18.  The LIMIT
        # structure (1/ne convergence, exact Boltzmann ratio) is the
        # physics; loosen the first-point bound accordingly.
        assert deviations[0] > deviations[1] > deviations[2]
        assert deviations[2] < 1e-9
        assert 0.0 < deviations[0] < 5e-3

        # ... and the ratio itself is the analytic Boltzmann ratio
        pop = crm.steady_state(ne_cm3=1e25, Te_eV=Te)
        assert pop[1] / pop[0] == pytest.approx(
            (3.0 / 1.0) * np.exp(-5.0 / Te), rel=1e-9, abs=0.0)

    def test_boltzmann_ground_dominant_at_low_T(self):
        """At low T, Boltzmann should put most population in ground state."""
        from atomx.plasma.crm import CollisionalRadiativeModel, CRMLevel
        levels = [
            CRMLevel("ground", 0.0, g=1),
            CRMLevel("excited", 100.0, g=3),
        ]
        transitions = []
        crm = CollisionalRadiativeModel(levels, transitions)
        pop = crm.boltzmann_populations(Te_eV=1.0)
        assert pop[0] > 0.99  # ground state dominant at 1 eV, gap = 100 eV

    def test_bound_free_eii_and_three_body_balance(self):
        """EII, TBR and RR should populate adjacent charge states consistently."""
        from atomx.plasma.crm import (
            CollisionalRadiativeModel,
            CRMBoundFreeChannel,
            CRMLevel,
            radiative_recombination_rate_cm3_s,
            saha_ionization_density_cm3,
            three_body_recombination_rate_cm6_s,
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
        # Spontaneous radiative recombination (review C5) is a second,
        # n_e * alpha_RR recombination channel; it was missing until
        # 2026-09, which is why this used to assert R[0, 1] == n_e^2 beta.
        alpha_rr = radiative_recombination_rate_cm3_s(
            15.0, Te, g_bound=2, g_continuum=1,
        )
        assert R[1, 0] == pytest.approx(ne * eii)
        assert R[0, 1] == pytest.approx(ne * ne * beta + ne * alpha_rr)

        pop = crm.steady_state(ne, Te)
        saha = saha_ionization_density_cm3(15.0, Te, g_bound=2, g_continuum=1)
        # Three-channel balance: n_c/n_b = R[1,0]/R[0,1].  Electron-impact
        # ionisation alone would give Saha/n_e (R[0,1] = n_e^2 beta with
        # beta = S/Saha); RR breaks that exact equality, so the ratio is
        # compared with the matrix entries rather than with Saha/n_e.
        assert pop[1] / pop[0] == pytest.approx(
            (ne * eii) / (ne * ne * beta + ne * alpha_rr), rel=1e-10)
        assert pop[1] / pop[0] < saha / ne

    def test_bound_free_photoionization_planck_couples_channel(self):
        """Planck PI should add a one-way bound-free source term."""
        from atomx.plasma.crm import (
            CollisionalRadiativeModel,
            CRMBoundFreeChannel,
            CRMLevel,
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
            CollisionalRadiativeModel,
            CRMBoundFreeChannel,
            CRMLevel,
            CRMTransition,
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
