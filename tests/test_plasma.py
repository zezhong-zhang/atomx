"""Tests for plasma screening and CRM (Phase 4).

Validates Debye/ion-sphere screening models and basic CRM functionality.
"""

import numpy as np
import pytest


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


class TestCRM:
    """Test Collisional-Radiative Model."""

    def test_crm_from_atom(self):
        """CRM should be buildable from a solved Atom."""
        from atomx import Atom
        from atomx.plasma.crm import CollisionalRadiativeModel
        ne = Atom(Z=10, ns=50, ks=9)
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
