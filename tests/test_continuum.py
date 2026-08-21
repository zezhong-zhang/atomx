"""Tests for continuum Dirac solver and photoionization cross-sections."""

import numpy as np
import pytest
from atomx.constants import ALPHA, BOHR_RADIUS_M
from atomx.constants import C_SQ
from atomx.methods.hydrogenic_cross_sections import stobbe_1s_pi_cross_section
from atomx.physics import Dirac, Schrodinger
from atomx.numerics.bspline.knots import make_knots
from atomx.numerics.bspline.basis import prepare_bspline_data
from atomx.numerics.bspline.dirac_assembly import build_dirac_matrices
from atomx.numerics.bspline.dirac_assembly import solve_dirac
from atomx.numerics.bspline.continuum import (
    continuum_energy_normalization_factor,
    solve_continuum,
    dipole_matrix_element,
    photoionization_cross_section,
)


BOHR_CM2 = (BOHR_RADIUS_M * 100.0) ** 2


class TestContinuumWaveNumber:
    def test_schrodinger_nonrelativistic(self):
        assert Schrodinger().continuum_wave_number(0.5) == pytest.approx(1.0)

    def test_dirac_relativistic_dispersion(self):
        E = 100.0
        k = Dirac().continuum_wave_number(E)
        assert k == pytest.approx(np.sqrt(2.0 * E * (1.0 + 0.5 * E / C_SQ)))
        # (ck)^2 = E (E + 2c^2), the exact relativistic dispersion
        assert C_SQ * k * k == pytest.approx(E * (E + 2.0 * C_SQ))

    def test_dirac_reduces_to_schrodinger_at_low_energy(self):
        E = 1e-4
        k_rel = Dirac().continuum_wave_number(E)
        k_nr = Schrodinger().continuum_wave_number(E)
        assert k_rel == pytest.approx(k_nr, rel=1e-8)
        assert k_rel > k_nr  # relativistic k is always the larger one

    def test_rejects_nonpositive_energy(self):
        for eq in (Dirac(), Schrodinger()):
            with pytest.raises(ValueError):
                eq.continuum_wave_number(0.0)
            with pytest.raises(ValueError):
                eq.continuum_wave_number(-1.0)


class TestContinuumSolver:
    def test_positive_energy(self):
        """Continuum solution should have energy close to target."""
        ns, ks = 40, 9
        t = make_knots(ns, ks, 0.01, 50.0)
        data = prepare_bspline_data(t, ns, ks)
        P, Q, E = solve_continuum(kappa=-1, energy=1.0, Z=1, bsp_data=data)
        # Should find a state near E=1.0 in the discretized continuum
        assert E > 0
        assert abs(E - 1.0) < 0.5  # within discretization spacing

    def test_continuum_nonzero(self):
        """Continuum wavefunction should be nonzero."""
        ns, ks = 40, 9
        t = make_knots(ns, ks, 0.01, 50.0)
        data = prepare_bspline_data(t, ns, ks)
        P, Q, E = solve_continuum(kappa=-1, energy=0.5, Z=1, bsp_data=data)
        assert np.max(np.abs(P)) > 1e-5

    def test_energy_normalized_scales_box_state(self):
        ns, ks = 40, 9
        t = make_knots(ns, ks, 0.01, 50.0)
        data = prepare_bspline_data(t, ns, ks)
        P_box, Q_box, E_box = solve_continuum(
            kappa=-1, energy=0.5, Z=1, bsp_data=data)
        P_en, Q_en, E_en = solve_continuum(
            kappa=-1, energy=0.5, Z=1, bsp_data=data,
            energy_normalized=True)

        factor = continuum_energy_normalization_factor(E_box, data)
        assert E_en == pytest.approx(E_box)
        np.testing.assert_allclose(P_en, factor * P_box)
        np.testing.assert_allclose(Q_en, factor * Q_box)

    def test_normalization_uses_relativistic_k_for_dirac(self):
        ns, ks = 40, 9
        t = make_knots(ns, ks, 0.01, 50.0)
        data = prepare_bspline_data(t, ns, ks)
        E = 500.0  # deep continuum, where the dispersions visibly differ

        f_default = continuum_energy_normalization_factor(E, data)
        f_dirac = continuum_energy_normalization_factor(E, data, Dirac())
        f_schrod = continuum_energy_normalization_factor(E, data, Schrodinger())

        assert f_default == pytest.approx(f_dirac)
        # factor ~ 1/sqrt(k): relativistic k > sqrt(2E) -> smaller factor
        k_ratio = Dirac().continuum_wave_number(E) / np.sqrt(2.0 * E)
        assert f_dirac == pytest.approx(f_schrod / np.sqrt(k_ratio))
        assert f_dirac < f_schrod


class TestDipoleMatrixElement:
    def test_nonzero_for_allowed(self):
        """Dipole element between 1s (kappa=-1) and p continuum should be nonzero."""
        ns, ks = 40, 9
        t = make_knots(ns, ks, 0.01, 50.0)
        data = prepare_bspline_data(t, ns, ks)

        # Get 1s bound state
        H, S = build_dirac_matrices(ns, -1, 1, data)
        result = solve_dirac(H, S, ns, -1)
        P_1s, Q_1s, E_1s = result.get_orbital(0)

        # Get p-wave continuum
        P_cont, Q_cont, E_cont = solve_continuum(kappa=1, energy=0.5, Z=1, bsp_data=data)

        dme = dipole_matrix_element(P_1s, Q_1s, P_cont, Q_cont, data)
        assert abs(dme) > 1e-5


class TestPhotoionization:
    @staticmethod
    def _hydrogen_1s_case(ns, r_box, photon_factor=1.5):
        ks = 9
        t = make_knots(ns, ks, 0.01, r_box)
        data = prepare_bspline_data(t, ns, ks)

        H, S = build_dirac_matrices(ns, -1, 1, data)
        result = solve_dirac(H, S, ns, -1)
        P_1s, Q_1s, E_1s = result.get_orbital(0)
        photon = np.array([-E_1s * photon_factor])
        sigma = photoionization_cross_section(
            P_1s, Q_1s, -1, E_1s, photon, Z=1, bsp_data=data)
        return data, P_1s, Q_1s, E_1s, photon, sigma

    def test_hydrogen_cross_section(self):
        """H 1s photoionization cross-section should be positive above threshold."""
        ns, ks = 40, 9
        t = make_knots(ns, ks, 0.01, 50.0)
        data = prepare_bspline_data(t, ns, ks)

        # Get 1s bound state
        H, S = build_dirac_matrices(ns, -1, 1, data)
        result = solve_dirac(H, S, ns, -1)
        P_1s, Q_1s, E_1s = result.get_orbital(0)

        # Photon energies: from just above threshold to 2× threshold
        threshold = -E_1s  # ~0.5 Hartree
        photon_energies = np.array([threshold * 1.1, threshold * 1.5, threshold * 2.0])

        sigma = photoionization_cross_section(
            P_1s, Q_1s, -1, E_1s, photon_energies, Z=1, bsp_data=data)

        # Cross-section should be positive above threshold
        assert np.all(sigma > 0), f"sigma={sigma}"

    def test_hydrogen_cross_section_box_invariant_with_energy_normalization(self):
        """The public PI path must not consume raw box-normalized states.

        Before HYP-1530, the same H 1s photon point fell by about 35 % when
        increasing the B-spline box from 100 a0 to 150 a0 because the finite-box
        pseudo-state was inserted directly into the delta-energy PI formula.
        """
        cases = [
            self._hydrogen_1s_case(80, 100.0),
            self._hydrogen_1s_case(120, 150.0),
        ]
        sigmas = np.array([case[-1][0] for case in cases])

        np.testing.assert_allclose(sigmas[0], sigmas[1], rtol=0.08)

    def test_hydrogen_cross_section_stobbe_scale_and_threshold(self):
        """Hydrogen 1s PI keeps the Stobbe absolute scale and threshold."""
        data, P_1s, Q_1s, E_1s, photon, sigma = self._hydrogen_1s_case(
            120, 150.0)
        stobbe_a0_sq = stobbe_1s_pi_cross_section(1, photon) / BOHR_CM2

        # This B-spline path is a finite-box Dirac-Coulomb approximation, not
        # the closed-form non-relativistic Stobbe solver.  The order-unity
        # agreement pins the absolute normalization convention while the
        # separate box-invariance test pins the missing density-of-states factor.
        assert sigma[0] / stobbe_a0_sq[0] == pytest.approx(4.5, rel=0.2)

        below = photoionization_cross_section(
            P_1s,
            Q_1s,
            -1,
            E_1s,
            np.array([-E_1s * 0.99]),
            Z=1,
            bsp_data=data,
        )
        assert below[0] == 0.0

    def test_hydrogen_1s_e1_channel_sum_nonzero_for_both_p_waves(self):
        data, P_1s, Q_1s, E_1s, photon, sigma_total = self._hydrogen_1s_case(
            80, 100.0)
        electron_energy = float(photon[0] + E_1s)

        from atomx.properties.transitions.angular import line_strength_factor

        partials = []
        for kf in (1, -2):
            angular_factor = line_strength_factor(-1, kf, 1, "E")
            Pf, Qf, _ = solve_continuum(
                kf, electron_energy, 1, data, energy_normalized=True)
            dme = dipole_matrix_element(P_1s, Q_1s, Pf, Qf, data)
            partials.append(
                4.0 * np.pi ** 2 * ALPHA * float(photon[0]) / 3.0
                * dme ** 2 * angular_factor
            )

        assert partials[0] > 0.0
        assert partials[1] > 0.0
        assert sum(partials) == pytest.approx(sigma_total[0])

    def test_below_threshold_zero(self):
        """Below ionization threshold, cross-section should be zero."""
        ns, ks = 40, 9
        t = make_knots(ns, ks, 0.01, 50.0)
        data = prepare_bspline_data(t, ns, ks)

        H, S = build_dirac_matrices(ns, -1, 1, data)
        result = solve_dirac(H, S, ns, -1)
        P_1s, Q_1s, E_1s = result.get_orbital(0)

        # Below threshold
        photon_energies = np.array([0.1, 0.2])
        sigma = photoionization_cross_section(
            P_1s, Q_1s, -1, E_1s, photon_energies, Z=1, bsp_data=data)
        assert np.all(sigma == 0.0)
