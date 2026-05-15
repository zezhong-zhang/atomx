"""Tests for continuum Dirac solver and photoionization cross-sections."""

import numpy as np
import pytest
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
