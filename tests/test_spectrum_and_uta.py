"""Tests for the Spectrum container and UTA statistical moments."""

from __future__ import annotations

import numpy as np
import pytest

from atomx.spectrum import Spectrum, synthesize_from_transitions
from atomx.uta import UTAArray, uta_moments, uta_simple_from_levels


# ---------------------------------------------------------------------------
# Spectrum container
# ---------------------------------------------------------------------------

class TestSpectrumConstructors:
    def test_linear(self):
        s = Spectrum.linear(100.0, 200.0, 1001)
        assert len(s) == 1001
        assert s.wavelength_nm[0] == pytest.approx(100.0)
        assert s.wavelength_nm[-1] == pytest.approx(200.0)
        # Uniformly spaced
        assert np.allclose(np.diff(s.wavelength_nm), 0.1)
        # Zero-initialised
        assert np.all(s.values == 0.0)

    def test_logarithmic(self):
        s = Spectrum.logarithmic(10.0, 1000.0, 4)
        # Geometric progression: factor of 10^(1/3)
        ratios = s.wavelength_nm[1:] / s.wavelength_nm[:-1]
        assert np.allclose(ratios, ratios[0])

    def test_from_grid(self):
        grid = np.array([1.0, 2.0, 3.5, 9.9])
        s = Spectrum.from_grid(grid)
        assert np.array_equal(s.wavelength_nm, grid)


class TestSpectrumAddLine:
    def test_gaussian_line_area_matches_intensity(self):
        s = Spectrum.linear(95.0, 105.0, 10001)
        s.add_line(100.0, intensity=1.0, sigma_nm=0.1)
        assert s.integrate() == pytest.approx(1.0, abs=1e-3)
        assert s.line_count == 1

    def test_lorentzian_line_area_matches_intensity(self):
        # Lorentzian: need wider grid because of fat tails
        s = Spectrum.linear(50.0, 150.0, 50001)
        s.add_line(100.0, intensity=1.0, gamma_nm=0.1)
        assert s.integrate() == pytest.approx(1.0, abs=5e-3)

    def test_voigt_line(self):
        s = Spectrum.linear(95.0, 105.0, 10001)
        s.add_line(100.0, intensity=1.0, sigma_nm=0.1, gamma_nm=0.05)
        # Voigt: Lorentzian wings leak out of the truncated window
        assert s.integrate() == pytest.approx(1.0, abs=1e-2)

    def test_multiple_lines_superpose(self):
        s = Spectrum.linear(95.0, 105.0, 10001)
        s.add_line(98.0, intensity=1.0, sigma_nm=0.1)
        s.add_line(102.0, intensity=2.0, sigma_nm=0.1)
        assert s.line_count == 2
        assert s.integrate() == pytest.approx(3.0, abs=1e-3)

    def test_must_supply_a_width(self):
        s = Spectrum.linear(0.0, 1.0, 11)
        with pytest.raises(ValueError, match="sigma_nm"):
            s.add_line(0.5, 1.0)

    def test_zero_intensity_skipped(self):
        s = Spectrum.linear(0.0, 1.0, 11)
        s.add_line(0.5, 0.0, sigma_nm=0.1)
        assert np.all(s.values == 0.0)
        assert s.line_count == 0


class TestSpectrumAnalysis:
    def test_peak(self):
        s = Spectrum.linear(95.0, 105.0, 10001)
        s.add_line(100.0, intensity=1.0, sigma_nm=0.1)
        lam, val = s.peak()
        assert lam == pytest.approx(100.0, abs=0.01)
        assert val > 0

    def test_normalize_area(self):
        s = Spectrum.linear(95.0, 105.0, 10001)
        s.add_line(100.0, intensity=5.0, sigma_nm=0.1)
        n = s.normalize()
        assert n.integrate() == pytest.approx(1.0, abs=1e-3)

    def test_integrate_sub_range(self):
        s = Spectrum.linear(95.0, 105.0, 10001)
        s.add_line(100.0, intensity=1.0, sigma_nm=0.01)
        # ±3σ should capture most of the area
        sub_area = s.integrate(99.97, 100.03)
        assert sub_area == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# synthesize_from_transitions
# ---------------------------------------------------------------------------

class TestSynthesizeFromTransitions:
    def test_from_tuples(self):
        lines = [(100.0, 1.0), (120.0, 0.5), (140.0, 0.3)]
        s = synthesize_from_transitions(
            lines, lam_min_nm=80, lam_max_nm=150,
            n_points=20001, default_sigma_nm=0.1,
        )
        assert s.line_count == 3
        # Integrated area ~ sum of intensities
        assert s.integrate() == pytest.approx(1.8, abs=0.05)

    def test_lines_outside_range_are_dropped(self):
        lines = [(100.0, 1.0), (200.0, 1.0)]   # 200 out of range
        s = synthesize_from_transitions(
            lines, lam_min_nm=80, lam_max_nm=150,
            n_points=10001, default_sigma_nm=0.1,
        )
        assert s.line_count == 1

    def test_from_dicts(self):
        lines = [
            {"wavelength_nm": 100.0, "A_einstein": 1.0},
            {"wavelength_nm": 120.0, "A_einstein": 0.5},
        ]
        s = synthesize_from_transitions(
            lines, lam_min_nm=80, lam_max_nm=150, n_points=5001,
            default_sigma_nm=0.2,
        )
        assert s.line_count == 2

    def test_requires_a_fallback_width(self):
        with pytest.raises(ValueError, match="fallback width"):
            synthesize_from_transitions(
                [(100.0, 1.0)], lam_min_nm=90, lam_max_nm=110, n_points=101,
            )


# ---------------------------------------------------------------------------
# UTA moments
# ---------------------------------------------------------------------------

class TestUTAMoments:
    def test_single_line_zero_variance(self):
        out = uta_moments([(5.0, 1.0)])
        assert isinstance(out, UTAArray)
        assert out.total_strength == 1.0
        assert out.centroid_eV == pytest.approx(5.0)
        assert out.variance_eV2 == pytest.approx(0.0, abs=1e-14)
        assert out.n_lines == 1

    def test_two_equal_weight_lines_centroid(self):
        out = uta_moments([(4.0, 1.0), (6.0, 1.0)])
        assert out.centroid_eV == pytest.approx(5.0)
        # σ² = ((4-5)² + (6-5)²)/2 = 1
        assert out.variance_eV2 == pytest.approx(1.0, rel=1e-12)
        assert out.std_eV == pytest.approx(1.0, rel=1e-12)

    def test_weighted_centroid(self):
        # 90 % at 5 eV, 10 % at 6 eV → centroid = 5.1
        out = uta_moments([(5.0, 9.0), (6.0, 1.0)])
        assert out.centroid_eV == pytest.approx(5.1, rel=1e-12)

    def test_fwhm_is_gaussian_equivalent(self):
        out = uta_moments([(4.0, 1.0), (6.0, 1.0)])
        # σ = 1 → FWHM = 2·√(2 ln 2)·σ ≈ 2.355
        assert out.fwhm_eV == pytest.approx(2.3548, abs=0.001)

    def test_centroid_nm_matches_hc_over_eV(self):
        out = uta_moments([(10.0, 1.0)])
        # hc = 1239.842 eV·nm → at 10 eV, λ = 123.98 nm
        assert out.centroid_nm == pytest.approx(123.984, abs=0.01)

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="at least one line"):
            uta_moments([])

    def test_rejects_zero_total_strength(self):
        with pytest.raises(ValueError, match="positive"):
            uta_moments([(5.0, 0.0)])


class TestCRMSynthesis:
    def _make_crm(self):
        """Minimal CRM: ground + two excited levels, two E1 transitions."""
        from atomx.plasma.crm import (
            CollisionalRadiativeModel, CRMLevel, CRMTransition,
        )
        levels = [
            CRMLevel(label="ground", energy_eV=0.0, g=1),
            CRMLevel(label="exc1",   energy_eV=10.0, g=3),
            CRMLevel(label="exc2",   energy_eV=12.4, g=5),
        ]
        transitions = [
            CRMTransition(upper=1, lower=0, A_rate=1.0e8, gf=0.5),
            CRMTransition(upper=2, lower=0, A_rate=2.0e8, gf=1.0),
            CRMTransition(upper=2, lower=1, A_rate=5.0e7, gf=0.1),
        ]
        return CollisionalRadiativeModel(levels=levels, transitions=transitions)

    def test_crm_emission_returns_spectrum(self):
        from atomx.spectrum import crm_emission_spectrum
        crm = self._make_crm()
        spec = crm_emission_spectrum(
            crm, n_e_cm3=1e14, T_e_eV=5.0,
            lam_min_nm=80.0, lam_max_nm=150.0, n_points=5001,
            ion_mass_amu=20.0,
        )
        assert spec.line_count > 0
        assert spec.metadata["n_e_cm3"] == 1e14
        assert spec.values.max() > 0

    def test_crm_emission_respects_profile_choice(self):
        from atomx.spectrum import crm_emission_spectrum
        crm = self._make_crm()
        s_g = crm_emission_spectrum(
            crm, n_e_cm3=1e14, T_e_eV=5.0,
            lam_min_nm=80.0, lam_max_nm=150.0, n_points=5001,
            ion_mass_amu=20.0, profile="gaussian",
        )
        s_v = crm_emission_spectrum(
            crm, n_e_cm3=1e14, T_e_eV=5.0,
            lam_min_nm=80.0, lam_max_nm=150.0, n_points=5001,
            ion_mass_amu=20.0, profile="voigt",
        )
        # Gaussian and Voigt at the same σ differ mainly in the wings;
        # integrated intensity is within a few percent.
        assert s_g.integrate() == pytest.approx(s_v.integrate(), rel=0.1)

    def test_crm_emission_invalid_profile_raises(self):
        from atomx.spectrum import crm_emission_spectrum
        crm = self._make_crm()
        with pytest.raises(ValueError, match="profile"):
            crm_emission_spectrum(
                crm, n_e_cm3=1e14, T_e_eV=5.0,
                lam_min_nm=80.0, lam_max_nm=150.0,
                ion_mass_amu=20.0, profile="triangle",
            )


class TestUTAFromLevels:
    def test_two_to_two_matrix(self):
        out = uta_simple_from_levels(
            E_upper_levels_eV=[10.0, 11.0],
            E_lower_levels_eV=[1.0, 2.0],
        )
        # All four transitions allowed; E_ij ∈ {8, 9, 9, 10}, equal weights
        assert out.n_lines == 4
        expected_centroid = (8 + 9 + 9 + 10) / 4
        assert out.centroid_eV == pytest.approx(expected_centroid, rel=1e-12)

    def test_negative_energy_transitions_dropped(self):
        out = uta_simple_from_levels(
            E_upper_levels_eV=[5.0, 0.5],   # 0.5 < 1.0 → dropped
            E_lower_levels_eV=[1.0],
        )
        assert out.n_lines == 1
        assert out.centroid_eV == pytest.approx(4.0)

    def test_weight_shapes_must_match(self):
        with pytest.raises(ValueError, match="match in length"):
            uta_simple_from_levels(
                E_upper_levels_eV=[10.0],
                E_lower_levels_eV=[1.0],
                g_upper=[1.0, 2.0],   # length mismatch
            )
