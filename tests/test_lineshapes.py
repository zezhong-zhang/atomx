"""Tests for line-profile primitives (Voigt / Lorentzian / Gaussian / Stark)."""

from __future__ import annotations

from math import log, pi, sqrt

import numpy as np
import pytest

from atomx.lineshapes import (
    doppler_width,
    gaussian,
    lorentzian,
    natural_width,
    stark_hwhm_quasistatic,
    voigt,
)
from atomx.lineshapes.profiles import (
    gaussian_fwhm,
    lorentzian_fwhm,
    voigt_fwhm,
)

# ---------------------------------------------------------------------------
# Normalisation & sanity
# ---------------------------------------------------------------------------

class TestNormalisation:
    def test_gaussian_integrates_to_unity(self):
        x = np.linspace(-20, 20, 10001)
        y = gaussian(x, x0=0.0, sigma=1.5)
        assert np.trapezoid(y, x) == pytest.approx(1.0, abs=1e-4)

    def test_lorentzian_integrates_to_unity(self):
        x = np.linspace(-500, 500, 100001)   # Lorentzian has fat tails
        y = lorentzian(x, x0=0.0, gamma=1.5)
        assert np.trapezoid(y, x) == pytest.approx(1.0, abs=5e-3)

    def test_voigt_integrates_to_unity(self):
        x = np.linspace(-500, 500, 100001)
        y = voigt(x, x0=0.0, sigma=1.2, gamma=0.8)
        # Lorentzian tail truncation limits numerical quadrature to ~0.1 %
        assert np.trapezoid(y, x) == pytest.approx(1.0, abs=5e-3)

    def test_voigt_reduces_to_gaussian(self):
        """Voigt(σ, γ=0) ≡ Gaussian(σ)."""
        x = np.linspace(-5, 5, 501)
        g = gaussian(x, x0=0.0, sigma=1.0)
        v = voigt(x, x0=0.0, sigma=1.0, gamma=1e-12)
        assert np.allclose(g, v, atol=1e-6)

    def test_voigt_reduces_to_lorentzian(self):
        """Voigt(σ=0, γ) ≡ Lorentzian(γ)."""
        x = np.linspace(-5, 5, 501)
        l_ = lorentzian(x, x0=0.0, gamma=1.0)
        v = voigt(x, x0=0.0, sigma=1e-12, gamma=1.0)
        assert np.allclose(l_, v, atol=1e-6)


# ---------------------------------------------------------------------------
# Peak & FWHM
# ---------------------------------------------------------------------------

class TestFWHM:
    def test_gaussian_fwhm_formula(self):
        sigma = 2.0
        expected = 2.0 * sqrt(2.0 * log(2.0)) * sigma
        assert gaussian_fwhm(sigma) == pytest.approx(expected, rel=1e-14)

    def test_lorentzian_fwhm_is_twice_gamma(self):
        assert lorentzian_fwhm(3.0) == pytest.approx(6.0, rel=1e-14)

    def test_voigt_fwhm_gaussian_limit(self):
        # Gaussian-dominated: γ tiny  → Voigt FWHM ≈ Gaussian FWHM
        sigma = 1.0
        fG = gaussian_fwhm(sigma)
        assert voigt_fwhm(sigma, 1e-9) == pytest.approx(fG, rel=1e-6)

    def test_voigt_fwhm_lorentzian_limit(self):
        # Lorentzian-dominated: σ tiny → Voigt FWHM ≈ Lorentzian FWHM
        # Olivero-Longbothum approximation has known ~0.02 % residual
        gamma = 2.0
        fL = lorentzian_fwhm(gamma)
        assert voigt_fwhm(1e-9, gamma) == pytest.approx(fL, rel=5e-5)


# ---------------------------------------------------------------------------
# Doppler & natural widths
# ---------------------------------------------------------------------------

class TestDopplerWidth:
    def test_h_alpha_room_temperature(self):
        """H-α at 656.28 nm (457 THz) at T=300 K, M=1.008 amu:
           σ = ν · √(kT/Mc²) ≈ 2.3 GHz.  (FWHM ≈ 5.5 GHz)
        """
        nu_THz = 457.0
        sigma_THz = doppler_width(nu_THz, T_kelvin=300.0, mass_amu=1.008)
        # σ in THz; convert → GHz for comparison
        sigma_GHz = sigma_THz * 1000.0
        assert sigma_GHz == pytest.approx(2.33, rel=0.05)

    def test_doppler_scales_with_sqrt_T(self):
        s1 = doppler_width(1.0, T_kelvin=100.0, mass_amu=1.0)
        s4 = doppler_width(1.0, T_kelvin=400.0, mass_amu=1.0)
        assert s4 == pytest.approx(2.0 * s1, rel=1e-12)

    def test_doppler_scales_inverse_sqrt_mass(self):
        s1 = doppler_width(1.0, T_kelvin=300.0, mass_amu=1.0)
        s4 = doppler_width(1.0, T_kelvin=300.0, mass_amu=4.0)
        assert s4 == pytest.approx(0.5 * s1, rel=1e-12)

    def test_rejects_bad_inputs(self):
        with pytest.raises(ValueError):
            doppler_width(1.0, T_kelvin=0.0, mass_amu=1.0)
        with pytest.raises(ValueError):
            doppler_width(1.0, T_kelvin=300.0, mass_amu=0.0)


class TestNaturalWidth:
    def test_natural_width_is_A_over_4pi(self):
        A = 6.28e8   # 1/s
        gamma = natural_width(A)
        assert gamma == pytest.approx(A / (4.0 * pi), rel=1e-14)


# ---------------------------------------------------------------------------
# Stark (Griem)
# ---------------------------------------------------------------------------

class TestStark:
    def test_h_alpha_reference_density(self):
        """Griem table: H-α HWHM ≈ 0.38 Å at n_e = 1e17 cm⁻³."""
        hwhm = stark_hwhm_quasistatic(n_upper=3, n_lower=2, n_e_per_cm3=1e17)
        assert hwhm == pytest.approx(0.38, rel=1e-6)

    def test_stark_scales_as_ne_two_thirds(self):
        """Δλ_S ∝ n_e^{2/3}:  increasing n_e 8× increases HWHM 4×."""
        h1 = stark_hwhm_quasistatic(3, 2, 1e17)
        h2 = stark_hwhm_quasistatic(3, 2, 8e17)
        assert h2 == pytest.approx(4.0 * h1, rel=1e-6)

    def test_unknown_transition_raises(self):
        with pytest.raises(KeyError):
            stark_hwhm_quasistatic(9, 2, 1e17)

    def test_non_hydrogen_not_supported(self):
        with pytest.raises(NotImplementedError):
            stark_hwhm_quasistatic(3, 2, 1e17, element="He")
