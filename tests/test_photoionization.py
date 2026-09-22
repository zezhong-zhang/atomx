"""Tests for the Fano-resonance photoionization helper."""

from __future__ import annotations

import numpy as np
import pytest

from atomx.scattering.photoionization import (
    FanoResonance,
    apply_resonances,
    fano_profile,
)


class TestFanoProfile:
    def test_peak_value_for_infinite_q(self):
        """q → ∞ gives a Lorentzian at ε=0:  F(0) = q² / 1."""
        E = np.linspace(9.5, 10.5, 1001)
        F = fano_profile(E, E_res_eV=10.0, gamma_eV=0.1, q=100.0)
        # Peak at ε=0 should be ≈ q²
        assert F.max() == pytest.approx(100 ** 2, rel=0.01)

    def test_window_resonance_q_zero(self):
        """q = 0 gives the "window" resonance:  F(ε=0) = 0."""
        F0 = fano_profile(10.0, 10.0, gamma_eV=0.1, q=0.0)
        assert pytest.approx(0.0, abs=1e-14) == F0

    def test_asymptotic_behaviour(self):
        """Far from the resonance F → 1 (slowly: 1/ε convergence).
        At ε = 1800 (E = 100, E_res = 10, Γ = 0.1, q = 2) the residual
        is ≈ 2q/ε ≈ 0.002."""
        F_far = fano_profile(100.0, 10.0, gamma_eV=0.1, q=2.0)
        assert F_far == pytest.approx(1.0, abs=0.01)
        # And the residual goes to zero as |E − E_res| grows.
        F_very_far = fano_profile(1e6, 10.0, gamma_eV=0.1, q=2.0)
        assert F_very_far == pytest.approx(1.0, abs=1e-6)

    def test_symmetric_limit(self):
        """The Fano profile is symmetric about 1: F(+ε) + F(-ε) = 2.

        First principles: with ε = 2(E - E_res)/Γ, F = (q + ε)²/(1 + ε²)
        expands to 1 + (q² - 1 + 2qε)/(1 + ε²), whose ε-odd part changes sign
        under ε → -ε, so the pair sums to exactly 2 for ρ² = 1 (the old
        version of this test asserted only F(0) = q² and never checked the
        symmetry its name claims).  At ε = 0 the profile is q².
        """
        eps = np.array([-2.0, -0.5, 0.5, 2.0, 8.0])
        E = 10.0 + eps * 0.1 / 2.0
        F = fano_profile(E, 10.0, gamma_eV=0.1, q=1.0)
        pairs = F[:2][::-1] + F[2:4]
        np.testing.assert_allclose(pairs, 2.0, rtol=1e-12)
        assert fano_profile(10.0, 10.0, gamma_eV=0.1, q=1.0) == pytest.approx(1.0)
        assert fano_profile(10.0, 10.0, gamma_eV=0.1, q=3.0) == pytest.approx(9.0)

    def test_rho_sq_partial_coupling(self):
        """With rho_sq < 1, F no longer reaches q² (diluted)."""
        F_full = fano_profile(10.0, 10.0, gamma_eV=0.1, q=10.0, rho_sq=1.0)
        F_half = fano_profile(10.0, 10.0, gamma_eV=0.1, q=10.0, rho_sq=0.5)
        assert F_half < F_full
        # With rho²=0, F ≡ 1
        F_zero = fano_profile(10.0, 10.0, gamma_eV=0.1, q=10.0, rho_sq=0.0)
        assert F_zero == pytest.approx(1.0)

    def test_rejects_bad_gamma(self):
        with pytest.raises(ValueError, match="gamma"):
            fano_profile(10.0, 10.0, gamma_eV=0.0, q=1.0)

    def test_rejects_bad_rho_sq(self):
        with pytest.raises(ValueError, match="rho_sq"):
            fano_profile(10.0, 10.0, gamma_eV=0.1, q=1.0, rho_sq=1.5)


class TestFanoResonance:
    def test_basic_construction(self):
        r = FanoResonance(E_res_eV=25.0, gamma_eV=0.05, q=3.0, label="3s3p")
        assert r.E_res_eV == 25.0
        assert r.gamma_eV == 0.05
        assert r.q == 3.0
        assert r.rho_sq == 1.0
        assert r.label == "3s3p"

    def test_rejects_negative_gamma(self):
        with pytest.raises(ValueError, match="positive"):
            FanoResonance(E_res_eV=25.0, gamma_eV=-0.1, q=3.0)

    def test_rejects_bad_rho(self):
        with pytest.raises(ValueError, match="rho_sq"):
            FanoResonance(E_res_eV=25.0, gamma_eV=0.1, q=3.0, rho_sq=1.5)


class TestApplyResonances:
    def test_single_resonance_modulates_background(self):
        E = np.linspace(5.0, 30.0, 5001)
        sigma_bg = np.ones_like(E) * 2.0    # 2 Mb background
        r = FanoResonance(E_res_eV=15.0, gamma_eV=0.2, q=5.0)
        sigma = apply_resonances(sigma_bg, E, [r])
        # Far from resonance → asymptotically unchanged (Fano 1/ε tail)
        assert sigma[0] == pytest.approx(2.0, abs=0.2)
        assert sigma[-1] == pytest.approx(2.0, abs=0.2)
        # Peak ≈ 2 · q² = 50
        assert sigma.max() == pytest.approx(2.0 * 25, rel=0.05)

    def test_multi_resonance_independent(self):
        """Two well-separated resonances act independently."""
        E = np.linspace(5.0, 30.0, 5001)
        bg = np.ones_like(E)
        r1 = FanoResonance(10.0, 0.05, q=3.0)
        r2 = FanoResonance(20.0, 0.05, q=4.0)
        sigma = apply_resonances(bg, E, [r1, r2])
        # Peak near r1 → q²(r1) = 9
        idx1 = np.argmin(np.abs(E - 10.0))
        assert sigma[idx1] == pytest.approx(9.0, rel=0.05)
        # Peak near r2 → q²(r2) = 16
        idx2 = np.argmin(np.abs(E - 20.0))
        assert sigma[idx2] == pytest.approx(16.0, rel=0.05)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape"):
            apply_resonances(
                sigma_bg=np.ones(5), E_eV=np.ones(6), resonances=[],
            )

    def test_no_resonances_returns_copy(self):
        bg = np.array([1.0, 2.0, 3.0])
        E = np.array([5.0, 10.0, 15.0])
        sigma = apply_resonances(bg, E, [])
        assert np.array_equal(sigma, bg)
        # Should be a copy, not the same array
        sigma[0] = 999
        assert bg[0] == 1.0
