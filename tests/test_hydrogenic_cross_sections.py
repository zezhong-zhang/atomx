"""Closed-form hydrogenic cross sections / rates pinned to NIST + Drake."""

import math

import numpy as np
import pytest

from atomx.methods.hydrogenic_cross_sections import (
    hydrogenic_e1_rate,
    kramers_pi_cross_section,
    milne_rr_cross_section,
    stobbe_1s_pi_cross_section,
)


def test_stobbe_threshold_h():
    """Stobbe σ(H 1s) at threshold within 1 % of NIST 6.30e-18 cm²."""
    sigma = stobbe_1s_pi_cross_section(1, np.array([0.5001]))[0]
    assert sigma == pytest.approx(6.30e-18, rel=0.01)


def test_stobbe_threshold_he_plus():
    """Stobbe σ(He+ 1s) at threshold within 1 % of 1.58e-18 cm²."""
    sigma = stobbe_1s_pi_cross_section(2, np.array([2.0001]))[0]
    assert sigma == pytest.approx(1.575e-18, rel=0.01)


def test_stobbe_z_squared_scaling():
    """At threshold σ_K(Z) = σ_K(1) / Z² (modulo relativistic corr)."""
    s_h = stobbe_1s_pi_cross_section(1, np.array([0.5001]))[0]
    s_he = stobbe_1s_pi_cross_section(2, np.array([2.0001]))[0]
    assert s_he / s_h == pytest.approx(1.0 / 4.0, rel=1e-3)


def test_stobbe_below_threshold():
    """Below threshold returns 0."""
    omega = np.array([0.4, 0.45, 0.499])
    sigma = stobbe_1s_pi_cross_section(1, omega)
    assert np.all(sigma == 0)


def test_stobbe_falls_at_high_omega():
    """σ(ω) is monotonically decreasing for ω > threshold."""
    omega = np.geomspace(0.51, 100.0, 20)
    sigma = stobbe_1s_pi_cross_section(1, omega)
    assert np.all(np.diff(sigma) < 0)


def test_kramers_high_omega_asymptote():
    """Kramers σ ∝ 1/ω³ at large ω."""
    omega = np.array([10.0, 30.0, 100.0])
    sigma = kramers_pi_cross_section(1, 1, omega)
    # σ(ω₂) / σ(ω₁) = (ω₁/ω₂)³
    r1 = sigma[1] / sigma[0]
    r2 = sigma[2] / sigma[1]
    assert r1 == pytest.approx((omega[0] / omega[1]) ** 3, rel=1e-9)
    assert r2 == pytest.approx((omega[1] / omega[2]) ** 3, rel=1e-9)


def test_milne_rr_he_plus_falls_with_energy():
    """σ_RR ~ 1/E at low E (Sommerfeld limit)."""
    Ee = np.geomspace(0.05, 50.0, 20) / 27.211
    sigma = milne_rr_cross_section(2, 1, Ee, g_recombined=2, g_recombining=1)
    assert np.all(np.diff(sigma) < 0)


def test_milne_rr_invalid_energy():
    with pytest.raises(ValueError):
        milne_rr_cross_section(2, 1, np.array([-1.0]))


def test_lyman_alpha_h():
    """A(2p → 1s, Z=1) = 6.265 × 10⁸ s⁻¹ (NIST)."""
    A = hydrogenic_e1_rate(Z=1, n_upper=2, n_lower=1, l_upper=1, l_lower=0)
    assert A == pytest.approx(6.27e8, rel=1e-3)


def test_lyman_alpha_z4_scaling():
    """A(2p→1s, Z) ∝ Z⁴."""
    A1 = hydrogenic_e1_rate(1, 2, 1, 1, 0)
    A2 = hydrogenic_e1_rate(2, 2, 1, 1, 0)
    A26 = hydrogenic_e1_rate(26, 2, 1, 1, 0)
    assert A2 == pytest.approx(16 * A1, rel=1e-12)
    assert A26 == pytest.approx(26**4 * A1, rel=1e-12)


def test_hydrogenic_e1_unknown_transition():
    with pytest.raises(KeyError):
        hydrogenic_e1_rate(Z=1, n_upper=5, n_lower=1, l_upper=1, l_lower=0)
