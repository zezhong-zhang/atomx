"""Two-photon decay rates pinned to Drake 1986 / FAC TwoPhotonRate."""

import math

import numpy as np
import pytest

from atomx.properties.transitions.two_photon import (
    two_photon_rate_h_like,
    two_photon_rate_he_like,
    two_photon_spectrum,
)


# Drake 1986 / FAC TwoPhotonRate(z, 0) reference values, computed from
# the exact closed-form fit so AtomX should match these to round-off.
_H_LIKE_REF = {
    1:   8.22943 * (
            (1 + 3.9448 * (1/137.035999)**2 - 2.04 * (1/137.035999)**4)
            / (1 + 4.6019 * (1/137.035999)**2)),
    10:  None,  # filled below
    54:  None,
}


def _drake_h(Z: int) -> float:
    """Reference Drake 1986 H-like fit, recomputed locally."""
    a2 = (Z / 137.035999)**2
    return 8.22943 * Z**6 * (1 + 3.9448*a2 - 2.04*a2*a2) / (1 + 4.6019*a2)


def _drake_he(Z: int) -> float:
    z_eff = Z - 0.806389
    a2 = (z_eff / 137.035999)**2
    z6 = z_eff**6
    b = (Z + 2.5)**2
    return 16.458762 * (
        z6 * (1 + 1.539/b)
        - z6 * a2 * (0.6571 + 2.04*a2) / (1 + 4.6019*a2)
    )


@pytest.mark.parametrize("Z", [1, 2, 5, 10, 20, 30, 54, 92])
def test_two_photon_h_like_matches_drake(Z):
    """H-like 2s → 1s + 2γ rate matches the Drake 1986 fit FAC uses."""
    val = two_photon_rate_h_like(Z)
    ref = _drake_h(Z)
    assert math.isfinite(val) and val > 0
    assert val == pytest.approx(ref, rel=1e-6), (
        f"Z={Z}: AtomX {val:.6e} vs Drake {ref:.6e}"
    )


@pytest.mark.parametrize("Z", [2, 4, 8, 10, 20, 30, 54])
def test_two_photon_he_like_matches_drake(Z):
    """He-like 1s 2s ¹S₀ → 1s² + 2γ rate matches Drake/FAC fit."""
    val = two_photon_rate_he_like(Z)
    ref = _drake_he(Z)
    assert math.isfinite(val) and val > 0
    assert val == pytest.approx(ref, rel=1e-6), (
        f"Z={Z}: AtomX {val:.6e} vs Drake {ref:.6e}"
    )


def test_two_photon_h_like_z_scaling():
    """Z⁶ scaling is recovered in the non-relativistic limit (low Z)."""
    r1 = two_photon_rate_h_like(2)
    r2 = two_photon_rate_h_like(4)
    # (4/2)⁶ = 64 — ratio should be very close to 64 since (αZ)² ≪ 1.
    assert r2 / r1 == pytest.approx(64.0, rel=1e-3)


def test_two_photon_h_like_below_threshold():
    with pytest.raises(ValueError):
        two_photon_rate_h_like(0)


def test_two_photon_he_like_below_threshold():
    with pytest.raises(ValueError):
        two_photon_rate_he_like(1)


def test_two_photon_spectrum_normalisation():
    """Symmetric polynomial spectrum integrates to 2 · Γ_total
    (each photon counted once)."""
    y = np.linspace(0.001, 0.999, 5001)
    Z = 1
    dGdy = two_photon_spectrum(y, Z)
    integral = float(np.trapezoid(dGdy, y))
    expected = 2.0 * two_photon_rate_h_like(Z)
    assert integral == pytest.approx(expected, rel=2e-3)


def test_two_photon_spectrum_symmetry():
    """ψ(y) = ψ(1−y) — Spitzer-Greenstein symmetry."""
    y = np.array([0.1, 0.2, 0.3, 0.4])
    sa = two_photon_spectrum(y, Z=1)
    sb = two_photon_spectrum(1.0 - y, Z=1)
    np.testing.assert_allclose(sa, sb, rtol=1e-12)


def test_two_photon_spectrum_invalid_y():
    with pytest.raises(ValueError):
        two_photon_spectrum(np.array([0.0, 0.5]), Z=1)
    with pytest.raises(ValueError):
        two_photon_spectrum(np.array([0.5, 1.0]), Z=1)


def test_two_photon_spectrum_unknown_state():
    with pytest.raises(ValueError):
        two_photon_spectrum(np.array([0.5]), Z=1, state="li-like")
