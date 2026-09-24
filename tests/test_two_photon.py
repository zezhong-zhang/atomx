"""Two-photon decay rates — frozen self-regression of the Drake 1986 fit.

CLASSIFICATION (audit S4 finding 22).  The "reference values" in this
file are a local re-typing of the very fit that
``atomx.properties.transitions.two_photon`` implements: the published
Drake coefficients (8.22943, 3.9448, 2.04, 4.6019, 16.458762, 0.806389)
written out again, with the fine-structure constant taken from the
shipped CODATA table on both sides.  These tests are therefore
``@pytest.mark.snapshot`` change-detectors on the fit coefficients, not
external validation.  The external oracle for two-photon rates is
``tests/test_two_photon_vs_literature.py::test_sos_h_like_*`` (an
independent B-spline sum-over-states path against the same fit) and the
Goldman & Drake 8.229 s⁻¹ point there.
"""

import math

import numpy as np
import pytest

from atomx.constants import ALPHA
from atomx.properties.transitions.two_photon import (
    two_photon_rate_h_like,
    two_photon_rate_he_like,
    two_photon_spectrum,
)


def _drake_h(Z: int) -> float:
    """Drake 1986 H-like fit, retyped locally with the shipped ALPHA."""
    a2 = (Z * ALPHA) ** 2
    return 8.22943 * Z**6 * (1 + 3.9448*a2 - 2.04*a2*a2) / (1 + 4.6019*a2)


def _drake_he(Z: int) -> float:
    z_eff = Z - 0.806389
    a2 = (z_eff * ALPHA) ** 2
    z6 = z_eff**6
    b = (Z + 2.5)**2
    return 16.458762 * (
        z6 * (1 + 1.539/b)
        - z6 * a2 * (0.6571 + 2.04*a2) / (1 + 4.6019*a2)
    )


@pytest.mark.snapshot
@pytest.mark.parametrize("Z", [1, 2, 5, 10, 20, 30, 54, 92])
def test_two_photon_h_like_matches_drake(Z):
    """Frozen: the code's Drake-fit coefficients == the retyped ones."""
    val = two_photon_rate_h_like(Z)
    ref = _drake_h(Z)
    assert math.isfinite(val) and val > 0
    assert val == pytest.approx(ref, rel=1e-6), (
        f"Z={Z}: AtomX {val:.6e} vs Drake {ref:.6e}"
    )


@pytest.mark.snapshot
@pytest.mark.parametrize("Z", [2, 4, 8, 10, 20, 30, 54])
def test_two_photon_he_like_matches_drake(Z):
    """Frozen: the code's He-like Drake/FAC fit == the retyped one."""
    val = two_photon_rate_he_like(Z)
    ref = _drake_he(Z)
    assert math.isfinite(val) and val > 0
    assert val == pytest.approx(ref, rel=1e-6), (
        f"Z={Z}: AtomX {val:.6e} vs Drake {ref:.6e}"
    )


@pytest.mark.snapshot
def test_two_photon_h_like_z_scaling():
    """Frozen: the fit's own Z⁶ prefactor is recovered at low Z."""
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


@pytest.mark.snapshot
def test_two_photon_spectrum_normalisation():
    """Frozen: ∫ψ dy = 2Γ with the source's own 2/(1/30) constant.

    The sum rule (each of the two photons counted once) is physical, but
    both sides come from the same analytic integral — see
    ``test_two_photon_spectrum_shape_is_not_the_literature_psi`` for the
    part of the shape that is *not* pinned by this normalisation.
    """
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


def test_two_photon_spectrum_is_positive_and_center_peaked():
    """The spectral shape must be the Spitzer-Greenstein ψ(y).

    A-class structural properties of the exact non-relativistic
    2s_{1/2} → 1s_{1/2} two-photon distribution (Spitzer & Greenstein,
    *Astrophys. J.* **114**, 407 (1951); Drake 1986): it is symmetric,
    vanishes only at the endpoints y → 0, 1, and is single-peaked at
    y = 1/2 — the y(1−y) dipole phase-space factor vanishes at the ends
    and nothing in the exact amplitude cancels at the centre.

    Measured: the implemented fit y(1−y)(1−4y(1−y)) has twin maxima
    0.0625 at y = 0.1464 / 0.8536 and is EXACTLY ZERO at y = 1/2, so the
    ``two_photon.py:168-174`` claim "accurate to ≤ 5 % across
    [0.05, 0.95]" is false by 100 % at the centre of the spectrum.
    """
    y = np.linspace(0.01, 0.99, 197)      # contains y = 0.5 exactly
    psi = two_photon_spectrum(y, Z=1)
    assert np.all(psi > 0.0), "ψ(y) must not vanish inside (0, 1)"
    psi_mid = float(psi[np.argmin(np.abs(y - 0.5))])
    assert psi_mid > 0.5 * float(psi.max()), (
        f"ψ(1/2) = {psi_mid:.6e} vs ψ_max = {psi.max():.6e} "
        f"at y = {y[int(np.argmax(psi))]:.4f}")


def test_two_photon_spectrum_unknown_state():
    with pytest.raises(ValueError):
        two_photon_spectrum(np.array([0.5]), Z=1, state="li-like")
