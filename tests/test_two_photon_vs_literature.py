"""Validation of two-photon E1E1 decay rates against Goldman & Drake.

Compares the sum-over-states (SOS) implementation in
``atomx.properties.transitions.two_photon`` with published reference values
from Goldman & Drake and the Drake 1986 fitting formula.

References
----------
* G. W. F. Drake, *Phys. Rev. A* **34**, 2871 (1986)
* S. P. Goldman & G. W. F. Drake, *Phys. Rev. A* **24**, 183 (1981)
* L. Spitzer & J. Greenstein, *Astrophys. J.* **114**, 407 (1951)
* M. F. Gu, *Astrophys. J.* **582**, 1241 (2003) — FAC TwoPhotonRate

Tolerances (per Z, measured 2026-09-25 at ns=60/ks=9/n_omega=100)
------------------------------------------------------------------
H-like SOS vs Drake 1986 fit:  Z=1 3.8e-5, Z=2 1.2e-4, Z=3 2.6e-4,
                               Z=5 7.1e-4, Z=10 2.7e-3.  The gate is
                               2-3x each measured point (1e-4 … 2e-3)
                               instead of the flat 1 % the file used.
He-like SOS vs Drake 1986 fit: ≤ 10 % (screened hydrogenic model;
                               measured 7.1 % at Z=2, 3.5 % at Z=4,
                               2.2 % at Z=6 — already at the 10 % claim)
Z⁶ scaling recovery:           ≤ 0.5 % (low-Z non-relativistic)
"""

import math

import pytest

from atomx.properties.transitions.two_photon import (
    two_photon_rate_h_like,
    two_photon_rate_he_like,
    two_photon_rate_sos_h_like,
    two_photon_rate_sos_he_like,
)

# ---------------------------------------------------------------------------
# H-like: SOS vs Drake 1986 / Goldman & Drake
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("Z,tol", [
    (1, 1e-4),   # measured 3.772e-05  (sos 8.228832 vs drake 8.229142)
    (2, 3e-4),   # measured 1.226e-04
    (3, 6e-4),   # measured 2.631e-04
    (5, 2e-3),   # measured 7.064e-04
])
def test_sos_h_like_matches_drake(Z, tol):
    """H-like SOS rate matches the Drake 1986 fit at the measured level.

    The SOS path is a genuine independent method (B-spline Dirac
    sum-over-states, ``two_photon.py:263-357``) and the Drake fit is
    external, so this is a real B-class gate; the flat 1 % it used to
    carry was 10-500x looser than the agreement it actually achieves.
    """
    sos = two_photon_rate_sos_h_like(Z, ns=60, ks=9, n_omega=100)
    drake = two_photon_rate_h_like(Z)
    assert sos > 0, f"Z={Z}: SOS rate must be positive"
    rel = abs(sos / drake - 1.0)
    assert rel < tol, (
        f"Z={Z}: SOS {sos:.4f} vs Drake {drake:.4f} "
        f"(rel {rel:.3e} > {tol:.0e})"
    )


def test_sos_h_like_hydrogen_goldman_drake():
    """H 2s → 1s + 2γ rate: SOS matches Goldman & Drake value 8.229 s⁻¹.

    The classical Spitzer-Greenstein / Goldman-Drake hydrogenic rate is
    A₂γ = 8.229 s⁻¹ at Z = 1 (non-relativistic limit).
    """
    val = two_photon_rate_sos_h_like(1, ns=60, ks=9, n_omega=100)
    assert math.isfinite(val) and val > 0
    # Measured: SOS 8.228832 s^-1 vs the Drake 1986 fit 8.229142 s^-1
    # (rel 3.8e-5); the literature value is quoted as 8.229.
    assert val == pytest.approx(8.229, rel=2e-4)


def test_sos_h_like_z6_scaling():
    """Z⁶ scaling recovered for low Z (non-relativistic regime).

    For Z ≪ 1/α the rate scales as Z⁶ with small relativistic corrections.
    """
    r2 = two_photon_rate_sos_h_like(2, ns=60, ks=9, n_omega=100)
    r4 = two_photon_rate_sos_h_like(4, ns=60, ks=9, n_omega=100)
    expected_ratio = (4.0 / 2.0) ** 6  # 64
    assert r4 / r2 == pytest.approx(expected_ratio, rel=0.005)


# ---------------------------------------------------------------------------
# He-like: SOS vs Drake 1986
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("Z", [2, 4, 6])
def test_sos_he_like_matches_drake(Z):
    """He-like SOS rate matches Drake 1986 fit within 10%.

    The He-like SOS uses a screened hydrogenic model (Z_eff = Z − σ)
    with a factor of 2 from exchange symmetry.  Agreement improves
    with Z as the screening becomes a smaller fraction of Z.
    """
    sos = two_photon_rate_sos_he_like(Z, ns=60, ks=9, n_omega=100)
    drake = two_photon_rate_he_like(Z)
    assert sos > 0, f"Z={Z}: SOS rate must be positive"
    rel = abs(sos / drake - 1.0)
    assert rel < 0.10, (
        f"Z={Z}: He-like SOS {sos:.4f} vs Drake {drake:.4f} "
        f"(rel {rel:.4f} > 10%)"
    )


def test_sos_he_like_z_scaling():
    """He-like rates scale approximately as Z⁶ (screened)."""
    r4 = two_photon_rate_sos_he_like(4, ns=60, ks=9, n_omega=100)
    r6 = two_photon_rate_sos_he_like(6, ns=60, ks=9, n_omega=100)
    # Not exact Z⁶ due to screening, but ratio should be in the right
    # ballpark: (6-σ)⁶ / (4-σ)⁶ ≈ (5.19/3.19)⁶ ≈ 23.8
    sigma = 0.806389
    expected = ((6.0 - sigma) / (4.0 - sigma)) ** 6
    assert r6 / r4 == pytest.approx(expected, rel=0.05)


# ---------------------------------------------------------------------------
# Consistency: Drake fit vs SOS agree to better than 1% for H-like
# ---------------------------------------------------------------------------

def test_drake_fit_self_consistent():
    """Drake fit and SOS use independent methods but must agree closely."""
    for Z in [1, 2, 5, 10]:
        drake = two_photon_rate_h_like(Z)
        sos = two_photon_rate_sos_h_like(Z, ns=60, ks=9, n_omega=100)
        assert drake == pytest.approx(sos, rel=0.01), (
            f"Z={Z}: Drake {drake:.6f} vs SOS {sos:.6f}"
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_sos_h_like_invalid_Z():
    with pytest.raises(ValueError):
        two_photon_rate_sos_h_like(0)


def test_sos_he_like_invalid_Z():
    with pytest.raises(ValueError):
        two_photon_rate_sos_he_like(1)
