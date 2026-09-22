"""EDM enhancement-factor closed-form scaling tests."""

import pytest

from atomx import Atom
from atomx.properties.edm import (
    edm_summary,
    electron_edm_enhancement_sandars,
    schiff_moment_enhancement_scaling,
)


def test_R_e_cs_like_order_of_magnitude():
    """Cs ²S₁/₂ (Z=55, n_eff=1.9) gives R_e ~ few × 10² — same OOM as
    Liu-Kelly all-order MBPT R_e ≈ 120."""
    R_e = electron_edm_enhancement_sandars(Z=55, n_eff=1.9, J=0.5, L=0)
    assert 30 < R_e < 500


def test_R_e_z_cubed_scaling():
    """R_e ∝ Z³ at fixed n_eff, J, L."""
    R1 = electron_edm_enhancement_sandars(Z=10, n_eff=2.0)
    R2 = electron_edm_enhancement_sandars(Z=20, n_eff=2.0)
    assert pytest.approx(8.0, rel=1e-12) == R2 / R1


def test_R_e_p_state_smaller():
    """L=1 (p-state) gives smaller R_e than L=0 by 1/3 factor."""
    R_s = electron_edm_enhancement_sandars(Z=55, n_eff=1.9, J=0.5, L=0)
    R_p = electron_edm_enhancement_sandars(Z=55, n_eff=1.9, J=0.5, L=1)
    assert R_p / R_s == pytest.approx(1.0 / 3.0, rel=1e-12)


def test_R_e_invalid():
    with pytest.raises(ValueError):
        electron_edm_enhancement_sandars(Z=0)
    with pytest.raises(ValueError):
        electron_edm_enhancement_sandars(Z=10, n_eff=0)
    with pytest.raises(ValueError):
        electron_edm_enhancement_sandars(Z=10, J=0)
    with pytest.raises(ValueError):
        electron_edm_enhancement_sandars(Z=10, L=-1)


def test_K_S_z_dependence():
    """K_S ∝ Z² (αZ)² = Z⁴ leading order."""
    K_lo = schiff_moment_enhancement_scaling(Z=20, A=40, valence="s")
    K_hi = schiff_moment_enhancement_scaling(Z=80, A=200, valence="s")
    ratio = K_hi / K_lo
    expected_z4 = (80.0 / 20.0) ** 4
    assert ratio == pytest.approx(expected_z4, rel=1e-12)


def test_K_S_hg_order():
    """Hg (Z=80, A=199) K_S ~ 5e-17 cm (Dzuba-Flambaum tabulate ~4e-17)."""
    K = schiff_moment_enhancement_scaling(Z=80, A=199, valence="closed")
    # Order-of-magnitude check
    assert 1e-18 < K < 1e-16


def test_K_S_invalid_valence():
    with pytest.raises(ValueError):
        schiff_moment_enhancement_scaling(Z=80, A=199, valence="banana")


def test_K_S_invalid_Z_A():
    with pytest.raises(ValueError):
        schiff_moment_enhancement_scaling(Z=0, A=10)
    with pytest.raises(ValueError):
        schiff_moment_enhancement_scaling(Z=10, A=0)


def test_summary_returns_all_keys():
    s = edm_summary(Z=80, A=199, valence="closed")
    for key in ("Z", "A", "n_eff", "J", "L", "valence",
                 "R_electron_edm", "K_schiff_cm"):
        assert key in s


def test_compute_edm_enhancement_high_level_api():
    """High-level compute_edm_enhancement accepts a solved Atom."""
    from atomx.methods.edm import compute_edm_enhancement

    atom = Atom(Z=3, config="1s2 2s1", engine="bspline", corrections="none")
    atom.solve()
    result = compute_edm_enhancement(atom, "2s1/2")
    assert hasattr(result, "R_electron_edm")
    assert hasattr(result, "per_orbital")


def test_edm_summary_scaling_path():
    res = edm_summary(Z=55, A=133, n_eff=1.9, J=0.5, L=0)
    assert res["R_electron_edm"] > 0
    assert res["K_schiff_cm"] > 0
