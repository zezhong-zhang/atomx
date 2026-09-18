"""M-substate-resolved cross-section weight tests."""

import pytest

from atomx.scattering.msub import (
    alignment_from_m_resolved,
    m_resolved_cross_section,
    m_resolved_weights,
)


def test_J0_to_J1_axial_photo_absorption():
    """For J_a = 0 → J_b = 1 axial geometry, only |M_b| = 1 is populated."""
    w = m_resolved_weights(J_a=0.0, J_b=1.0, L=1)
    assert w[0.0] == pytest.approx(0.0, abs=1e-12)
    assert w[1.0] == pytest.approx(1.0, abs=1e-12)


def test_weights_sum_to_one():
    """For any allowed transition the weights sum to 1."""
    for (J_a, J_b) in [(1.0, 1.0), (1.0, 2.0), (0.5, 0.5), (1.5, 0.5)]:
        w = m_resolved_weights(J_a, J_b, L=1)
        assert sum(w.values()) == pytest.approx(1.0, rel=1e-12)


def test_forbidden_returns_uniform():
    """Triangle-rule violation gives uniform distribution (no preference)."""
    # E1 from J_a = 0 to J_b = 2 is forbidden (Δ J = 2 > 1)
    w = m_resolved_weights(J_a=0.0, J_b=2.0, L=1)
    expected = 1.0 / len(w)
    for v in w.values():
        assert v == pytest.approx(expected, rel=1e-12)


def test_split_preserves_total():
    """Σ σ(M) = σ_total."""
    sigma = m_resolved_cross_section(1.234, J_a=1.0, J_b=2.0, L=1)
    assert sum(sigma.values()) == pytest.approx(1.234, rel=1e-12)


def test_alignment_from_axial_excitation():
    """Axial J=0→1 excitation gives A_2 = √(2/3)/2 = √(1/6)."""
    sigma = m_resolved_cross_section(1.0, J_a=0.0, J_b=1.0, L=1)
    A_2 = alignment_from_m_resolved(sigma, J_b=1.0)
    # Inal-Dubau: A_2 = √(1/6) ≈ 0.408 for pure |M|=1 alignment
    expected = (1.0 / 6.0) ** 0.5
    assert pytest.approx(expected, rel=1e-3) == A_2


def test_higher_multipoles_not_implemented():
    with pytest.raises(NotImplementedError):
        m_resolved_weights(J_a=1.0, J_b=2.0, L=2)


def test_J_half_integer():
    """Half-integer J transitions work (J_a = 1/2 → J_b = 3/2)."""
    w = m_resolved_weights(J_a=0.5, J_b=1.5, L=1)
    # |M| = 1/2, 3/2 entries
    assert 0.5 in w
    assert 1.5 in w
    assert sum(w.values()) == pytest.approx(1.0, rel=1e-12)


def test_J0_to_J0_forbidden_e1():
    """0 → 0 E1 transition forbidden (single-photon)."""
    w = m_resolved_weights(J_a=0.0, J_b=0.0, L=1)
    # Returns uniform 1.0 for the single |M|=0 entry (no anisotropy)
    assert w[0.0] == pytest.approx(1.0, rel=1e-12)
