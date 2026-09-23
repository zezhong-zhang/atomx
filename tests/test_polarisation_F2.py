"""Steffen-Alder F_2 coefficient regression tests."""

import math

import pytest

from atomx.scattering.polarisation import steffen_alder_F2

# Textbook table from Steffen-Alder 1965 / Krause 1979 for E1 (L=1):
#
#   J_u → J_l   F_2
#   1   → 0     +1/√2  ≈ +0.7071
#   1   → 1     −1/(2√2) ≈ −0.3536
#   1   → 2     +1/(10·√2) ≈ +0.0707
#   2   → 1     ≈ +0.4183
#   2   → 2     ≈ −0.4183  (sign mirror of 2→1 by 6j symmetry)
#   2   → 3     ≈ +0.1195
#   3/2 → 1/2   = 0      (special — verify finite)


@pytest.mark.parametrize("J_u,J_l,expected", [
    (1.0, 0.0, +1.0 / math.sqrt(2)),
    (1.0, 1.0, -1.0 / (2.0 * math.sqrt(2))),
    (1.0, 2.0, +1.0 / (10.0 * math.sqrt(2))),
])
def test_F2_J1_known_values(J_u, J_l, expected):
    """E1 line F_2 from a J=1 upper level — three reference points."""
    val = steffen_alder_F2(J_u, J_l, L=1)
    assert val == pytest.approx(expected, rel=1e-10), (
        f"F_2({J_u}→{J_l}) = {val:.4f}, expected {expected:.4f}"
    )


def test_F2_J_half_integer_alignment_zero():
    """J_u = 1/2 cannot be aligned (rank-2 tensor vanishes)."""
    assert steffen_alder_F2(0.5, 0.5, L=1) == 0.0


def test_F2_forbidden_triangle_zero():
    """Triangle-rule violation returns 0."""
    # J_u=2, J_l=0 forbidden for E1 (|J_u−J_l|=2 > L=1)
    assert steffen_alder_F2(2.0, 0.0, L=1) == 0.0


def test_F2_invalid_L():
    with pytest.raises(ValueError):
        steffen_alder_F2(1.0, 0.0, L=0)


def test_F2_J_u_zero_returns_zero():
    """Aligning J_u = 0 is impossible (no orientation)."""
    assert steffen_alder_F2(0.0, 1.0, L=1) == 0.0


def test_F2_e2_exact_value():
    """E2 (L=2) J_u=2 → J_l=0 has the exact value 2/√10.

    A-class: the Steffen-Alder E2 alignment coefficient for the stretched
    J=2 → J=0 channel is the closed form 2/√10 = 0.6324555320 (Steffen &
    Alder, *At. Data Nucl. Data Tables* (1965); the E1 rows of the same
    table are gated in ``test_F2_J1_known_values``).  Measured
    0.6324555320336752, i.e. machine precision.  The old test asserted
    only ``val != 0`` and ``isfinite``.
    """
    val = steffen_alder_F2(2.0, 0.0, L=2)
    assert pytest.approx(2.0 / math.sqrt(10.0), rel=1e-12) == val, (
        f"F_2(2→0, E2) = {val!r}, exact 2/sqrt(10)")
    # J_u = 0 has no orientation to align: F_2 = 0 for any L (same principle
    # as test_F2_J_u_zero_returns_zero for E1).
    assert steffen_alder_F2(0.0, 2.0, L=2) == 0.0


def test_F2_e2_finite():
    """E2 (L=2) F_2 is finite for all allowed/inelastic channels."""
    val = steffen_alder_F2(2.0, 1.0, L=2)
    assert math.isfinite(val)
