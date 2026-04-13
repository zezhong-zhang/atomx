"""M0 acceptance: CODATA 2018 constants."""

import pytest
from atomx.constants import (
    HARTREE_EV, HARTREE_CM1, FINE_STRUCTURE_CONSTANT,
    SPEED_OF_LIGHT, BOHR_RADIUS_M, ALPHA,
)


def test_hartree_ev_nist():
    """Validate Hartree-to-eV against NIST CODATA 2018."""
    assert HARTREE_EV == pytest.approx(27.211386245988, rel=1e-12)


def test_hartree_cm1_nist():
    assert HARTREE_CM1 == pytest.approx(219474.6313632, rel=1e-10)


def test_fine_structure_constant():
    assert FINE_STRUCTURE_CONSTANT == pytest.approx(7.2973525693e-3, rel=1e-10)


def test_speed_of_light_inverse_alpha():
    """c = 1/alpha in atomic units."""
    assert SPEED_OF_LIGHT == pytest.approx(1.0 / ALPHA, rel=1e-14)


def test_bohr_radius():
    assert BOHR_RADIUS_M == pytest.approx(5.29177210903e-11, rel=1e-10)
