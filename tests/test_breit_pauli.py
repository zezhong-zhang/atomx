"""Regression tests for Breit-Pauli hydrogenic correction factors."""

import pytest

from atomx.constants import ALPHA
from atomx.physics.breit_pauli import (
    darwin_shift,
    mass_velocity_shift,
    spin_orbit_shifts,
    spin_orbit_xi,
)


ALPHA_SQ = ALPHA * ALPHA


def test_spin_orbit_xi_hydrogen_2p_matches_cowan_factor():
    assert spin_orbit_xi(n=2, l=1, Z_eff=1.0) == pytest.approx(
        ALPHA_SQ / 48.0,
        rel=0.0,
        abs=1e-12,
    )


def test_hydrogen_2p_sommerfeld_splitting():
    xi = spin_orbit_xi(n=2, l=1, Z_eff=1.0)
    E_so_plus, E_so_minus = spin_orbit_shifts(l=1, xi=xi)

    assert E_so_plus - E_so_minus == pytest.approx(
        ALPHA_SQ / 32.0,
        rel=0.0,
        abs=1e-12,
    )


def test_hydrogen_2s12_2p12_breit_pauli_degeneracy():
    Z_eff = 1.0
    two_s12 = (
        mass_velocity_shift(n=2, l=0, Z_eff=Z_eff)
        + darwin_shift(n=2, l=0, Z_eff=Z_eff)
    )

    xi_2p = spin_orbit_xi(n=2, l=1, Z_eff=Z_eff)
    _, two_p12_so = spin_orbit_shifts(l=1, xi=xi_2p)
    two_p12 = (
        mass_velocity_shift(n=2, l=1, Z_eff=Z_eff)
        + darwin_shift(n=2, l=1, Z_eff=Z_eff)
        + two_p12_so
    )

    assert two_p12 - two_s12 == pytest.approx(0.0, rel=0.0, abs=1e-15)


def test_spin_orbit_xi_scales_as_z_four():
    xi_h = spin_orbit_xi(n=2, l=1, Z_eff=1.0)
    xi_ne = spin_orbit_xi(n=2, l=1, Z_eff=10.0)

    assert xi_ne == pytest.approx(xi_h * 10.0**4, rel=1e-14)
