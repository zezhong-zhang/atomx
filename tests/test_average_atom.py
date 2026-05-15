"""Average-atom Thomas-Fermi model regression tests."""

import pytest

from atomx.plasma.average_atom import (
    AverageAtomResult,
    average_atom_thomas_fermi,
)


def test_cold_neutral_fe_returns_zero_ionisation():
    """Cold Fe plasma stays neutral."""
    res = average_atom_thomas_fermi(Z=26, T_eV=1.0, ne_cm3=1e15)
    assert isinstance(res, AverageAtomResult)
    assert res.Z_bar == pytest.approx(0.0, abs=0.5)


def test_very_hot_fe_fully_stripped():
    """Fe at T = 5 keV / dilute plasma fully ionises (⟨Z*⟩ ≈ Z)."""
    res = average_atom_thomas_fermi(Z=26, T_eV=5000.0, ne_cm3=1e20)
    assert res.Z_bar == pytest.approx(26.0, abs=0.5)


def test_hot_h_fully_ionised():
    """Hydrogen at T ≫ 13.6 eV fully ionises."""
    res = average_atom_thomas_fermi(Z=1, T_eV=50.0, ne_cm3=1e15)
    assert res.Z_bar == pytest.approx(1.0, abs=0.05)


def test_cold_h_neutral():
    """Hydrogen at T ≪ 13.6 eV stays neutral."""
    res = average_atom_thomas_fermi(Z=1, T_eV=0.5, ne_cm3=1e15)
    assert res.Z_bar == pytest.approx(0.0, abs=0.05)


def test_R_WS_in_atomic_units():
    """R_WS should be in Bohr radii (a₀), of order ~10 a₀ for typical
    plasma densities."""
    res = average_atom_thomas_fermi(Z=10, T_eV=500.0, ne_cm3=1e22)
    assert 1.0 < res.R_ws_au < 100.0


def test_occupations_sum_to_bound_electrons():
    """Σ n̄(n,l) ≈ Z − ⟨Z*⟩."""
    res = average_atom_thomas_fermi(Z=26, T_eV=100.0, ne_cm3=1e22)
    total_occ = sum(res.occupations.values())
    expected = 26 - res.Z_bar
    assert total_occ == pytest.approx(expected, rel=5e-2)


def test_occupations_capped_at_2nsq():
    """No shell exceeds its 2 n² capacity."""
    res = average_atom_thomas_fermi(Z=26, T_eV=10.0, ne_cm3=1e15)
    for (n, l), occ in res.occupations.items():
        cap = 2 * (2 * l + 1)
        assert occ <= cap + 1e-9


def test_invalid_inputs():
    with pytest.raises(ValueError):
        average_atom_thomas_fermi(Z=0, T_eV=100.0, ne_cm3=1e22)
    with pytest.raises(ValueError):
        average_atom_thomas_fermi(Z=10, T_eV=0.0, ne_cm3=1e22)
    with pytest.raises(ValueError):
        average_atom_thomas_fermi(Z=10, T_eV=100.0, ne_cm3=0.0)


def test_z_bar_increases_with_temperature():
    """At fixed ne, ⟨Z*⟩ should be a non-decreasing function of T."""
    Ts = [10.0, 100.0, 1000.0]
    Z_bars = [average_atom_thomas_fermi(Z=10, T_eV=T, ne_cm3=1e21).Z_bar
              for T in Ts]
    assert Z_bars[1] >= Z_bars[0] - 1e-3
    assert Z_bars[2] >= Z_bars[1] - 1e-3
