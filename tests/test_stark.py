"""External-field perturbative Stark + Stark-Zeeman tests."""

import numpy as np
import pytest

from atomx.prop.stark import (
    StarkZeemanShift,
    linear_stark_shift_hydrogenic,
    quadratic_stark_shift,
    stark_zeeman_table,
)


def test_linear_stark_h_n2_textbook():
    """H n=2 splitting at F=1e7 V/m: ΔE = (3/2)·n·k·F·a₀ → ±384 GHz."""
    dE_plus = linear_stark_shift_hydrogenic(n=2, k=+1, F_Vpm=1e7, Z=1)
    dE_minus = linear_stark_shift_hydrogenic(n=2, k=-1, F_Vpm=1e7, Z=1)
    assert dE_plus == pytest.approx(-dE_minus, rel=1e-12)
    # 3/2·2·1·1e7/(5.142e11) Hartree → eV → MHz
    expected = 1.5 * 2 * 1 * 1e7 / 5.14220674763e11 * 27.211 * 2.41798934e8
    assert dE_plus == pytest.approx(expected, rel=1e-3)


def test_linear_stark_zero_at_k_zero():
    """k=0 sublevel of any n is unshifted at first order."""
    assert linear_stark_shift_hydrogenic(n=3, k=0, F_Vpm=1e8) == 0.0


def test_linear_stark_invalid():
    with pytest.raises(ValueError):
        linear_stark_shift_hydrogenic(n=0, k=0, F_Vpm=1e8)
    with pytest.raises(ValueError):
        linear_stark_shift_hydrogenic(n=2, k=2, F_Vpm=1e8)  # |k| not < n


def test_linear_stark_z_scaling():
    """Linear Stark scales as 1/Z (a₀ → a₀/Z for Z⁺-like ions)."""
    dE_h = linear_stark_shift_hydrogenic(n=2, k=1, F_Vpm=1e7, Z=1)
    dE_he = linear_stark_shift_hydrogenic(n=2, k=1, F_Vpm=1e7, Z=2)
    assert dE_he == pytest.approx(dE_h / 2.0, rel=1e-12)


def test_quadratic_stark_negative_for_alpha_pos():
    """Positive polarizability → negative shift at any F (force on dipole)."""
    dE = quadratic_stark_shift(alpha_a0_cubed=4.5, F_Vpm=1e6)
    assert dE < 0


def test_quadratic_stark_h_ground():
    """H 1s polarizability is ~4.5 a₀³; shift at F=1e7 V/m of order MHz."""
    dE = quadratic_stark_shift(alpha_a0_cubed=4.5, F_Vpm=1e7)
    assert -10.0 < dE < 0  # ~few MHz negative


def test_quadratic_stark_F_squared_scaling():
    """ΔE ∝ F² — doubling F quadruples the shift."""
    dE_1 = quadratic_stark_shift(4.5, 1e6)
    dE_2 = quadratic_stark_shift(4.5, 2e6)
    assert dE_2 / dE_1 == pytest.approx(4.0, rel=1e-12)


def test_stark_zeeman_table_structure():
    """One entry per (m_J, B, F) tuple."""
    B = np.array([0.5, 1.0])
    F = np.array([0.0, 1e6])
    tbl = stark_zeeman_table(gJ=2.0, J=1.5,
                              alpha_a0_cubed=10.0,
                              B_Tesla_grid=B, F_Vpm_grid=F)
    n_m = int(2 * 1.5 + 1)   # m = -3/2..+3/2 = 4 values
    assert len(tbl) == n_m * len(B) * len(F)
    for entry in tbl:
        assert isinstance(entry, StarkZeemanShift)


def test_stark_zeeman_F_zero_recovers_pure_zeeman():
    """At F = 0, total shift = pure Zeeman."""
    tbl = stark_zeeman_table(gJ=2.0, J=0.5,
                              alpha_a0_cubed=10.0,
                              B_Tesla_grid=np.array([1.0]),
                              F_Vpm_grid=np.array([0.0]))
    for entry in tbl:
        assert entry.delta_E_stark_MHz == 0.0
        assert entry.delta_E_total_MHz == entry.delta_E_zeeman_MHz


def test_stark_zeeman_B_zero_recovers_pure_stark():
    """At B = 0, total shift = pure Stark."""
    tbl = stark_zeeman_table(gJ=2.0, J=0.5,
                              alpha_a0_cubed=10.0,
                              B_Tesla_grid=np.array([0.0]),
                              F_Vpm_grid=np.array([1e7]))
    for entry in tbl:
        assert entry.delta_E_zeeman_MHz == 0.0
        assert entry.delta_E_total_MHz == entry.delta_E_stark_MHz
